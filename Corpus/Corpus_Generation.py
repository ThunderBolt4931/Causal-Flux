import os
import json
import time
import random
import asyncio
import aiofiles
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# =============================================================================
# Pydantic Models
# =============================================================================

class CompletionModel(BaseModel):
    """Model for completed call fields."""
    domain: str
    intent: str
    reason_for_the_call: str


class PredefinedInteractionDriverModel(BaseModel):
    """Model for predefined interaction drivers."""
    driver: str
    dialogue_pair_index: int


class IdentifiedInteractionDriverModel(BaseModel):
    """Model for identified interaction drivers."""
    keyword: str
    definition: str
    dialogue_pair_index: int


class Metadata(BaseModel):
    """Complete metadata model for call analysis."""
    completed_fields: CompletionModel
    call_summary: str
    outcome: str
    predefined_interaction_drivers: List[PredefinedInteractionDriverModel] = []
    identified_interaction_drivers: List[IdentifiedInteractionDriverModel] = []


# =============================================================================
# Transcript Processor
# =============================================================================

class GeminiTranscriptProcessor:
    """Handles async processing of call transcripts using Gemini API."""
    
    def __init__(self, api_key: str, system_prompt: str, output_file: str):
        """
        Initialize the processor.
        
        Args:
            api_key: Google Gemini API key
            system_prompt: System prompt for the LLM
            output_file: Path to output JSONL file
        """
        self.client = genai.Client(api_key=api_key)
        self.system_prompt = system_prompt
        self.output_file = output_file
        self.write_lock = asyncio.Lock()
    
    @staticmethod
    def build_call_string(call: Dict[str, Any]) -> str:
        """
        Convert call dictionary to formatted string for LLM processing.
        
        Args:
            call: Call transcript dictionary
            
        Returns:
            Formatted string representation of the call
        """
        # Build header
        header = (
            f"domain: {call.get('domain', '')}\n"
            f"intent: {call.get('intent', '')}\n"
            f"reason_for_call: {call.get('reason_for_call', '')}\n\n"
        )
        
        # Build turns
        turns = []
        for turn_idx, turn in enumerate(call.get("turns", [])):
            # Join all utterances in this turn
            utterances = " | ".join(
                f"{u['speaker']}: {u['utterance']}"
                for u in turn.get("conversation", [])
            )
            
            # Add sentiment information
            sentiment = turn.get("sentiment", {})
            sentiment_str = (
                f"[sentiment_score={sentiment.get('score')} "
                f"label={sentiment.get('label')}]"
            )
            
            turns.append(f"turn {turn_idx}: {utterances} {sentiment_str}")
        
        return header + "\n".join(turns)
    
    async def process_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single call transcript using Gemini API.
        
        Args:
            call: Call transcript dictionary
            
        Returns:
            Processed call with metadata
            
        Raises:
            ValueError: If API response is invalid
        """
        call_string = self.build_call_string(call)
        
        response = await self.client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=call_string,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                response_schema=Metadata.model_json_schema(),
                temperature=0.1
            )
        )
        
        if not response.parsed:
            raise ValueError("Empty or invalid parsed response from Gemini API")
        
        # Build output object
        return {
            "transcript_id": call.get("transcript_id", ""),
            "domain": call.get("domain", ""),
            "intent": call.get("intent", ""),
            "reason_for_call": call.get("reason_for_call", ""),
            "turns": call.get("turns", []),
            "metadata": response.parsed
        }
    
    async def append_result(self, result: Dict[str, Any]) -> None:
        """
        Safely append result to output file (JSONL format).
        
        Args:
            result: Processed call result to write
        """
        async with self.write_lock:
            async with aiofiles.open(self.output_file, "a", encoding="utf-8") as f:
                json_line = json.dumps(result, ensure_ascii=False) + "\n"
                await f.write(json_line)


# =============================================================================
# Worker Pool
# =============================================================================

class WorkerPool:
    """Manages async workers for parallel processing."""
    
    def __init__(
        self,
        processor: GeminiTranscriptProcessor,
        num_workers: int = 4
    ):
        """
        Initialize the worker pool.
        
        Args:
            processor: GeminiTranscriptProcessor instance
            num_workers: Number of concurrent workers
        """
        self.processor = processor
        self.num_workers = num_workers
        self.queue = asyncio.Queue()
    
    async def worker(self, name: str) -> None:
        """
        Worker coroutine that processes calls from queue.
        
        Args:
            name: Worker identifier for logging
        """
        while True:
            call = await self.queue.get()
            
            # Sentinel value to stop worker
            if call is None:
                self.queue.task_done()
                break
            
            try:
                result = await self.processor.process_call(call)
                await self.processor.append_result(result)
                print(f"[{name}] ✓ Completed {call.get('transcript_id')}")
            except Exception as e:
                print(f"[{name}] ✗ ERROR on {call.get('transcript_id')}: {e}")
            
            self.queue.task_done()
    
    async def process_all(self, calls: List[Dict[str, Any]]) -> None:
        """
        Process all calls using worker pool.
        
        Args:
            calls: List of call transcripts to process
        """
        # Enqueue all calls
        for call in calls:
            await self.queue.put(call)
        
        # Create worker tasks
        workers = [
            asyncio.create_task(self.worker(f"Worker-{i+1}"))
            for i in range(self.num_workers)
        ]
        
        # Add sentinel values to stop workers
        for _ in range(self.num_workers):
            await self.queue.put(None)
        
        # Wait for all tasks to complete
        await self.queue.join()
        
        # Wait for workers to finish
        await asyncio.gather(*workers)


# =============================================================================
# Main Pipeline
# =============================================================================

class Pipeline:
    """Main pipeline for processing transcripts."""
    
    def __init__(
        self,
        api_key: str,
        transcript_file: str,
        sysprompt_file: str,
        output_file: str,
        num_workers: int = 4,
        limit: Optional[int] = None,
        shuffle: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            api_key: Google Gemini API key
            transcript_file: Path to input JSON file with transcripts
            sysprompt_file: Path to system prompt text file
            output_file: Path to output JSONL file
            num_workers: Number of concurrent workers
            limit: Maximum number of transcripts to process (None for all)
            shuffle: Whether to shuffle transcripts before processing
        """
        self.transcript_file = transcript_file
        self.sysprompt_file = sysprompt_file
        self.output_file = output_file
        self.num_workers = num_workers
        self.limit = limit
        self.shuffle = shuffle
        
        # Load system prompt
        with open(sysprompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        
        # Initialize processor
        self.processor = GeminiTranscriptProcessor(
            api_key=api_key,
            system_prompt=system_prompt,
            output_file=output_file
        )
    
    def load_transcripts(self) -> List[Dict[str, Any]]:
        """
        Load and prepare transcripts for processing.
        
        Returns:
            List of transcript dictionaries
        """
        with open(self.transcript_file, "r", encoding="utf-8") as f:
            transcripts = json.load(f)
        
        if self.shuffle:
            random.shuffle(transcripts)
        
        if self.limit:
            transcripts = transcripts[:self.limit]
        
        return transcripts
    
    async def run(self) -> None:
        """Execute the complete pipeline."""
        print(f"{'='*60}")
        print(f"Gemini Transcript Processor")
        print(f"{'='*60}")
        print(f"Input: {self.transcript_file}")
        print(f"Output: {self.output_file}")
        print(f"Workers: {self.num_workers}")
        
        # Load transcripts
        print(f"\nLoading transcripts...")
        transcripts = self.load_transcripts()
        print(f"Loaded {len(transcripts)} transcripts")
        
        # Clear output file
        open(self.output_file, "w").close()
        
        # Create worker pool
        worker_pool = WorkerPool(self.processor, self.num_workers)
        
        # Process all transcripts
        print(f"\nStarting processing at {time.strftime('%X')}...\n")
        start_time = time.time()
        
        await worker_pool.process_all(transcripts)
        
        elapsed_time = time.time() - start_time
        
        # Summary
        print(f"\n{'='*60}")
        print(f"✓ Processing Complete!")
        print(f"{'='*60}")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Average: {elapsed_time/len(transcripts):.2f}s per transcript")
        print(f"Output saved to: {self.output_file}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main execution function."""
    
    # Configuration
    API_KEY = os.environ.get('GEMINI_API_KEY', 'your-api-key-here')
    TRANSCRIPT_FILE = 'Default_corpus.json'
    SYSPROMPT_FILE = 'system_prompt.txt'
    TEMP_OUTPUT_FILE = 'processed_transcripts.jsonl'
    NUM_WORKERS = 5
    LIMIT = 1000  # Set to None to process all
    
    # Create and run pipeline
    pipeline = Pipeline(
        api_key=API_KEY,
        transcript_file=TRANSCRIPT_FILE,
        sysprompt_file=SYSPROMPT_FILE,
        output_file=TEMP_OUTPUT_FILE,
        num_workers=NUM_WORKERS,
        limit=LIMIT,
        shuffle=True
    )
    
    asyncio.run(pipeline.run())
    
    
def rephraser():
    input_file = "processed_transcripts.jsonl"
    output_file = "corpus.json"

    items = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(items, out, indent=2, ensure_ascii=False)

    print("Done! Saved to", output_file)

if __name__ == "__main__":
    main()
    rephraser()
    