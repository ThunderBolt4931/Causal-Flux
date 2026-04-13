import json
import os
import ast
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


class TranscriptProcessor:
    """Handles loading and processing of transcript data."""
    
    def __init__(self, transcript_path: str):
        """
        Initialize the processor with transcript data.
        
        Args:
            transcript_path: Path to the JSON file containing transcripts
        """
        self.transcript_path = transcript_path
        self.all_data = self._load_transcripts()
    
    def _load_transcripts(self) -> Dict[str, Any]:
        """Load transcript data from JSON file."""
        with open(self.transcript_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return {d["transcript_id"]: d for d in data}
    
    def get_context(self, transcript_ids: List[str]) -> str:
        """
        Extract call summaries for given transcript IDs.
        
        Args:
            transcript_ids: List of transcript IDs to retrieve
            
        Returns:
            Concatenated call summaries
        """
        summaries = []
        for tid in transcript_ids:
            if tid in self.all_data:
                metadata = self.all_data[tid].get("metadata", {})
                if "call_summary" in metadata:
                    summaries.append(metadata["call_summary"])
        
        return "\n".join(summaries)


class DataCleaner:
    """Utilities for cleaning and normalizing data."""
    
    @staticmethod
    def clean_tid_list(value: Any) -> List:
        """
        Convert various input types to a clean list of transcript IDs.
        
        Args:
            value: Input value (string, list, int, float, or NaN)
            
        Returns:
            List of transcript IDs
        """
        # Already a list
        if isinstance(value, list):
            return value
        
        # NaN or None
        if pd.isna(value):
            return []
        
        # Numeric types
        if isinstance(value, (int, float)):
            return [int(value)]
        
        # String types
        if isinstance(value, str):
            try:
                value = value.strip()
                
                # List-like string
                if value.startswith("[") and value.endswith("]"):
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list):
                        return parsed
                
                # Numeric string
                if value.isdigit():
                    return [int(value)]
                
                return [value]
            except (ValueError, SyntaxError):
                return [value]
        
        return []


class FollowUpGenerator:
    """Generates follow-up questions using OpenAI's API."""
    
    def __init__(self, api_key: str, model: str = "gpt-o4"):
        """
        Initialize the generator with API credentials.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use (default: gpt-4)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_followups(
        self, 
        context: str, 
        original_question: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate follow-up questions with expected answers.
        
        Args:
            context: Call transcript context
            original_question: The original question to build upon
            
        Returns:
            Dictionary containing list of follow-up Q&A pairs
        """
        prompt = self._build_prompt(context, original_question)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return json.loads(response.choices[0].message.content)
    
    @staticmethod
    def _build_prompt(context: str, first_question: str) -> str:
        """Build the prompt for the LLM."""
        return f"""
You are an expert at generating **follow-up causal reasoning questions** from call transcripts.

You have already generated the FIRST QUESTION:
"{first_question}"

YOUR TASK NOW:
1. Generate **1 to 3 follow-up questions**.
2. For EACH follow-up question, also generate an **expected_answer** grounded in the call context.
3. Each follow-up question must be:
   - multi-hop (2–3 step causal chain)
   - grounded strictly in the call context
   - generalized (do NOT include specific names, times, or numbers)
   - not a rephrasing of the first question
   - not using repetitive phrasing such as:
       • "What caused the customer's frustration"
       • "What caused the agent's frustration"
       • "What event triggered…"
       • Any close variant of the above
4. Each question must explore a **different reasoning angle**.
5. Answers must be:
   - concise
   - multi-step causal
   - directly grounded in the call content
   - not speculative
   - minimal paraphrasing allowed

OUTPUT FORMAT (STRICT):
{{
  "follow_ups": [
    {{ "question": "<q1>", "expected_answer": "<a1>" }},
    {{ "question": "<q2>", "expected_answer": "<a2>" }},
    {{ "question": "<q3>", "expected_answer": "<a3>" }}
  ]
}}

NOTES:
- If the context only supports 1 or 2 follow-ups, output fewer.
- Do NOT include explanations outside JSON.
- Keep questions human-sounding and short.

Context:
{context}
"""


class Pipeline:
    """Main pipeline for processing datasets and generating follow-ups."""
    
    def __init__(
        self,
        transcript_processor: TranscriptProcessor,
        generator: FollowUpGenerator,
        max_questions: int = 3
    ):
        """
        Initialize the pipeline.
        
        Args:
            transcript_processor: TranscriptProcessor instance
            generator: FollowUpGenerator instance
            max_questions: Maximum number of follow-up questions to generate
        """
        self.processor = transcript_processor
        self.generator = generator
        self.max_questions = max_questions
        self.cleaner = DataCleaner()
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> pd.DataFrame:
        """
        Process a dataframe and generate follow-up questions.
        
        Args:
            df: Input dataframe with 't_id' and 'Question' columns
            output_path: Path to save the output CSV
            
        Returns:
            Processed dataframe with follow-up questions
        """
        # Clean transcript IDs
        df['t_id'] = df['t_id'].apply(self.cleaner.clean_tid_list)
        
        qa_rows = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            try:
                row_data = self._process_row(row)
                qa_rows.append(row_data)
            except Exception as e:
                print(f"Error processing t_id {row.get('t_id', 'Unknown')}: {e}")
                continue
        
        result_df = pd.DataFrame(qa_rows)
        result_df.to_csv(output_path, index=False)
        
        return result_df
    
    def _process_row(self, row: pd.Series) -> Dict[str, Any]:
        """Process a single row and generate follow-ups."""
        context = self.processor.get_context(row['t_id'])
        response = self.generator.generate_followups(context, row['Question'])
        followups = response.get("follow_ups", [])
        
        # Initialize row data
        row_data = {
            "t_id": row["t_id"],
            "original_question": row["Question"],
        }
        
        # Pre-fill columns
        for k in range(1, self.max_questions + 1):
            row_data[f"follow_up_{k}"] = None
            row_data[f"answer_{k}"] = None
        
        # Fill with generated data
        for i, qa in enumerate(followups[:self.max_questions], start=1):
            row_data[f"follow_up_{i}"] = qa.get("question")
            row_data[f"answer_{i}"] = qa.get("expected_answer")
        
        return row_data


def load_json_as_dataframe(json_path: str) -> pd.DataFrame:
    """
    Load JSON file and convert to DataFrame.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        DataFrame representation of the JSON data
    """
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data)


def main():
    """Main execution function."""
    
    # Configuration
    TRANSCRIPT_PATH = 'mergedout.json'
    INPUT_PATH = 'input_questions.csv'
    OUTPUT_PATH = 'follow_up_questions.csv'
    API_KEY = os.environ.get('OPENAI_API_KEY', 'your-api-key-here')
    
    print("Initializing components...")
    
    # Initialize components
    transcript_processor = TranscriptProcessor(TRANSCRIPT_PATH)
    generator = FollowUpGenerator(api_key=API_KEY, model="gpt-o4")
    pipeline = Pipeline(transcript_processor, generator, max_questions=3)
    
    print(f"Loading input data from: {INPUT_PATH}")
    
    # Load input data (supports both CSV and JSON)
    if INPUT_PATH.endswith('.json'):
        df = load_json_as_dataframe(INPUT_PATH)
    else:
        df = pd.read_csv(INPUT_PATH)
    
    print(f"Processing {len(df)} rows...")
    
    # Process and save
    result_df = pipeline.process_dataframe(df, OUTPUT_PATH)
    
    print(f"\n✓ Complete! Generated {len(result_df)} rows with follow-up questions")
    print(f"✓ Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()