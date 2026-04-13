import os
import tiktoken
import asyncio
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

try:
    from Rags_and_Graphs.build_graph import CLUSTER_PPR
    from Rags_and_Graphs.reranker import rerank_documents
    from langchain_core.documents import Document
    from LLM.model import run_llm, run_llm_stream
    from Rephraser.query_router import route_query
    from Rephraser.sub_query_router import process_query_with_linear_context
    from Rephraser.splitter import split_query
    from Rephraser.intent_identifier import classify_intents
    from LLM.caching import insert_assistant_message, insert_user_message, get_chat_history
    from Plots.plot_generator import (
        generate_intents_bar_chart,
        generate_frequency_chart,
        generate_cluster_pie,
        generate_bubble_chart
    )
except ImportError as e:
    raise ImportError(
        f"Critical Error: Missing custom modules. Ensure 'Rags_and_Graphs', 'Rephraser', 'LLM', and 'Plots' folders are present. Details: {e}")

load_dotenv()

app = FastAPI(title="RAG Chatbot API", version="1.4.0",
              description="Backend API for RAG-Powered Chatbot with Visualizations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(base_dir, 'final_dataset.json')
CLUSTERINGS_FILE = os.path.join(base_dir, 'Rags_and_Graphs', 'clustered_transcripts.json')
DATASET_CACHE: Optional[List[Dict]] = None

def load_dataset() -> List[Dict]:
    """Load dataset from disk once and cache in memory"""
    global DATASET_CACHE
    if DATASET_CACHE is None:
        print("[CACHE] Loading dataset into memory...")
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            DATASET_CACHE = json.load(f)
        print(f"[CACHE] Dataset loaded: {len(DATASET_CACHE)} records") #type:ignore
    return DATASET_CACHE #type: ignore

NEO4J_URI2 = os.getenv("NEO4J_URI2")
NEO4J_USER2 = os.getenv("NEO4J_USER2")
NEO4J_PASSWORD2 = os.getenv("NEO4J_PASSWORD2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_CHOICES = {
    "OpenAI GPT-4o": {"model": "gpt-4o", "provider": "openai"},
    "OpenAI GPT-4o-mini": {"model": "gpt-4o-mini", "provider": "openai"},
    "Claude 3.5 Sonnet": {"model": "claude-3-5-sonnet-20241022", "provider": "anthropic"},
    "Claude 3 Opus": {"model": "claude-3-opus-20240229", "provider": "anthropic"},
    "Claude 3 Haiku": {"model": "claude-3-haiku-20240307", "provider": "anthropic"},
    "Groq Llama 3.3 70B": {"model": "llama-3.3-70b-versatile", "provider": "groq"},
    "Groq Llama 3.1 8B": {"model": "llama-3.1-8b-instant", "provider": "groq"},
    "Groq GPT OSS": {"model": "openai/gpt-oss-120b", "provider": "groq"},
    "Gemini 2.5 Pro": {"model": "gemini-2.5-pro", "provider": "gemini"},
    "Gemini 2.0 Flash": {"model": "gemini-2.5-flash", "provider": "gemini"},
}

def get_api_key_for_provider(provider: str) -> str:
    if provider == 'openai':
        return OPENAI_API_KEY  # type: ignore
    elif provider == 'anthropic':
        return ANTHROPIC_API_KEY  # type: ignore
    elif provider == 'groq':
        return GROQ_API_KEY  # type: ignore
    elif provider == 'gemini':
        return GEMINI_API_KEY  # type: ignore
    else:
        raise ValueError(f"Unknown provider: {provider}")

def Cluster_ppr(query: str, file_path: str):
    rag = CLUSTER_PPR(NEO4J_URI2, NEO4J_USER2,
                      NEO4J_PASSWORD2, data_file=file_path)
    results = rag.query(query, top_k=20)
    content_results = []
    meta_info = None
    for r in results:
        if "retrieved_count" in r:
            meta_info = r
        else:
            content_results.append(r)

    documents = []
    for r in content_results:
        content = r.get('preview', '')
        if content:
            documents.append(Document(page_content=str(content), metadata=r))

    reranked_docs = rerank_documents(query, documents, final_k=10)
    final_results = [doc.metadata for doc in reranked_docs]
    if meta_info:
        final_results.append(meta_info)

    return final_results

def get_transcript_data(transcript_id: str):
    """Helper to find a specific transcript in the dataset (uses cache)"""
    try:
        data = load_dataset()  
        for item in data:
            curr_id = item.get('transcript_id') or item.get('id') or item.get('filename')
            if str(curr_id) == str(transcript_id):
                return item
        return None
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return None

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's query")
    model_choice: str = Field(
        default="OpenAI GPT-4o-mini", description="Key from MODEL_CHOICES")
    task_mode: str = Field(
        default="task1", description="Task mode: 'task1' for causal inquiry (no history), 'task2' for follow-up (with history)")

class ChatResponse(BaseModel):
    response: str
    metadata: Dict[str, Any]

class VisualizationRequest(BaseModel):
    transcript_ids: List[str] = Field(..., description="List of transcript IDs from RAG results")
    drivers: List[str] = Field(default=[], description="List of interaction drivers (optional, will be classified from query)")
    query_text: str = Field(..., description="Original query text for intent classification")

@app.get("/", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "active", "service": "RAG Chatbot Backend"}

@app.get("/config/models", tags=["Configuration"])
async def get_models():
    """List available LLM models."""
    return {"models": MODEL_CHOICES}

@app.get("/transcript/{transcript_id}", tags=["Chat"])
async def get_transcript(transcript_id: str):
    """Fetch details for a specific transcript ID."""
    data = get_transcript_data(transcript_id)
    if not data:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return data

@app.post("/api/visualizations", tags=["Visualizations"])
async def generate_visualizations(request: VisualizationRequest):
    """
    Generate visualization plots for query results.
    Uses classify_intents() to identify query intents for plotting.
    Returns base64-encoded PNG images.
    """
    try:
        query_intents = []
        if request.query_text:
            print(f"[VISUALIZATION] Classifying intents for query: {request.query_text[:50]}...")
            query_intents = classify_intents(request.query_text)
            print(f"[VISUALIZATION] Classified intents: {query_intents}")
        print(f"[VISUALIZATION] Request received. Transcript IDs: {len(request.transcript_ids)}, Classified Intents: {query_intents}")
        
        plots = {}

        if request.transcript_ids and query_intents:
            print("[VISUALIZATION] Generating intent/frequency plots...")
            plots['intents_bar'] = generate_intents_bar_chart(
                DATA_FILE, query_intents, request.transcript_ids
            )
            plots['frequency_bar'] = generate_frequency_chart(
                DATA_FILE, query_intents, request.transcript_ids
            )
        if request.transcript_ids:
            print("[VISUALIZATION] Generating cluster/bubble plots...")
            plots['cluster_pie'] = generate_cluster_pie(
                CLUSTERINGS_FILE, request.transcript_ids
            )
            plots['bubble_chart'] = generate_bubble_chart(
                CLUSTERINGS_FILE, request.transcript_ids
            )

        print(f"[VISUALIZATION] Generated plots: {list(plots.keys())}")
        return {"plots": plots, "query_intents": query_intents, "success": True}
    
    except Exception as e:
        print(f"Visualization Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate plots: {str(e)}")

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint with Task-1 and Task-2 modes:
    
    Task-1 (Initial Causal Inquiry - No follow-up assumed):
    - Single-hop: router -> if RAG -> cluster_ppr -> run_llm (NO history)
    - Multi-hop: splitter -> sub-queries -> sub_query_router -> final summary
    - History is NOT used, each query is independent
    
    Task-2 (Contextual Follow-up - History enabled):
    - Same logic but history is ALWAYS used
    - Outputs are saved and used for follow-up queries
    """
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"[NEW REQUEST] Query: {request.message[:50]}...")
    print(f"[CONFIG] Model: {request.model_choice} | Task Mode: {request.task_mode}")
    print(f"{'='*60}")
    
    try:
        # Validate model
        model_info = MODEL_CHOICES.get(request.model_choice)
        if not model_info:
            available = list(MODEL_CHOICES.keys())
            raise HTTPException(status_code=400, detail=f"Invalid model_choice. Available: {available}")

        model_name = model_info["model"]
        provider = model_info["provider"]
        api_key = get_api_key_for_provider(provider)

        if not api_key:
            raise HTTPException(status_code=500, detail=f"API Key missing for provider: {provider}")

        # CORE DIFFERENCE: Task-1 = no history, Task-2 = history enabled
        is_task1 = request.task_mode == "task1"
        use_history = not is_task1  # Task-1: False, Task-2: True
        
        print(f"[TASK MODE] {'TASK-1 (No History)' if is_task1 else 'TASK-2 (History Enabled)'}")

        token_count = 0
        retrieved_ids = []
        query_drivers = []
        final_response_text = ""

        # Step 1: Route query to determine if RAG is needed
        print("[ROUTER] Checking if RAG is needed...")
        route_decision = route_query(request.message)
        action = route_decision.get("action", "RAG")
        print(f"[ROUTER] Decision: {action}")
        print(f"[LOG] Task-1 (No History) vs Task-2 (History) logic active. Messages will be saved to DB in both cases.")

        if action == "RAG":
            # Step 2: Check if multi-hop (query splitter)
            print("[SPLITTER] Checking for sub-queries...")
            sub_queries_struct = split_query(request.message)
            
            if isinstance(sub_queries_struct, list) and len(sub_queries_struct) > 0:
                sub_queries = [item.get('text', item) if isinstance(item, dict) else item for item in sub_queries_struct]
            else:
                sub_queries = [request.message]
            
            print(f"[SPLITTER] Sub-queries: {sub_queries}")

            if len(sub_queries) > 1:
                # ============ MULTI-HOP FLOW ============
                print(f"[MULTI-HOP] Processing {len(sub_queries)} sub-queries...")
                accumulated_context = ""
                
                for i, sub_q in enumerate(sub_queries):
                    print(f"\n{'='*40}")
                    print(f"  [HOP {i+1}/{len(sub_queries)}] Sub-query: {sub_q[:]}...")
                    print(f"  [LOG] Processing sub-query {i+1} for Task Mode: {request.task_mode}")
                    
                    # Always save sub-query to history (for potential future Task-2 context)
                    print(f"  [DB] Saving User sub-query to DB (ID: {sub_q[:]}...)")
                    insert_user_message(sub_q)
                    
                    need_retrieval = False
                    search_query = sub_q

                    if i == 0:
                        # First sub-query: ALWAYS use RAG
                        print("    [SUB-ROUTER] First hop -> Forcing RAG retrieval")
                        need_retrieval = True
                    else:
                        # Other sub-queries: Use sub_query_router
                        print("    [SUB-ROUTER] Checking if context is sufficient...")
                        history_objs = get_chat_history() if use_history else []
                        last_queries = [h['content'] for h in history_objs if h['role'] == 'user'][-3:]
                        
                        router_out = process_query_with_linear_context(last_queries, sub_q)
                        
                        if router_out.get("doable"):
                            # Can answer from previous context
                            print("    [SUB-ROUTER] Context sufficient -> Using previous output")
                            print(f"    [LOG] Skipping RAG. Using context from previous hops.")
                            sub_ans = router_out.get("answer", "")
                            if sub_ans:
                                print(f"  [DB] Saving Assistant sub-answer to DB...")
                                insert_assistant_message(sub_ans)
                                accumulated_context += f"\nQ{i+1}: {sub_q}\nA{i+1}: {sub_ans}\n"
                                continue  # Skip retrieval
                            else:
                                print("    [SUB-ROUTER] Empty answer -> Fallback to RAG")
                                need_retrieval = True
                        else:
                            # Need fresh retrieval
                            print("    [SUB-ROUTER] Need retrieval for this sub-query")
                            need_retrieval = True
                            search_query = router_out.get("retrieval_query", sub_q)

                    if need_retrieval:
                        print(f"    [RAG] Retrieving context for: {search_query[:]}...")
                        print(f"    [LOG] Performing Cluster PPR search...")
                        context_content = Cluster_ppr(query=search_query, file_path=DATA_FILE)
                        
                        if isinstance(context_content, list):
                            found_ids = [item.get('transcript_id') for item in context_content if 'transcript_id' in item]
                            retrieved_ids.extend(found_ids)
                            print(f"    [RAG] Found {len(found_ids)} transcripts")
                            for item in context_content:
                                if 'drivers' in item: 
                                    query_drivers.extend(item['drivers'])
                        
                        sub_prompt = (
                            f"{context_content}\n"
                            f"> *Sub-Question:* {sub_q}\n"
                            "> *Provide a focused answer based on the context.*"
                        )
                        
                        print("    [LLM] Generating sub-answer...")
                        sub_ans = run_llm(
                            user_query_for_db=sub_prompt,
                            model_name=model_name,
                            need_chat_history=use_history,  
                            api_key=api_key,
                            provider=provider,
                            temperature=0.3
                        )
                        
                        print(f"  [DB] Saving Assistant sub-answer to DB...")
                        insert_assistant_message(sub_ans)
                        
                        accumulated_context += f"\nQ{i+1}: {sub_q}\nA{i+1}: {sub_ans}\n"
                
                print("\n[SYNTHESIS] Combining all sub-answers into final response...")
                final_prompt = (
                    f"Based on the following intermediate Q&A, provide a comprehensive final answer.\n\n"
                    f"Original Query: {request.message}\n\n"
                    f"Intermediate Steps:\n{accumulated_context}\n\n"
                    f"Synthesize a direct, complete answer citing relevant Transcript IDs."
                )
                
                final_response_text = run_llm(
                    user_query_for_db=final_prompt,
                    model_name=model_name,
                    need_chat_history=False,
                    api_key=api_key,
                    provider=provider,
                    temperature=0.3
                )

                print(f"  [DB] Saving Final Response to DB...")
                insert_assistant_message(final_response_text)

            else:
                # ============ SINGLE-HOP FLOW ============
                print("[SINGLE-HOP] Processing single query...")
                print(f"[LOG] Task Mode: {request.task_mode}. History for Generation: {use_history}")
                
                print(f"  [DB] Saving User query to DB...")
                insert_user_message(request.message)
                
                context_content = Cluster_ppr(file_path=DATA_FILE, query=request.message)

                if isinstance(context_content, list):
                    retrieved_ids = [item.get('transcript_id') for item in context_content if 'transcript_id' in item]
                    print(f"  [RAG] Found {len(retrieved_ids)} transcripts")
                    for item in context_content:
                        if 'drivers' in item:
                            query_drivers.extend(item['drivers'])
                
                final_prompt = (
                    f"{context_content}\n"
                    f"> *User Question:* {request.message}\n"
                    "> *IMPORTANT: Answer based on context. Cite Transcript IDs.*"
                )

                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    token_count = len(encoding.encode(final_prompt))
                except Exception:
                    token_count = 0

                print(f"  [LLM] Generating response (history={'ON' if use_history else 'OFF'})...")
                final_response_text = run_llm(
                    user_query_for_db=final_prompt,
                    model_name=model_name,
                    need_chat_history=use_history,  
                    api_key=api_key,
                    provider=provider,
                    temperature=0.3
                )
                
                print(f"  [DB] Saving Final Response to DB...")
                insert_assistant_message(final_response_text)

            retrieved_ids = list(set(retrieved_ids))
            query_drivers = list(set(query_drivers))

        else:
            print("[NON-RAG] Direct LLM response...")
            print(f"[LOG] Skipping RAG pipeline. Routing directly to LLM.")
            
            insert_user_message(request.message)
            
            final_response_text = run_llm(
                user_query_for_db=request.message,
                model_name=model_name,
                need_chat_history=use_history,
                api_key=api_key,
                provider=provider,
                temperature=0.7
            )
            
            print(f"  [DB] Saving Final Response to DB...")
            insert_assistant_message(final_response_text)

        print(f"\n[COMPLETE] Response generated successfully")
        
        metadata = {
            "provider": provider.upper(),
            "model": model_name,
            "route": action,
            "tokens": token_count,
            "source": "CLUSTER_PPR",
            "task_mode": request.task_mode,
            "history_used": use_history,
            "transcript_ids": retrieved_ids,
            "query_drivers": query_drivers,
            "query_text": request.message
        }

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n[PERFORMANCE] Total Query Execution Time: {elapsed_time:.4f} seconds")
        metadata["execution_time"] = round(elapsed_time, 4)

        return ChatResponse(response=final_response_text, metadata=metadata)

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(request: ChatRequest):
    """
    Process a user query and stream the LLM response in real-time using Server-Sent Events.
    """
    try:
        model_info = MODEL_CHOICES.get(request.model_choice)
        if not model_info:
            available = list(MODEL_CHOICES.keys())
            raise HTTPException(
                status_code=400, detail=f"Invalid model_choice. Available: {available}")
 
        model_name = model_info["model"]
        provider = model_info["provider"]
        api_key = get_api_key_for_provider(provider)

        if not api_key:
            raise HTTPException(
                status_code=500, detail=f"API Key missing for provider: {provider}")
        is_task1 = request.task_mode == "task1"
        need_chat_history = not is_task1
        route_decision = route_query(request.message)
        action = route_decision.get("action", "RAG")

        token_count = 0
        final_user_prompt = request.message
        retrieved_ids = []
        query_drivers = []

        if action == "RAG":
            context_content = Cluster_ppr(
                file_path=DATA_FILE, query=request.message)
            if isinstance(context_content, list):
                retrieved_ids = [item.get('transcript_id') for item in context_content if 'transcript_id' in item]
                for item in context_content:
                    if 'drivers' in item:
                        query_drivers.extend(item['drivers'])
                query_drivers = list(set(query_drivers))

            final_user_prompt = (
                f"{context_content}\n"
                f"> *User Question:* {request.message}\n"
                "> *Answer the question based on the context provided.*"
            )

            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens = encoding.encode(final_user_prompt)
                token_count = len(tokens)
            except Exception:
                token_count = 0
        async def generate():
            accumulated_text = ""
            try:
                stream_gen, messages_sent = run_llm_stream(
                    user_query_for_db=final_user_prompt,
                    model_name=model_name,
                    need_chat_history=need_chat_history,
                    api_key=api_key,
                    provider=provider
                )
                for chunk in stream_gen:
                    accumulated_text += chunk
                    yield f"data: {{\"type\":\"content\",\"text\":{json.dumps(chunk)}}}\n\n"
                    await asyncio.sleep(0.01)
                
                insert_assistant_message(accumulated_text)
                metadata = {
                    "provider": provider.upper(),
                    "model": model_name,
                    "route": action,
                    "tokens": token_count,
                    "source": "CLUSTER PPR",
                    "task_mode": request.task_mode,
                    "history_used": need_chat_history,
                    "transcript_ids": retrieved_ids,
                    "query_drivers": query_drivers,
                    "query_text": request.message
                }
                
                yield f"data: {{\"type\":\"metadata\",\"metadata\":{json.dumps(metadata)}}}\n\n"
                yield "data: {\"type\":\"done\"}\n\n"
                
            except Exception as e:
                yield f"data: {{\"type\":\"error\",\"error\":{json.dumps(str(e))}}}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)