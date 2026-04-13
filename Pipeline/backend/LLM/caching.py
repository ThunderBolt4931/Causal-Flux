import os
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Dict, Any

load_dotenv()
MAX_HISTORY_EXCHANGES = 5
MAX_MESSAGES_IN_HISTORY = MAX_HISTORY_EXCHANGES * 2

try:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL or SUPABASE_KEY not found in .env file")
    supabase: Client = create_client(url, key)
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    exit(1)

def insert_user_message(user_prompt: str):
    """
    Inserts only the user's message into the database.
    """
    try:
        supabase.table("chat_history").insert({
            "role": "user",
            "content": user_prompt
        }).execute()
    except Exception as e:
        print(f"Error inserting user message to Supabase: {e}")

def get_chat_history() -> List[Dict[str, Any]]:
    """
    Fetches the last MAX_MESSAGES_IN_HISTORY from Supabase
    and returns them in the correct chronological order (oldest first).
    """
    try:
        response = supabase.table("chat_history") \
            .select("role", "content") \
            .order("created_at", desc=True) \
            .limit(MAX_MESSAGES_IN_HISTORY) \
            .execute()
        
        if response.data:
            return list(reversed(response.data))
        return []

    except Exception as e:
        print(f"Error fetching history from Supabase: {e}")
        return []
    
def insert_assistant_message(response: str):
    """
    Inserts only the assistant's message into the database.
    """
    try:
        supabase.table("chat_history").insert({
                "role": "assistant",
                "content": response
            }).execute()
    except Exception as e:
        print(f"Error inserting assistant message to Supabase: {e}")

def clear_chat_history():
    """Helper to clear history at the start of a new complex query"""
    if not supabase: return
    try:
        # Delete all rows (unsafe for prod, fine for local pipeline testing)
        supabase.table("chat_history").delete().neq("id", 0).execute() 
    except Exception:
        pass