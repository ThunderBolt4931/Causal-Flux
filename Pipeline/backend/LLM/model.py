import os
import json
from groq import Groq
from google.genai import types, Client
import tiktoken
from openai import OpenAI
from anthropic import Anthropic
from typing import Optional, Iterator, Generator
from LLM.caching import get_chat_history, insert_user_message, insert_assistant_message

base_dir = os.path.dirname(os.path.abspath(__file__))
SYSPROMPT=os.path.join(base_dir, "sysprompt.txt")

with open(SYSPROMPT, "r") as f:
    sysprompt_content = f.read()
    
SYSTEM_MESSAGE = {"role": "system", "content": sysprompt_content}


def detect_provider(model_name: str) -> str:
    """
    Detect LLM provider based on model name.
    Returns: 'openai', 'anthropic', 'gemini', or 'groq'
    """
    model_lower = model_name.lower()
    
    if model_lower.startswith('gpt-') or model_lower.startswith('o1-'):
        print("openai")
        return 'openai'
    elif model_lower.startswith('claude-'):
        return 'anthropic'
    elif model_lower.startswith('gemini-'):
        return 'gemini'
    else:
        return 'groq'


def run_llm_openai(
    messages: list,
    model_name: str,
    api_key: str,
    temperature: float = 0.7
) -> str:
    """Run OpenAI models"""
    client = OpenAI(api_key=api_key)
    print("run llm openai")
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=temperature,
    )
    
    return chat_completion.choices[0].message.content or ""


def run_llm_openai_stream(
    messages: list,
    model_name: str,
    api_key: str,
    temperature: float = 0.7
) -> Generator[str, None, None]:
    """Run OpenAI models with streaming"""
    client = OpenAI(api_key=api_key)
    print("[STREAM] run llm openai stream")
    
    stream = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=temperature,
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def run_llm_anthropic(
    messages: list,
    model_name: str,
    api_key: str,
    temperature: float = 0.7
) -> str:
    """Run Anthropic Claude models"""
    client = Anthropic(api_key=api_key)
    system_msg = ""
    user_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            user_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    response = client.messages.create(
        model=model_name,
        max_tokens=2048,
        temperature=temperature,
        system=system_msg,
        messages=user_messages
    )
    
    return response.content[0].text if response.content else "" #type:ignore


def run_llm_anthropic_stream(
    messages: list,
    model_name: str,
    api_key: str,
    temperature: float = 0.7
) -> Generator[str, None, None]:
    """Run Anthropic Claude models with streaming"""
    client = Anthropic(api_key=api_key)
    print("[STREAM] run llm anthropic stream")
    system_msg = ""
    user_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            user_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    with client.messages.stream(
        model=model_name,
        max_tokens=2048,
        temperature=temperature,
        system=system_msg,
        messages=user_messages
    ) as stream:
        for text in stream.text_stream:
            yield text


def run_llm_groq(
    messages: list,
    model_name: str,
    api_key: str,
    temperature: float = 0.7
) -> str:
    """Run Groq models"""
    client = Groq(api_key=api_key)
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=temperature,
    )
    
    return chat_completion.choices[0].message.content or ""


def run_llm_groq_stream(
    messages: list,
    model_name: str,
    api_key: str,
    temperature: float = 0.7
) -> Generator[str, None, None]:
    """Run Groq models with streaming"""
    client = Groq(api_key=api_key)
    print("[STREAM] run llm groq stream")
    
    stream = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=temperature,
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def run_llm_gemini(
    messages: list,
    model_name: str,
    api_key: str,
    temperature: float = 0.7
) -> str:
    """Run Google Gemini models"""
    client = Client(api_key=api_key)
    system_instruction = ""
    gemini_contents = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        else:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=str(msg["content"]))]
                )
            )
    
    response = client.models.generate_content(
        model=model_name,
        contents=gemini_contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=2048
        )
    )
    
    return response.text if response.text else ""


def run_llm_gemini_stream(
    messages: list,
    model_name: str,
    api_key: str,
    temperature: float = 0.7
) -> Generator[str, None, None]:
    """Run Google Gemini models with streaming"""
    client = Client(api_key=api_key)
    print("[STREAM] run llm gemini stream")
    system_instruction = ""
    gemini_contents = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        else:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=str(msg["content"]))]
                )
            )
    
    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=gemini_contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=2048
        )
    ):
        if chunk.text:
            yield chunk.text


def run_llm(
    user_query_for_db: str, 
    model_name: str, 
    need_chat_history: bool, 
    api_key: str,
    provider: Optional[str] = None,
    temperature: float = 0.7
) -> str:
    """
    Unified LLM runner supporting OpenAI, Anthropic, Groq, and Gemini.
    
    Args:
        user_query_for_db: The user query to process
        model_name: Name of the model (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-2.0-flash-exp')
        need_chat_history: Whether to include chat history
        api_key: API key for the provider
        provider: Optional provider override ('openai', 'anthropic', 'groq', 'gemini')
        temperature: Temperature for generation
        
    Returns:
        Assistant response as string
    """
    if api_key is None:
        raise EnvironmentError(f"API key not set for model {model_name}")
    if provider is None:
        provider = detect_provider(model_name)
    
    print(f"[{provider.upper()}] Using model: {model_name}")
    insert_user_message(user_query_for_db)
    messages_to_send = [SYSTEM_MESSAGE]
    
    if need_chat_history:
        conversation_messages = get_chat_history()
        messages_to_send.extend(conversation_messages)
    
    messages_to_send.append({"role": "user", "content": user_query_for_db})
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(str(messages_to_send))
    print(f"[TOKEN COUNT] {len(tokens)}")
    try:
        print("------------------------------------Assistant Running -------------------------------------------")
        
        if provider == 'openai':
            assistant_response = run_llm_openai(messages_to_send, model_name, api_key, temperature)
        elif provider == 'anthropic':
            assistant_response = run_llm_anthropic(messages_to_send, model_name, api_key, temperature)
        elif provider == 'groq':
            assistant_response = run_llm_groq(messages_to_send, model_name, api_key, temperature)
        elif provider == 'gemini':
            assistant_response = run_llm_gemini(messages_to_send, model_name, api_key, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        if not assistant_response:
            print("[WARNING] Assistant returned no content.")
            assistant_response = ""
        insert_assistant_message(assistant_response)
        
        return assistant_response
        
    except Exception as e:
        error_msg = f"Error calling {provider} API with model {model_name}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        raise


def run_llm_stream(
    user_query_for_db: str, 
    model_name: str, 
    need_chat_history: bool, 
    api_key: str,
    provider: Optional[str] = None,
    temperature: float = 0.7
) -> tuple[Generator[str, None, None], list]:
    """
    Unified LLM streaming runner supporting OpenAI, Anthropic, Groq, and Gemini.
    
    Args:
        user_query_for_db: The user query to process
        model_name: Name of the model
        need_chat_history: Whether to include chat history
        api_key: API key for the provider
        provider: Optional provider override
        temperature: Temperature for generation
        
    Returns:
        Tuple of (text_generator, messages_sent)
    """
    if api_key is None:
        raise EnvironmentError(f"API key not set for model {model_name}")
    if provider is None:
        provider = detect_provider(model_name)
    
    print(f"[{provider.upper()} STREAM] Using model: {model_name}")
    insert_user_message(user_query_for_db)
    messages_to_send = [SYSTEM_MESSAGE]
    
    if need_chat_history:
        conversation_messages = get_chat_history()
        messages_to_send.extend(conversation_messages)
    
    messages_to_send.append({"role": "user", "content": user_query_for_db})
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(str(messages_to_send))
    print(f"[TOKEN COUNT] {len(tokens)}")
    
    print("------------------------------------Assistant Streaming -------------------------------------------")
    
    if provider == 'openai':
        stream_gen = run_llm_openai_stream(messages_to_send, model_name, api_key, temperature)
    elif provider == 'anthropic':
        stream_gen = run_llm_anthropic_stream(messages_to_send, model_name, api_key, temperature)
    elif provider == 'groq':
        stream_gen = run_llm_groq_stream(messages_to_send, model_name, api_key, temperature)
    elif provider == 'gemini':
        stream_gen = run_llm_gemini_stream(messages_to_send, model_name, api_key, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return stream_gen, messages_to_send