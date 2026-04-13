export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    provider?: string;
    model?: string;
    route?: string;
    tokens?: number;
    source?: string;
    history_used?: boolean;
    transcript_ids?: string[];
    query_drivers?: string[];
    query_text?: string;
    [key: string]: any;
  };
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export interface ModelChoice {
  name: string;
  model: string;
  provider: string;
}

export const MODEL_CHOICES: Record<string, { model: string; provider: string }> = {
  "OpenAI GPT-4o": { model: "gpt-4o", provider: "openai" },
  "OpenAI GPT-4o-mini": { model: "gpt-4o-mini", provider: "openai" },
  "Claude 3.5 Sonnet": { model: "claude-3-5-sonnet-20241022", provider: "anthropic" },
  "Claude 3 Opus": { model: "claude-3-opus-20240229", provider: "anthropic" },
  "Claude 3 Haiku": { model: "claude-3-haiku-20240307", provider: "anthropic" },
  "Groq Llama 3.3 70B": { model: "llama-3.3-70b-versatile", provider: "groq" },
  "Groq Llama 3.1 8B": { model: "llama-3.1-8b-instant", provider: "groq" },
  "Groq GPT OSS": { model: "openai/gpt-oss-120b", provider: "groq" },
  "Gemini 2.5 Pro": { model: "gemini-2.5-pro", provider: "gemini" },
  "Gemini 2.0 Flash": { model: "gemini-2.5-flash", provider: "gemini" },
};