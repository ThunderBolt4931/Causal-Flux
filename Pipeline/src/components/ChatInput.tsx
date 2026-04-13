import { useState } from "react";
import { Send, Zap, GitBranch } from "lucide-react";
import { ModelSelector } from "./ModelSelector";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSend: (message: string, model: string, taskMode: "task1" | "task2") => void;
  isLoading: boolean;
}

export function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [message, setMessage] = useState("");
  const [selectedModel, setSelectedModel] = useState("OpenAI GPT-4o-mini");
  const [taskMode, setTaskMode] = useState<"task1" | "task2">("task1");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSend(message.trim(), selectedModel, taskMode);
      setMessage("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto">
      <div className="rounded-2xl border border-border bg-card shadow-lg">
        {/* Text Input */}
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={taskMode === "task1" ? "Ask a causal question..." : "Ask a follow-up question..."}
          rows={1}
          className="w-full resize-none bg-transparent px-5 py-4 text-foreground placeholder:text-muted-foreground focus:outline-none text-base"
          style={{ minHeight: "56px", maxHeight: "200px" }}
          disabled={isLoading}
        />

        {/* Controls */}
        <div className="flex items-center justify-between border-t border-border/50 px-4 py-3">
          <div className="flex items-center gap-2">
            {/* Model Selector */}
            <ModelSelector
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
            />

            {/* Task Mode Toggle */}
            <button
              type="button"
              onClick={() => setTaskMode(taskMode === "task1" ? "task2" : "task1")}
              className={cn(
                "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-all",
                taskMode === "task1"
                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                  : "bg-purple-500/20 text-purple-400 border border-purple-500/30"
              )}
              title={taskMode === "task1" ? "Task 1: Initial Causal Inquiry" : "Task 2: Contextual Follow-up"}
            >
              {taskMode === "task1" ? (
                <>
                  <Zap className="h-4 w-4" />
                  <span>Task 1</span>
                </>
              ) : (
                <>
                  <GitBranch className="h-4 w-4" />
                  <span>Task 2</span>
                </>
              )}
            </button>
          </div>

          {/* Send Button */}
          <button
            type="submit"
            disabled={!message.trim() || isLoading}
            className={cn(
              "flex h-10 w-10 items-center justify-center rounded-xl transition-all duration-200",
              message.trim() && !isLoading
                ? "bg-primary text-primary-foreground hover:scale-105 shadow-lg"
                : "bg-muted text-muted-foreground cursor-not-allowed"
            )}
          >
            {isLoading ? (
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </button>
        </div>
      </div>
    </form>
  );
}
