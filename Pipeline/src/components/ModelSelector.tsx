import { ChevronDown, Check } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { MODEL_CHOICES } from "@/types/chat";
import { cn } from "@/lib/utils";

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
}

export function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const modelKeys = Object.keys(MODEL_CHOICES);
  const getProviderBadge = (key: string) => {
    const provider = MODEL_CHOICES[key].provider;
    const badges: Record<string, { label: string; className: string }> = {
      openai: { label: "", className: "" },
      anthropic: { label: "", className: "" },
      groq: { label: "fast", className: "bg-emerald-500/20 text-emerald-400" },
      gemini: { label: "new", className: "bg-amber-500/20 text-amber-400" },
    };
    return badges[provider];
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 rounded-lg bg-secondary/50 px-3 py-1.5 text-sm text-secondary-foreground hover:bg-secondary transition-colors"
      >
        <span className="truncate max-w-[140px]">{selectedModel}</span>
        <ChevronDown className={cn("h-4 w-4 transition-transform", isOpen && "rotate-180")} />
      </button>

      {isOpen && (
        <div className="absolute bottom-full left-0 mb-2 w-72 rounded-xl border border-border bg-popover p-2 shadow-2xl z-[9999]">
          <div className="max-h-[400px] overflow-y-auto scrollbar-thin space-y-0.5">
            {modelKeys.map((key) => {
              const badge = getProviderBadge(key);
              return (
                <button
                  key={key}
                  onClick={() => {
                    onModelChange(key);
                    setIsOpen(false);
                  }}
                  className={cn(
                    "flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-left text-sm transition-colors",
                    selectedModel === key
                      ? "bg-primary/10 text-foreground"
                      : "text-popover-foreground hover:bg-muted"
                  )}
                >
                  <span className="flex items-center gap-2">
                    {key}
                    {badge?.label && (
                      <span className={cn("rounded px-1.5 py-0.5 text-[10px] font-medium uppercase", badge.className)}>
                        {badge.label}
                      </span>
                    )}
                  </span>
                  {selectedModel === key && <Check className="h-4 w-4 text-primary" />}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
