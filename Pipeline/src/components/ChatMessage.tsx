import { Message } from "@/types/chat";
import { cn } from "@/lib/utils";
import { User, Bot, FileText, CheckCircle2, BarChart3 } from "lucide-react";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { memo } from "react";

interface ChatMessageProps {
  message: Message;
  index: number;
  onTranscriptClick?: (id: string) => void;
  onGraphsClick?: (metadata: any) => void;
}

function ChatMessageComponent({ message, index, onTranscriptClick, onGraphsClick }: ChatMessageProps) {
  const isUser = message.role === "user";

  // Extract transcript IDs and visualization metadata
  const hasTranscripts = !isUser && message.metadata?.transcript_ids && Array.isArray(message.metadata.transcript_ids) && message.metadata.transcript_ids.length > 0;
  const hasVisualizationData = hasTranscripts; // Show graphs button whenever we have transcript IDs

  // Define components with proper typing
  const markdownComponents: Components = {
    p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
    strong: ({ node, ...props }) => <span className="font-bold" {...props} />,
    em: ({ node, ...props }) => <span className="italic" {...props} />,
    ul: ({ node, ...props }) => <ul className="list-disc pl-4 mb-2 space-y-1" {...props} />,
    ol: ({ node, ...props }) => <ol className="list-decimal pl-4 mb-2 space-y-1" {...props} />,
    li: ({ node, ...props }) => <li className="pl-1" {...props} />,
    h1: ({ node, ...props }) => <h1 className="text-xl font-bold mb-2 mt-4" {...props} />,
    h2: ({ node, ...props }) => <h2 className="text-lg font-bold mb-2 mt-3" {...props} />,
    h3: ({ node, ...props }) => <h3 className="text-base font-bold mb-1 mt-2" {...props} />,
    table: ({ node, ...props }) => (
      <div className="my-4 w-full overflow-x-auto rounded-lg border border-white/10">
        <table className="w-full text-left text-sm" {...props} />
      </div>
    ),
    thead: ({ node, ...props }) => (
      <thead className={isUser ? "bg-white/20" : "bg-muted/50"} {...props} />
    ),
    tbody: ({ node, ...props }) => <tbody className="divide-y divide-white/10" {...props} />,
    tr: ({ node, ...props }) => <tr className="hover:bg-white/5" {...props} />,
    th: ({ node, ...props }) => <th className="px-4 py-2 font-semibold" {...props} />,
    td: ({ node, ...props }) => <td className="px-4 py-2" {...props} />,
    code: ({ node, inline, className, children, ...props }) => {
      const match = /language-(\w+)/.exec(className || "");
      if (inline || !match) {
        return (
          <code
            className="bg-black/20 rounded px-1 py-0.5 font-mono text-xs"
            {...props}
          >
            {children}
          </code>
        );
      }
      return (
        <div className="relative my-2 rounded-lg bg-black/40 p-3 font-mono text-xs overflow-x-auto">
          <code className={className} {...props}>
            {children}
          </code>
        </div>
      );
    },
  };

  return (
    <div
      className={cn(
        "flex px-4 py-4 animate-fade-in",
        isUser ? "justify-end" : "justify-start"
      )}
      style={{ animationDelay: `${index * 40}ms` }}
    >
      {/* Avatar - Assistant */}
      {!isUser && (
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent/10 border border-accent/20 mr-3">
          <Bot className="h-5 w-5 text-accent" />
        </div>
      )}

      {/* Message Content */}
      <div
        className={cn(
          "flex flex-col max-w-[85%] md:max-w-[75%]",
          isUser ? "items-end" : "items-start"
        )}
      >
        <span className="text-xs font-medium text-muted-foreground mb-1">
          {isUser ? "You" : "CausalFlux"}
        </span>

        <div
          className={cn(
            "rounded-2xl px-4 py-3 shadow-sm border overflow-hidden w-full transition-all",
            isUser
              ? "bg-primary text-primary-foreground border-primary/20"
              : "bg-card text-foreground border-border"
          )}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {message.content}
          </ReactMarkdown>

          {/* TRANSCRIPT BUTTONS SECTION */}
          {hasTranscripts && (
            <div className="mt-4 pt-3 border-t border-white/10 flex flex-col gap-2 animate-fade-in">
              <div className="flex items-center gap-2 text-[10px] text-muted-foreground uppercase tracking-wider font-semibold">
                <CheckCircle2 className="h-3 w-3 text-emerald-500" />
                <span>Verified Sources</span>
              </div>

              <div className="flex flex-wrap gap-2">
                {message.metadata?.transcript_ids?.map((id: string) => (
                  <button
                    key={id}
                    onClick={() => onTranscriptClick?.(id)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-secondary/50 hover:bg-primary/20 hover:text-primary border border-white/5 hover:border-primary/30 transition-all duration-200 group text-xs font-medium text-secondary-foreground"
                    title={`View Transcript ${id}`}
                  >
                    <FileText className="h-3.5 w-3.5 opacity-70 group-hover:opacity-100" />
                    <span>Transcript {id}</span>
                  </button>
                ))}
              </div>

              {/* VIEW GRAPHS BUTTON - Placed inside transcript section */}
              {hasVisualizationData && (
                <button
                  onClick={() => onGraphsClick?.(message.metadata)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-purple-500/20 hover:bg-purple-500/30 border border-purple-500/30 hover:border-purple-500/50 transition-all duration-200 group text-xs font-medium text-purple-300 w-fit"
                  title="View Visualization Graphs"
                >
                  <BarChart3 className="h-3.5 w-3.5" />
                  <span>View Graphs</span>
                </button>
              )}
            </div>
          )}
          {/* END TRANSCRIPT BUTTONS */}

        </div>
      </div>

      {/* Avatar - User */}
      {isUser && (
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 border border-primary/20 ml-3">
          <User className="h-5 w-5 text-primary" />
        </div>
      )}
    </div>
  );
}

// Memoize to prevent re-rendering old messages when new ones arrive
export const ChatMessage = memo(ChatMessageComponent, (prevProps, nextProps) => {
  // Only re-render if message content or ID changes
  return (
    prevProps.message.id === nextProps.message.id &&
    prevProps.message.content === nextProps.message.content &&
    prevProps.index === nextProps.index
  );
});