import { useEffect, useRef, useState } from "react";
import { Message } from "@/types/chat";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { TranscriptViewer } from "./TranscriptViewer";
import { VisualizationModal } from "./VisualizationModal";

interface ChatContainerProps {
  messages: Message[];
  onSend: (message: string, model: string, taskMode: "task1" | "task2") => void;
  isLoading: boolean;
  isNewChat: boolean;
}

export function ChatContainer({ messages, onSend, isLoading, isNewChat }: ChatContainerProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // State for Transcript Viewer Modal
  const [viewingTranscriptId, setViewingTranscriptId] = useState<string | null>(null);

  // State for Visualization Modal
  const [viewingGraphsMetadata, setViewingGraphsMetadata] = useState<any>(null);

  // Auto-scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="relative flex flex-1 flex-col h-screen w-full overflow-hidden bg-transparent">

      {/* Transcript Viewer Modal (Overlays everything when active) */}
      {viewingTranscriptId && (
        <TranscriptViewer
          transcriptId={viewingTranscriptId}
          onClose={() => setViewingTranscriptId(null)}
        />
      )}

      {/* Visualization Modal */}
      <VisualizationModal
        isOpen={!!viewingGraphsMetadata}
        onClose={() => setViewingGraphsMetadata(null)}
        metadata={viewingGraphsMetadata}
      />

      {/* Content Layer */}
      <div className="relative z-10 flex flex-1 flex-col h-screen pointer-events-none">

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto scrollbar-thin pointer-events-auto">
          {isNewChat && messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full px-4">
              <h1 className="text-5xl md:text-6xl font-light tracking-tight text-foreground mb-12 animate-fade-in drop-shadow-md">
                <span className="font-normal">Causal</span>
                <span className="text-primary">Flux</span>
              </h1>

              <div className="w-full max-w-3xl animate-fade-in" style={{ animationDelay: "100ms" }}>
                <ChatInput onSend={onSend} isLoading={isLoading} />
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto">
              {messages.map((message, index) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  index={index}
                  onTranscriptClick={(id) => setViewingTranscriptId(id)}
                  onGraphsClick={(metadata) => setViewingGraphsMetadata(metadata)}
                />
              ))}

              {/* Loading Indicator */}
              {isLoading && (
                <div
                  className="flex px-4 py-4 animate-fade-in justify-start"
                  style={{ animationDelay: `${messages.length * 40}ms` }}
                >
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-green-400/20 border border-green-400 mr-3">
                    <span className="text-green-600 font-bold">A</span>
                  </div>
                  <div className="flex flex-col max-w-[75%] space-y-1">
                    <div className="text-xs font-medium text-gray-400">CausalFlux</div>
                    <div className="rounded-2xl px-4 py-3 shadow-sm border bg-gray-800/80 backdrop-blur-sm border-gray-700 text-gray-400">
                      thinking...
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Bottom Input Area */}
        {(!isNewChat || messages.length > 0) ? (
          <div className="bg-background/80 backdrop-blur-md p-4 pointer-events-auto">
            <ChatInput onSend={onSend} isLoading={isLoading} />
          </div>
        ) : null}
      </div>
    </div>
  );
}