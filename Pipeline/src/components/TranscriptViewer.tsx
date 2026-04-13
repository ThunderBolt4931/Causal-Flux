import { X, Loader2, User, Phone } from "lucide-react";
import { useEffect, useState } from "react";
import { apiUrl } from "@/lib/api";

interface TranscriptViewerProps {
    transcriptId: string | null;
    onClose: () => void;
}

export function TranscriptViewer({ transcriptId, onClose }: TranscriptViewerProps) {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    useEffect(() => {
        if (!transcriptId) return;

        console.log(`[TranscriptViewer] Fetching transcript: ${transcriptId}`);
        setLoading(true);
        setError("");

        // Use apiUrl for Docker-compatible URLs
        const url = apiUrl(`/transcript/${transcriptId}`);
        console.log(`[TranscriptViewer] Request URL: ${url}`);

        fetch(url)
            .then((res) => {
                console.log(`[TranscriptViewer] Response status: ${res.status}`);
                if (!res.ok) throw new Error("Failed to load transcript");
                return res.json();
            })
            .then((data) => {
                console.log(`[TranscriptViewer] Data received:`, data);
                setData(data);
            })
            .catch((err) => {
                console.error(`[TranscriptViewer] Error:`, err);
                setError(err.message);
            })
            .finally(() => setLoading(false));
    }, [transcriptId]);

    if (!transcriptId) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-fade-in">
            <div className="w-full max-w-2xl bg-card border border-border rounded-xl shadow-2xl flex flex-col max-h-[80vh]">

                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-border">
                    <div>
                        <h2 className="text-lg font-semibold text-foreground">Transcript Details</h2>
                        <p className="text-xs text-muted-foreground font-mono">{transcriptId}</p>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-muted rounded-lg transition-colors">
                        <X className="h-5 w-5 text-muted-foreground" />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-4 scrollbar-thin">
                    {loading ? (
                        <div className="flex h-40 items-center justify-center">
                            <Loader2 className="h-8 w-8 animate-spin text-primary" />
                        </div>
                    ) : error ? (
                        <div className="p-4 text-destructive bg-destructive/10 rounded-lg">Error: {error}</div>
                    ) : data ? (
                        <div className="space-y-6">
                            {/* Metadata Badge */}
                            <div className="flex gap-2 flex-wrap">
                                <span className="px-2 py-1 bg-primary/10 text-primary text-xs rounded border border-primary/20">
                                    Domain: {data.domain}
                                </span>
                                <span className="px-2 py-1 bg-accent/10 text-accent text-xs rounded border border-accent/20">
                                    Intent: {data.intent || data.call_intent}
                                </span>
                            </div>

                            {/* Conversation History */}
                            <div className="space-y-4">
                                {data.turns?.map((turn: any, i: number) => (
                                    <div key={i} className="space-y-2">
                                        {turn.conversation.map((msg: any, j: number) => {
                                            const isAgent = msg.speaker === "Agent";
                                            return (
                                                <div key={j} className={`flex gap-3 ${isAgent ? "" : "flex-row-reverse"}`}>
                                                    <div className={`h-8 w-8 shrink-0 flex items-center justify-center rounded-lg border ${isAgent ? "bg-accent/10 border-accent/20" : "bg-primary/10 border-primary/20"}`}>
                                                        {isAgent ? <Phone className="h-4 w-4 text-accent" /> : <User className="h-4 w-4 text-primary" />}
                                                    </div>
                                                    <div className={`p-3 rounded-lg text-sm max-w-[80%] ${isAgent ? "bg-muted/30" : "bg-primary/5 border border-primary/10"}`}>
                                                        <p className="font-semibold text-xs mb-1 opacity-70">{msg.speaker}</p>
                                                        <p className="leading-relaxed">{msg.utterance}</p>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : null}
                </div>
            </div>
        </div>
    );
}