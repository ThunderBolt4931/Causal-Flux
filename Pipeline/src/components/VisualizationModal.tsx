import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Loader2 } from "lucide-react";
import { apiUrl } from "@/lib/api";

interface VisualizationModalProps {
    isOpen: boolean;
    onClose: () => void;
    metadata: {
        transcript_ids?: string[];
        query_drivers?: string[];
        query_text?: string;
    } | null;
}

export function VisualizationModal({ isOpen, onClose, metadata }: VisualizationModalProps) {
    const [plots, setPlots] = useState<Record<string, string>>({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchVisualizations = async () => {
        if (!metadata || !metadata.transcript_ids || metadata.transcript_ids.length === 0) {
            setError("No visualization data available");
            return;
        }

        setLoading(true);
        setError(null);

        try {
            console.log("Fetching visualizations for:", metadata);
            const response = await fetch(apiUrl("/api/visualizations"), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    transcript_ids: metadata.transcript_ids,
                    drivers: metadata.query_drivers || [],
                    query_text: metadata.query_text || "",
                }),
            });

            if (!response.ok) {
                throw new Error(`Failed to fetch visualizations: ${response.statusText}`);
            }

            const data = await response.json();
            console.log("Visualization response:", data);
            setPlots(data.plots || {});
        } catch (err) {
            console.error("Visualization fetch error:", err);
            setError(err instanceof Error ? err.message : "Failed to load visualizations");
        } finally {
            setLoading(false);
        }
    };

    // Fetch visualizations when modal opens - FIXED: was incorrectly using useState
    useEffect(() => {
        if (isOpen && metadata) {
            fetchVisualizations();
        }
    }, [isOpen, metadata]);

    // Check if we have any valid plots (non-null values)
    const hasPlots = Object.values(plots).some(plot => !!plot);

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto bg-slate-900 text-white border-slate-700">
                <DialogHeader>
                    <DialogTitle className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                        Visualization Graphs
                    </DialogTitle>
                </DialogHeader>

                {loading && (
                    <div className="flex items-center justify-center py-12">
                        <Loader2 className="h-8 w-8 animate-spin text-purple-400" />
                        <span className="ml-3 text-slate-300">Generating visualizations...</span>
                    </div>
                )}

                {error && (
                    <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-4 text-red-300">
                        <p className="font-semibold">Error:</p>
                        <p>{error}</p>
                    </div>
                )}

                {!loading && !error && hasPlots && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
                        {plots.intents_bar && (
                            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                                <h3 className="text-lg font-semibold mb-3 text-purple-300">Calls per Driver</h3>
                                <img
                                    src={`data:image/png;base64,${plots.intents_bar}`}
                                    alt="Intents Bar Chart"
                                    className="w-full rounded"
                                />
                            </div>
                        )}

                        {plots.frequency_bar && (
                            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                                <h3 className="text-lg font-semibold mb-3 text-purple-300">Driver Frequency</h3>
                                <img
                                    src={`data:image/png;base64,${plots.frequency_bar}`}
                                    alt="Frequency Bar Chart"
                                    className="w-full rounded"
                                />
                            </div>
                        )}

                        {plots.cluster_pie && (
                            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                                <h3 className="text-lg font-semibold mb-3 text-purple-300">Cluster Breakdown</h3>
                                <img
                                    src={`data:image/png;base64,${plots.cluster_pie}`}
                                    alt="Cluster Pie Chart"
                                    className="w-full rounded"
                                />
                            </div>
                        )}

                        {plots.bubble_chart && (
                            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                                <h3 className="text-lg font-semibold mb-3 text-purple-300">Cluster Distribution</h3>
                                <img
                                    src={`data:image/png;base64,${plots.bubble_chart}`}
                                    alt="Bubble Chart"
                                    className="w-full rounded"
                                />
                            </div>
                        )}
                    </div>
                )}

                {!loading && !error && !hasPlots && (
                    <div className="text-center py-12 text-slate-400">
                        <p>No visualizations available for this query.</p>
                        <p className="text-sm mt-2 text-slate-500">Try a query that matches more transcripts or drivers.</p>
                    </div>
                )}
            </DialogContent>
        </Dialog>
    );
}
