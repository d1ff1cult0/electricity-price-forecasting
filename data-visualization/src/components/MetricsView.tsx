"use client";

import { useEffect, useState } from "react";
import { AlertCircle } from "lucide-react";
import { Selection } from "./Dashboard";

interface MetricsProps {
    selections: Selection[];
}

export default function MetricsView({ selections }: MetricsProps) {
    const [metricsData, setMetricsData] = useState<Record<string, Record<string, number>>>({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (selections.length === 0) return;

        setLoading(true);
        setError(null);

        Promise.all(
            selections.map(sel => {
                const isAverage = sel.run === 'average';
                let url = `/api/metrics?experiment=${encodeURIComponent(sel.experiment)}&model=${encodeURIComponent(sel.model)}`;
                if (!isAverage) {
                    url += `&run=${encodeURIComponent(sel.run)}`;
                }

                return fetch(url)
                    .then(res => res.json())
                    .then(data => ({ id: sel.id, data }));
            })
        )
            .then(results => {
                const newMetrics: Record<string, Record<string, number>> = {};
                results.forEach(res => {
                    if (!res.data.error) {
                        newMetrics[res.id] = res.data.mean_metrics || res.data;
                    }
                });
                setMetricsData(newMetrics);
                setLoading(false);
            })
            .catch(err => {
                setError(err.message);
                setLoading(false);
            });
    }, [selections]);

    if (loading && Object.keys(metricsData).length === 0) {
        return (
            <div className="h-64 glass rounded-xl bg-white/5 animate-pulse" />
        );
    }

    if (error || Object.keys(metricsData).length === 0) {
        return null; /* handled gracefully or silently */
    }

    // Format to display visually pleasing numbers
    const formatValue = (num: number) => {
        if (typeof num !== 'number') return String(num);
        return new Intl.NumberFormat('en-US', {
            maximumFractionDigits: 3,
            minimumFractionDigits: 0
        }).format(num);
    };

    // Extract all unique metric keys across all selections
    const allMetricKeys = Array.from(new Set(
        Object.keys(metricsData).flatMap(id => Object.keys(metricsData[id] || {}))
    )).filter(k => {
        // Ensure it's a number for at least one model
        return Object.values(metricsData).some(m => typeof m[k] === 'number');
    });

    const keyMetricsOrder = ["MAE", "RMSE", "MAPE", "R2", "MSE", "PICP", "MPIW"];
    const sortedKeys = allMetricKeys.sort((a, b) => {
        const idxA = keyMetricsOrder.indexOf(a);
        const idxB = keyMetricsOrder.indexOf(b);
        if (idxA !== -1 && idxB !== -1) return idxA - idxB;
        if (idxA !== -1) return -1;
        if (idxB !== -1) return 1;
        return a.localeCompare(b);
    });

    return (
        <div className="glass-card overflow-hidden">
            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left text-muted-foreground">
                    <thead className="text-xs uppercase bg-white/5 text-foreground/80 border-b border-border">
                        <tr>
                            <th scope="col" className="px-6 py-4 font-semibold tracking-wider">Metric</th>
                            {selections.map(sel => (
                                <th key={sel.id} scope="col" className="px-6 py-4 font-semibold tracking-wider">
                                    <div className="flex flex-col">
                                        <span className="text-[10px] text-muted-foreground/60">{sel.experiment}</span>
                                        <div className="flex items-center text-foreground">
                                            <div className="w-2.5 h-2.5 rounded-full mr-2" style={{ backgroundColor: sel.color }} />
                                            {sel.model} ({sel.run === 'average' ? 'Avg' : sel.run.replace('run_', 'R')})
                                        </div>
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {sortedKeys.map((metricKey, idx) => (
                            <tr key={metricKey} className={`border-b border-white/5 hover:bg-white/5 transition-colors ${idx % 2 === 0 ? 'bg-transparent' : 'bg-black/10'}`}>
                                <th scope="row" className="px-6 py-3 font-medium text-foreground uppercase tracking-wider whitespace-nowrap">
                                    {metricKey.replace(/_/g, ' ')}
                                </th>
                                {selections.map(sel => {
                                    const val = metricsData[sel.id]?.[metricKey];
                                    // Find if this is the best value across the row to highlight
                                    // For R2 PICP, higher is better. For errors, lower is better.
                                    let isBest = false;
                                    if (val !== undefined && Object.keys(metricsData).length > 1) {
                                        const allValsRow = selections.map(s => metricsData[s.id]?.[metricKey]).filter(v => v !== undefined) as number[];
                                        if (["R2", "PICP", "PINAW"].includes(metricKey)) {
                                            isBest = val === Math.max(...allValsRow);
                                        } else {
                                            isBest = val === Math.min(...allValsRow);
                                        }
                                    }

                                    return (
                                        <td key={sel.id} className={`px-6 py-3 ${isBest ? 'text-primary font-bold' : ''}`}>
                                            {val !== undefined ? formatValue(val) : '-'}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
