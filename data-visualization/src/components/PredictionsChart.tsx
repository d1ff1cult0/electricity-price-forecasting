"use client";

import { useEffect, useState } from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Brush,
    Area,
    ComposedChart,
    Legend
} from "recharts";
import { Loader2 } from "lucide-react";
import { Selection } from "./Dashboard";

interface PredictionsChartProps {
    selections: Selection[];
}

export default function PredictionsChart({ selections }: PredictionsChartProps) {
    const [data, setData] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (selections.length === 0) return;

        setLoading(true);

        Promise.all(
            selections.map(sel => {
                const url = `/api/predictions?experiment=${encodeURIComponent(sel.experiment)}&model=${encodeURIComponent(sel.model)}&run=${encodeURIComponent(sel.run)}`;
                return fetch(url).then(res => res.json()).then(data => ({ id: sel.id, color: sel.color, data, sel }));
            })
        )
            .then(results => {
                // Check for errors in array
                const errRes = results.find(r => r.data.error);
                if (errRes) {
                    setError(errRes.data.error);
                    setLoading(false);
                    return;
                }

                // Transform multiple prediction arrays into a single unified timeline 
                // We'll use the maximum length among returned models
                const maxLen = Math.max(...results.map(r => r.data.length || 0));
                if (maxLen === 0) {
                    setError("No prediction data");
                    setLoading(false);
                    return;
                }

                const unifiedData = [];
                // Find the first selection that provides an "actual" series to act as our core Actual line
                const actualProvider = results.find(r => r.data[0] && r.data[0].actual !== undefined);

                for (let i = 0; i < maxLen; i++) {
                    const pt: any = { index: i };
                    if (actualProvider && actualProvider.data[i]) {
                        pt.actual = actualProvider.data[i].actual;
                    }

                    results.forEach(res => {
                        if (res.data[i]) {
                            pt[`pred_${res.id}`] = res.data[i].predicted;
                            // If only 1 model is selected, we include confidence intervals safely
                            if (results.length === 1) {
                                pt.interval_95 = res.data[i]["q_0.025"] != null && res.data[i]["q_0.975"] != null
                                    ? [res.data[i]["q_0.025"], res.data[i]["q_0.975"]] : undefined;
                                pt.interval_50 = res.data[i]["q_0.25"] != null && res.data[i]["q_0.75"] != null
                                    ? [res.data[i]["q_0.25"], res.data[i]["q_0.75"]] : undefined;
                            }
                        }
                    });
                    unifiedData.push(pt);
                }

                setData(unifiedData);
                setError(null);
                setLoading(false);
            })
            .catch(err => {
                setError(err.message);
                setLoading(false);
            });
    }, [selections]);

    if (loading && data.length === 0) {
        return (
            <div className="w-full h-full min-h-[400px] flex flex-col items-center justify-center text-muted-foreground">
                <Loader2 className="w-8 h-8 animate-spin mb-4 text-primary" />
                <p>Loading prediction data...</p>
            </div>
        );
    }

    if (error || data.length === 0) {
        return (
            <div className="w-full h-full min-h-[400px] flex items-center justify-center text-muted-foreground/60">
                <p>Could not load predictions. ({error || "empty"})</p>
            </div>
        );
    }

    // Determine bounds for zoom brush
    const displayCount = Math.min(data.length - 1, 500);

    return (
        <div className="w-full h-[500px]">
            <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                    data={data}
                    margin={{ top: 10, right: 30, left: 0, bottom: 20 }}
                >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                    <XAxis
                        dataKey="index"
                        stroke="rgba(255,255,255,0.3)"
                        tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                        tickMargin={10}
                    />
                    <YAxis
                        stroke="rgba(255,255,255,0.3)"
                        tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                        domain={['dataMin - 10', 'dataMax + 10']}
                    />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: 'rgba(10, 10, 10, 0.95)',
                            borderColor: 'rgba(255,255,255,0.1)',
                            borderRadius: '8px',
                            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)',
                            color: '#ededed'
                        }}
                        itemStyle={{ color: '#ebebeb', fontSize: 12 }}
                        labelStyle={{ color: '#888', marginBottom: '8px' }}
                        formatter={(value: any, name: string) => {
                            if (name === 'actual') return [Number(value).toFixed(2), 'Actual Price'];
                            if (name.startsWith('pred_')) {
                                const id = name.replace('pred_', '');
                                const sel = selections.find(s => s.id === id);
                                return [Number(value).toFixed(2), sel ? `${sel.model} ${sel.run === 'average' ? '(Avg)' : ''}` : 'Predicted'];
                            }
                            return [Number(value).toFixed(2), name];
                        }}
                    />
                    <Legend wrapperStyle={{ paddingTop: "20px", fontSize: '12px' }} />

                    {/* 95% Confidence Interval Area (only if 1 model is selected for clarity) */}
                    {selections.length === 1 && data[0]?.interval_95 && (
                        <Area
                            type="monotone"
                            dataKey="interval_95"
                            stroke="none"
                            fill="#ffffff"
                            fillOpacity={0.05}
                            isAnimationActive={false}
                            legendType="none"
                        />
                    )}

                    {/* 50% Confidence Interval Area (only if 1 model selected) */}
                    {selections.length === 1 && data[0]?.interval_50 && (
                        <Area
                            type="monotone"
                            dataKey="interval_50"
                            stroke="none"
                            fill="#ffffff"
                            fillOpacity={0.1}
                            isAnimationActive={false}
                            legendType="none"
                        />
                    )}

                    <Line
                        type="monotone"
                        dataKey="actual"
                        stroke="rgba(255,255,255,0.8)"
                        strokeWidth={1.5}
                        dot={false}
                        name="Actual Price"
                        isAnimationActive={false}
                    />

                    {selections.map(sel => (
                        <Line
                            key={sel.id}
                            type="monotone"
                            dataKey={`pred_${sel.id}`}
                            stroke={sel.color}
                            strokeWidth={1.5}
                            strokeDasharray={selections.length > 1 ? undefined : "4 4"}
                            dot={false}
                            name={`${sel.model} ${sel.run === 'average' ? '(Avg)' : ''}`}
                            isAnimationActive={false}
                        />
                    ))}

                    <Brush
                        dataKey="index"
                        height={30}
                        stroke="rgba(255,255,255,0.2)"
                        fill="rgba(20,20,25,0.5)"
                        tickFormatter={(i) => i.toString()}
                        startIndex={0}
                        endIndex={displayCount}
                    />
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    );
}
