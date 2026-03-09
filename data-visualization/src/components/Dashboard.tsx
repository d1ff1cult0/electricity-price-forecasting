"use client";

import { useEffect, useState } from "react";
import { FolderGit2, Activity, Play, ChevronRight, ChevronDown, BarChart3, LineChart, Plus, Trash2 } from "lucide-react";
import PredictionsChart from "./PredictionsChart";
import MetricsView from "./MetricsView";

type ExperimentData = {
    name: string;
    models: {
        name: string;
        runs: string[];
    }[];
};

export type Selection = {
    id: string; // Unique ID like "exp-model-run"
    experiment: string;
    model: string;
    run: string;
    color: string;
};

const COLORS = ["#4b6bfb", "#fba24b", "#10b981", "#8b5cf6", "#ec4899", "#06b6d4"];

export default function Dashboard() {
    const [data, setData] = useState<ExperimentData[]>([]);
    const [loading, setLoading] = useState(true);

    // Array of selections for comparison
    const [selections, setSelections] = useState<Selection[]>([]);

    const [expandedExperiments, setExpandedExperiments] = useState<string[]>([]);
    const [expandedModels, setExpandedModels] = useState<string[]>([]);

    useEffect(() => {
        fetch("/api/experiments")
            .then((res) => res.json())
            .then((resData) => {
                setData(resData);
                setLoading(false);
            })
            .catch((err) => {
                console.error(err);
                setLoading(false);
            });
    }, []);

    const toggleExperiment = (name: string) => {
        setExpandedExperiments((prev) =>
            prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]
        );
    };

    const toggleModel = (expModelKey: string) => {
        setExpandedModels((prev) =>
            prev.includes(expModelKey) ? prev.filter((k) => k !== expModelKey) : [...prev, expModelKey]
        );
    };

    const addSelection = (exp: string, mod: string, run: string) => {
        const id = `${exp}|${mod}|${run}`;
        if (selections.find(s => s.id === id)) {
            // Remove if already selected (toggle behavior)
            setSelections(prev => prev.filter(s => s.id !== id));
            return;
        }

        // Assign a color
        const nextColor = COLORS[selections.length % COLORS.length];

        setSelections(prev => [
            ...prev,
            { id, experiment: exp, model: mod, run, color: nextColor }
        ]);
    };

    const removeSelection = (id: string) => {
        setSelections(prev => prev.filter(s => s.id !== id));
    };

    const clearSelections = () => {
        setSelections([]);
    };

    return (
        <div className="flex w-full min-h-screen">
            {/* Sidebar */}
            <aside className="w-80 border-r border-border bg-card/60 glass flex flex-col h-screen sticky top-0 shrink-0">
                <div className="h-16 flex items-center px-6 border-b border-white/5">
                    <Activity className="w-5 h-5 text-primary mr-3" />
                    <h1 className="font-bold text-lg tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
                        Forecasting View
                    </h1>
                </div>

                <div className="p-4 border-b border-white/5">
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="uppercase text-xs font-semibold text-muted-foreground tracking-wider">Active Compare</h3>
                        {selections.length > 0 && (
                            <button onClick={clearSelections} className="text-xs text-destructive hover:text-white transition-colors">Clear</button>
                        )}
                    </div>

                    {selections.length === 0 ? (
                        <p className="text-xs text-muted-foreground/60">Click on any run or average to add it to the comparison view.</p>
                    ) : (
                        <div className="space-y-1.5 max-h-40 overflow-y-auto">
                            {selections.map(sel => (
                                <div key={sel.id} className="flex items-center justify-between bg-black/20 p-2 rounded-md border border-white/5">
                                    <div className="flex flex-col truncate pr-2">
                                        <span className="text-[10px] text-muted-foreground truncate">{sel.experiment} &gt; {sel.model}</span>
                                        <span className="text-xs font-medium flex items-center">
                                            <div className="w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: sel.color }} />
                                            {sel.run === 'average' ? 'Average' : sel.run.replace('run_', 'Run ')}
                                        </span>
                                    </div>
                                    <button onClick={() => removeSelection(sel.id)} className="text-muted-foreground hover:text-destructive shrink-0">
                                        <Trash2 className="w-3.5 h-3.5" />
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-2">
                    <h3 className="uppercase text-xs font-semibold text-muted-foreground tracking-wider mb-2">Experiments</h3>
                    {loading ? (
                        <div className="animate-pulse flex space-x-2 p-2">
                            <div className="w-4 h-4 rounded-full bg-white/10" />
                            <div className="h-4 w-32 bg-white/10 rounded" />
                        </div>
                    ) : data.length === 0 ? (
                        <p className="text-sm text-muted-foreground p-2">No experiments found.</p>
                    ) : (
                        data.map((exp) => (
                            <div key={exp.name} className="flex flex-col">
                                <button
                                    onClick={() => toggleExperiment(exp.name)}
                                    className="flex items-center w-full p-2 rounded-lg hover:bg-white/5 text-left transition-colors text-sm font-medium"
                                >
                                    {expandedExperiments.includes(exp.name) ? (
                                        <ChevronDown className="w-4 h-4 mr-2 text-muted-foreground" />
                                    ) : (
                                        <ChevronRight className="w-4 h-4 mr-2 text-muted-foreground" />
                                    )}
                                    <FolderGit2 className="w-4 h-4 mr-2 text-primary/70" />
                                    <span className="truncate">{exp.name}</span>
                                </button>

                                {expandedExperiments.includes(exp.name) && (
                                    <div className="ml-5 mt-1 relative before:absolute before:inset-y-0 before:left-[11px] before:w-px before:bg-white/10">
                                        {exp.models.map((mod) => {
                                            const modKey = `${exp.name}-${mod.name}`;
                                            return (
                                                <div key={mod.name} className="flex flex-col mb-1 relative">
                                                    <div className="absolute left-[11px] top-[14px] w-[14px] h-px bg-white/10" />
                                                    <button
                                                        onClick={() => toggleModel(modKey)}
                                                        className="flex items-center w-full p-1.5 pl-8 rounded-md hover:bg-white/5 text-left transition-colors text-xs font-medium text-foreground/80"
                                                    >
                                                        {expandedModels.includes(modKey) ? (
                                                            <ChevronDown className="w-3 h-3 mr-1.5 text-muted-foreground" />
                                                        ) : (
                                                            <ChevronRight className="w-3 h-3 mr-1.5 text-muted-foreground" />
                                                        )}
                                                        <BarChart3 className="w-3.5 h-3.5 mr-2 text-secondary-foreground/60" />
                                                        <span className="truncate">{mod.name}</span>
                                                    </button>

                                                    {expandedModels.includes(modKey) && (
                                                        <div className="ml-8 mt-1 space-y-0.5 relative before:absolute before:inset-y-0 before:left-[8px] before:w-px before:bg-white/5">
                                                            <button
                                                                onClick={() => addSelection(exp.name, mod.name, 'average')}
                                                                className={`flex items-center w-full p-1.5 pl-6 rounded-md transition-colors text-xs relative ${selections.find(s => s.id === `${exp.name}|${mod.name}|average`)
                                                                        ? "bg-primary/20 text-primary font-semibold"
                                                                        : "hover:bg-white/5 text-muted-foreground hover:text-foreground"
                                                                    }`}
                                                            >
                                                                <div className="absolute left-[8px] top-[14px] w-[12px] h-px bg-white/5" />
                                                                <LineChart className="w-3 h-3 mr-2" />
                                                                Average
                                                            </button>

                                                            {mod.runs.map((run) => (
                                                                <button
                                                                    key={run}
                                                                    onClick={() => addSelection(exp.name, mod.name, run)}
                                                                    className={`flex items-center w-full p-1.5 pl-6 rounded-md transition-colors text-xs relative ${selections.find(s => s.id === `${exp.name}|${mod.name}|${run}`)
                                                                            ? "bg-primary/20 text-primary font-semibold"
                                                                            : "hover:bg-white/5 text-muted-foreground hover:text-foreground"
                                                                        }`}
                                                                >
                                                                    <div className="absolute left-[8px] top-[14px] w-[12px] h-px bg-white/5" />
                                                                    <Play className="w-3 h-3 mr-2" />
                                                                    {run.replace('run_', 'Run ')}
                                                                </button>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>
                        ))
                    )
                    }
                </div >
            </aside >

            {/* Main Content */}
            < main className="flex-1 overflow-y-auto p-8 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-background to-black" >
                {
                    selections.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full text-muted-foreground/60">
                            <Activity className="w-16 h-16 mb-4 opacity-20" />
                            <p className="text-lg">Select experiments, models, and runs from the sidebar to compare.</p>
                        </div>
                    ) : (
                        <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                            <header className="glass-card p-6 flex flex-col justify-between">
                                <div>
                                    <h2 className="text-2xl font-bold flex items-center">
                                        <BarChart3 className="w-6 h-6 mr-3 text-primary" />
                                        Comparison Dashboard
                                    </h2>
                                    <p className="text-sm text-muted-foreground mt-1">
                                        Comparing {selections.length} model runs across experiments.
                                    </p>
                                </div>
                            </header>

                            <MetricsView selections={selections} />

                            <div className="glass-card p-6 min-h-[500px]">
                                <h3 className="text-lg font-semibold mb-6 flex items-center">
                                    <LineChart className="w-5 h-5 mr-3 text-primary" />
                                    Predictions vs Actual Prices
                                </h3>
                                <PredictionsChart selections={selections} />
                            </div>
                        </div>
                    )
                }
            </main >
        </div >
    );
}
