import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Shield, AlertTriangle, Crosshair, Wifi, Database, Lock } from 'lucide-react';

interface AnalysisResult {
    probability: number;
    risk_level: string;
    confidence: number;
    domain_details?: {
        domain: string;
        age_days: number;
        dns_valid: boolean;
        registrar: string;
        country: string;
    };
    feature_scores?: {
        deepfake_analysis?: {
            is_deepfake: boolean;
            confidence: number;
            reasons: string[];
        };
    };
}

interface AROverlayProps {
    result: AnalysisResult;
    onClose: () => void;
}

const AROverlay: React.FC<AROverlayProps> = ({ result, onClose }) => {
    const isHighRisk = result.probability > 0.7;
    const color = isHighRisk ? 'red' : result.probability > 0.4 ? 'orange' : 'cyan';

    // Simulated HUD data
    const [scanLines, setScanLines] = useState<number[]>([]);

    useEffect(() => {
        const interval = setInterval(() => {
            setScanLines(prev => [...prev.slice(-5), Math.random()]);
        }, 200);
        return () => clearInterval(interval);
    }, []);

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 pointer-events-none flex items-center justify-center overflow-hidden"
        >
            {/* 1. CRT Scanline Effect Background - Click to Close */}
            <div className="absolute inset-0 bg-black/20 z-0 cursor-pointer pointer-events-auto" onClick={onClose} />
            <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-0 bg-[length:100%_4px,3px_100%] pointer-events-none" />

            {/* 2. Main HUD Container - Allow clicks inside */}
            <div className="relative w-full h-full max-w-7xl mx-auto p-4 pointer-events-auto">

                {/* Corners */}
                <CornerBrackets color={color} />

                {/* Fixed Close Button (Top Right) */}
                <button
                    onClick={onClose}
                    className={`absolute -top-2 -right-2 z-[60] p-2 bg-black/80 border border-${color}-500/50 rounded-full hover:bg-${color}-500/20 text-${color}-500 transition-all group`}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="group-hover:rotate-90 transition-transform">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>

                {/* Top Header Bar */}
                <div className={`absolute top-10 left-10 right-10 h-16 border-b-2 border-${color}-500/50 flex items-center justify-between px-4 bg-black/40 backdrop-blur-sm`}>
                    <div className="flex items-center gap-4 text-cyan-400 font-mono text-sm">
                        <Activity className="w-5 h-5 animate-pulse" />
                        <span>SYSTEM_ACTIVE</span>
                        <span className="hidden md:inline">| MODE: NEURO_ANALYSIS</span>
                        <span className="hidden md:inline">| {new Date().toLocaleTimeString()}</span>
                    </div>
                    <div className={`text-${color}-400 font-bold font-mono tracking-widest`}>
                        {isHighRisk ? '⚠️ THREAT DETECTED' : '🛡️ SYSTEM SECURED'}
                    </div>
                    <button
                        onClick={onClose}
                        className="px-6 py-1 border border-cyan-500/50 text-cyan-400 font-mono hover:bg-cyan-500/20 transition-all uppercase text-xs tracking-wider"
                    >
                        [ DEACTIVATE HUD ]
                    </button>
                </div>

                {/* Center Target Reticle */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                    <motion.div
                        animate={{ rotate: 360, scale: [1, 1.1, 1] }}
                        transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                        className={`w-64 h-64 border border-${color}-500/30 rounded-full flex items-center justify-center relative`}
                    >
                        <div className={`w-48 h-48 border border-${color}-500/50 rounded-full border-dashed`} />
                        <div className={`absolute top-0 w-1 h-4 bg-${color}-500`} />
                        <div className={`absolute bottom-0 w-1 h-4 bg-${color}-500`} />
                        <div className={`absolute left-0 w-4 h-1 bg-${color}-500`} />
                        <div className={`absolute right-0 w-4 h-1 bg-${color}-500`} />

                        {/* Center Text */}
                        <div className="absolute text-center bg-black/60 backdrop-blur-md p-4 rounded-lg border border-white/10">
                            <div className="text-xs text-gray-400 uppercase tracking-widest mb-1">Risk Probability</div>
                            <div className={`text-4xl font-black text-${color}-500 font-mono`}>
                                {(result.probability * 100).toFixed(0)}%
                            </div>
                        </div>
                    </motion.div>
                </div>

                {/* Left Data Column */}
                <div className="absolute top-32 left-10 w-64 space-y-4">
                    <HudBox title="DOMAIN INTEL" color={color}>
                        <div className="space-y-2 mt-2 font-mono text-xs">
                            <div className="flex justify-between text-gray-300">
                                <span>AGE:</span>
                                <span className={result.domain_details?.age_days && result.domain_details.age_days < 30 ? "text-red-400" : "text-cyan-400"}>
                                    {result.domain_details?.age_days || "UNKNOWN"} DAYS
                                </span>
                            </div>
                            <div className="flex justify-between text-gray-300">
                                <span>DNS:</span>
                                <span className={result.domain_details?.dns_valid ? "text-green-400" : "text-red-400"}>
                                    {result.domain_details?.dns_valid ? "VERIFIED" : "FAILED"}
                                </span>
                            </div>
                            <div className="flex justify-between text-gray-300">
                                <span>LOC:</span>
                                <span className="text-cyan-400">{result.domain_details?.country || "N/A"}</span>
                            </div>
                        </div>
                    </HudBox>

                    <HudBox title="AI ANALYSIS" color={color}>
                        <div className="mt-2 space-y-1">
                            {[1, 2, 3].map(i => (
                                <div key={i} className="h-1 w-full bg-gray-800 rounded overflow-hidden">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${Math.random() * 100}%` }}
                                        transition={{ duration: 1, repeat: Infinity, repeatType: "reverse" }}
                                        className={`h-full bg-${color}-500/70`}
                                    />
                                </div>
                            ))}
                            <div className="flex justify-between text-xs font-mono text-gray-400 mt-2">
                                <span>CONFIDENCE</span>
                                <span>{(result.confidence * 100).toFixed(0)}%</span>
                            </div>
                        </div>
                    </HudBox>

                    {/* Deepfake DNA Module */}
                    {result.feature_scores?.deepfake_analysis && (
                        <HudBox title="DEEPFAKE DNA" color={result.feature_scores.deepfake_analysis.is_deepfake ? 'red' : 'green'}>
                            <div className="mt-2 font-mono text-xs">
                                <div className="flex justify-between text-gray-300">
                                    <span>STATUS:</span>
                                    <span className={result.feature_scores.deepfake_analysis.is_deepfake ? "text-red-500 animate-pulse font-bold" : "text-green-500"}>
                                        {result.feature_scores.deepfake_analysis.is_deepfake ? "SYNTHETIC" : "HUMAN"}
                                    </span>
                                </div>
                                {result.feature_scores.deepfake_analysis.is_deepfake && (
                                    <div className="text-[10px] text-red-400 mt-1">
                                        MARKERS: {result.feature_scores.deepfake_analysis.reasons.join(", ")}
                                    </div>
                                )}
                            </div>
                        </HudBox>
                    )}
                </div>

                {/* Right Data Column */}
                <div className="absolute top-32 right-10 w-64 space-y-4 text-right">
                    <HudBox title="THREAT SPECTRUM" color={color} align="right">
                        <div className="grid grid-cols-4 gap-1 mt-2">
                            {Array.from({ length: 16 }).map((_, i) => (
                                <div key={i} className={`h-4 w-full bg-${color}-500/${Math.random() > 0.5 ? '20' : '80'}`} />
                            ))}
                        </div>
                    </HudBox>

                    {isHighRisk && (
                        <motion.div
                            animate={{ opacity: [1, 0.5, 1] }}
                            transition={{ duration: 0.5, repeat: Infinity }}
                            className="border border-red-500 bg-red-500/10 p-4 rounded"
                        >
                            <div className="text-red-500 font-bold font-mono text-sm flex items-center justify-end gap-2">
                                <span>ACTIVE DEFENSE READY</span>
                                <Shield className="w-4 h-4" />
                            </div>
                            <div className="text-[10px] text-red-400 mt-1 font-mono">
                                POISON PILL INJECTION MODULE ONLINE. AWAITING AUTHORIZATION.
                            </div>
                        </motion.div>
                    )}
                </div>

                {/* Bottom Status Bar */}
                <div className="absolute bottom-10 left-10 right-10 h-12 flex items-center gap-4 border-t border-gray-800 bg-black/60 backdrop-blur-md px-4">
                    <Wifi className={`w-4 h-4 text-${color}-500`} />
                    <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                        <motion.div
                            initial={{ width: "0%" }}
                            animate={{ width: "100%" }}
                            className={`h-full bg-${color}-500`}
                        />
                    </div>
                    <div className={`font-mono text-xs text-${color}-500`}>
                        SCAN_COMPLETE_ID_{Math.random().toString(36).substr(2, 9).toUpperCase()}
                    </div>
                </div>

            </div>
        </motion.div>
    );
};

const CornerBrackets = ({ color }: { color: string }) => (
    <>
        <div className={`absolute top-4 left-4 w-16 h-16 border-t-2 border-l-2 border-${color}-500 rounded-tl-lg`} />
        <div className={`absolute top-4 right-4 w-16 h-16 border-t-2 border-r-2 border-${color}-500 rounded-tr-lg`} />
        <div className={`absolute bottom-4 left-4 w-16 h-16 border-b-2 border-l-2 border-${color}-500 rounded-bl-lg`} />
        <div className={`absolute bottom-4 right-4 w-16 h-16 border-b-2 border-r-2 border-${color}-500 rounded-br-lg`} />
    </>
);

const HudBox = ({ title, children, color, align = 'left' }: { title: string, children: React.ReactNode, color: string, align?: 'left' | 'right' }) => (
    <div className={`bg-black/40 border-l-2 border-${color}-500/50 p-3 backdrop-blur-sm`}>
        <h3 className={`text-${color}-400 font-mono text-xs font-bold tracking-widest mb-1 ${align === 'right' ? 'text-right' : ''}`}>
            {title}
        </h3>
        {children}
    </div>
);

export default AROverlay;
