import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link2, Shield, Database, Lock, Clock, AlertTriangle, Plus, Server } from 'lucide-react';
import toast from 'react-hot-toast';

interface Threat {
    url: string;
    risk_level: string;
    source: string;
    description?: string;
}

interface Block {
    index: number;
    timestamp: number;
    proof: number;
    previous_hash: string;
    threats: Threat[];
}

export default function ThreatLedger() {
    const [chain, setChain] = useState<Block[]>([]);
    const [pendingThreats, setPendingThreats] = useState<Threat[]>([]);
    const [isMining, setIsMining] = useState(false);
    const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
    const [manualThreatUrl, setManualThreatUrl] = useState('');

    useEffect(() => {
        fetchChain();
        const interval = setInterval(fetchChain, 5000);
        return () => clearInterval(interval);
    }, []);

    const fetchChain = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:8000/api/v1/ledger/chain', {
                headers: { Authorization: `Bearer ${token}` }
            });
            if (response.ok) {
                const data = await response.json();
                setChain(data.chain);
                setPendingThreats(data.pending_threats);
                setLastUpdated(new Date());
            }
        } catch (error) {
            console.error('Failed to fetch blockchain', error);
        }
    };

    const handleMineBlock = async () => {
        if (pendingThreats.length === 0) {
            toast.error("No pending threats to mine!");
            return;
        }

        setIsMining(true);
        try {
            const token = localStorage.getItem('token');
            // Simulate mining delay for effect
            await new Promise(resolve => setTimeout(resolve, 2000));

            const response = await fetch('http://localhost:8000/api/v1/ledger/mine', {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` }
            });

            if (response.ok) {
                const data = await response.json();
                toast.success(`Block #${data.index} successfully mined!`);
                fetchChain();
            } else {
                toast.error('Mining failed');
            }
        } catch (error) {
            toast.error('Mining error');
        } finally {
            setIsMining(false);
        }
    };

    const handleAddManualThreat = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!manualThreatUrl) return;

        try {
            const token = localStorage.getItem('token');
            const threatData = {
                url: manualThreatUrl,
                risk_level: "High",
                source: "Manual Report",
                details: { reporter: "Admin User", type: "Phishing" }
            };

            const response = await fetch('http://localhost:8000/api/v1/ledger/add_threat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`
                },
                body: JSON.stringify(threatData)
            });

            if (response.ok) {
                toast.success('Threat added to pending pool');
                setManualThreatUrl('');
                fetchChain();
            }
        } catch (error) {
            toast.error('Failed to add threat');
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 p-8 space-y-8">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gradient-to-r from-slate-900 to-slate-800 rounded-3xl p-8 text-white shadow-xl relative overflow-hidden"
            >
                <div className="relative z-10 flex justify-between items-center">
                    <div>
                        <h1 className="text-4xl font-bold flex items-center gap-3">
                            <Link2 className="w-10 h-10 text-cyan-400" />
                            Blockchain Threat Ledger
                        </h1>
                        <p className="text-slate-400 mt-2 max-w-2xl">
                            Immutable, decentralized record of confirmed phishing threats.
                            Uses SHA-256 Proof-of-Work to verify and seal threat intelligence.
                        </p>
                    </div>
                    <div className="text-right hidden lg:block">
                        <div className="bg-slate-800/50 backdrop-blur-sm p-4 rounded-xl border border-slate-700">
                            <div className="text-sm text-slate-400">Total Blocks Mined</div>
                            <div className="text-3xl font-mono font-bold text-cyan-400">{chain.length}</div>
                        </div>
                    </div>
                </div>
                {/* Decorative background */}
                <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl -mr-16 -mt-16" />
            </motion.div>

            <div className="grid lg:grid-cols-3 gap-8">
                {/* Pending Threats Pool */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="lg:col-span-1 space-y-6"
                >
                    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
                        <h2 className="text-xl font-bold flex items-center gap-2 mb-4">
                            <Database className="w-5 h-5 text-orange-500" />
                            Pending Threat Pool
                        </h2>

                        <div className="bg-orange-50 rounded-xl p-4 mb-6 border border-orange-100">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-sm font-medium text-orange-800">Unconfirmed Transactions</span>
                                <span className="text-xs bg-white px-2 py-1 rounded-full text-orange-600 font-bold">
                                    {pendingThreats.length}
                                </span>
                            </div>
                            <div className="space-y-2">
                                {pendingThreats.length === 0 ? (
                                    <p className="text-xs text-orange-400 italic text-center py-4">Pool is empty</p>
                                ) : (
                                    pendingThreats.map((t, i) => (
                                        <div key={i} className="text-xs bg-white p-2 rounded border border-orange-100 truncate shadow-sm">
                                            {t.url}
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>

                        <button
                            onClick={handleMineBlock}
                            disabled={isMining || pendingThreats.length === 0}
                            className={`w-full py-4 rounded-xl font-bold text-lg shadow-lg flex items-center justify-center gap-2 transition-all ${isMining || pendingThreats.length === 0
                                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                                    : 'bg-gradient-to-r from-orange-500 to-amber-500 text-white hover:shadow-orange-200 hover:scale-[1.02]'
                                }`}
                        >
                            {isMining ? (
                                <>
                                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                    Solving Proof of Work...
                                </>
                            ) : (
                                <>
                                    <Server className="w-5 h-5" />
                                    Mine Pending Block
                                </>
                            )}
                        </button>
                    </div>

                    {/* Manual Add Tool */}
                    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
                        <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4">Report New Threat</h3>
                        <form onSubmit={handleAddManualThreat} className="flex gap-2">
                            <input
                                type="text"
                                value={manualThreatUrl}
                                onChange={(e) => setManualThreatUrl(e.target.value)}
                                placeholder="http://malicious-site.com"
                                className="flex-1 px-4 py-2 bg-gray-50 rounded-lg border border-gray-200 text-sm focus:ring-2 focus:ring-cyan-500"
                            />
                            <button type="submit" className="p-2 bg-slate-900 text-white rounded-lg hover:bg-slate-800">
                                <Plus className="w-5 h-5" />
                            </button>
                        </form>
                    </div>
                </motion.div>

                {/* Blockchain Visualization */}
                <div className="lg:col-span-2">
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-xl font-bold flex items-center gap-2">
                            <Shield className="w-5 h-5 text-cyan-600" />
                            Confirmed Ledger
                        </h2>
                        <div className="text-xs text-gray-400 flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            Live Sync: {lastUpdated.toLocaleTimeString()}
                        </div>
                    </div>

                    <div className="space-y-4 relative">
                        {/* Connecting Line */}
                        <div className="absolute left-8 top-8 bottom-8 w-0.5 bg-gray-200 -z-10" />

                        <AnimatePresence>
                            {[...chain].reverse().map((block, i) => (
                                <motion.div
                                    key={block.index}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: i * 0.1 }}
                                    className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 ml-4 relative hover:shadow-md transition-shadow"
                                >
                                    {/* Connection Node */}
                                    <div className="absolute top-1/2 -left-[33px] -mt-3 w-6 h-6 rounded-full bg-cyan-100 border-4 border-white flex items-center justify-center">
                                        <div className="w-2 h-2 bg-cyan-500 rounded-full" />
                                    </div>

                                    <div className="flex justify-between items-start mb-4">
                                        <div>
                                            <div className="flex items-center gap-3">
                                                <span className="bg-slate-100 text-slate-600 px-2 py-1 rounded text-xs font-mono font-bold">
                                                    BLOCK #{block.index}
                                                </span>
                                                <span className="text-xs text-gray-400">
                                                    {new Date(block.timestamp * 1000).toLocaleString()}
                                                </span>
                                            </div>
                                            <div className="mt-2 font-mono text-[10px] text-gray-500 flex flex-col gap-1">
                                                <div><span className="font-bold text-gray-400">PREV:</span> {block.previous_hash}</div>
                                                <div><span className="font-bold text-cyan-600">HASH:</span> {block.previous_hash.substring(0, 10)}...{block.previous_hash.substring(54)} (Dynamic in real app)</div>
                                            </div>
                                        </div>
                                        <div className="bg-cyan-50 px-3 py-1 rounded-full text-cyan-700 text-xs font-bold border border-cyan-100">
                                            Proof: {block.proof}
                                        </div>
                                    </div>

                                    <div className="space-y-2">
                                        {block.threats.length === 0 ? (
                                            <div className="text-sm text-gray-400 italic">Genesis / Empty Block</div>
                                        ) : (
                                            block.threats.map((threat, idx) => (
                                                <div key={idx} className="flex items-center gap-3 p-2 bg-red-50/50 rounded-lg border border-red-50">
                                                    <AlertTriangle className="w-4 h-4 text-red-500" />
                                                    <span className="text-sm font-medium text-gray-700 truncate flex-1">{threat.url}</span>
                                                    <span className="text-xs text-red-600 font-bold px-2 py-0.5 bg-red-100 rounded-full">
                                                        {threat.risk_level}
                                                    </span>
                                                </div>
                                            ))
                                        )}
                                    </div>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </div>
                </div>
            </div>
        </div>
    );
}
