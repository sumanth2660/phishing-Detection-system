import React, { useState, useEffect } from 'react';
import { Plus, Send, BarChart2, Users, Mail } from 'lucide-react';
import { useForm } from 'react-hook-form';
import toast from 'react-hot-toast';

interface Simulation {
    id: string;
    name: string;
    status: string;
    created_at: string;
    stats: {
        total: number;
        sent: number;
        clicked: number;
    };
}

interface CreateSimulationForm {
    name: string;
    template_subject: string;
    template_body: string;
    target_emails: string; // Comma separated
}

export default function Simulation() {
    const [simulations, setSimulations] = useState<Simulation[]>([]);
    const [isCreating, setIsCreating] = useState(false);
    const { register, handleSubmit, reset } = useForm<CreateSimulationForm>();

    useEffect(() => {
        fetchSimulations();
    }, []);

    const fetchSimulations = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:8000/api/v1/simulation/list', {
                headers: { Authorization: `Bearer ${token}` }
            });
            if (response.ok) {
                const data = await response.json();
                setSimulations(data);
            }
        } catch (error) {
            console.error('Failed to fetch simulations', error);
        }
    };

    const onSubmit = async (data: CreateSimulationForm) => {
        try {
            const token = localStorage.getItem('token');
            const emails = data.target_emails.split(',').map(e => e.trim());

            const response = await fetch('http://localhost:8000/api/v1/simulation/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`
                },
                body: JSON.stringify({
                    name: data.name,
                    template_subject: data.template_subject,
                    template_body: data.template_body,
                    target_emails: emails
                })
            });

            if (response.ok) {
                toast.success('Campaign created successfully!');
                setIsCreating(false);
                reset();
                fetchSimulations();
            } else {
                toast.error('Failed to create campaign');
            }
        } catch (error) {
            toast.error('Error creating campaign');
        }
    };

    const triggerSend = async (id: string) => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch(`http://localhost:8000/api/v1/simulation/send/${id}`, {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` }
            });
            if (response.ok) {
                toast.success('Emails sending in background');
                fetchSimulations();
            }
        } catch (error) {
            toast.error('Failed to trigger send');
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Phishing Simulation</h1>
                    <p className="text-gray-500 mt-1">Train your team with simulated phishing campaigns</p>
                </div>
                <button
                    onClick={() => setIsCreating(!isCreating)}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
                >
                    <Plus className="w-4 h-4" />
                    New Campaign
                </button>
            </div>

            {isCreating && (
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                    <h2 className="text-lg font-semibold mb-4">Create New Campaign</h2>
                    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700">Campaign Name</label>
                            <input {...register('name')} className="mt-1 w-full p-2 border rounded-md" placeholder="e.g., Q4 Security Test" required />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700">Email Subject</label>
                            <input {...register('template_subject')} className="mt-1 w-full p-2 border rounded-md" placeholder="Urgent: Password Reset Required" required />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700">Email Body</label>
                            <textarea {...register('template_body')} className="mt-1 w-full p-2 border rounded-md h-32" placeholder="Click here to reset your password..." required />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700">Target Emails (comma separated)</label>
                            <input {...register('target_emails')} className="mt-1 w-full p-2 border rounded-md" placeholder="employee1@company.com, employee2@company.com" required />
                        </div>
                        <div className="flex justify-end gap-2">
                            <button type="button" onClick={() => setIsCreating(false)} className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg">Cancel</button>
                            <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">Create Campaign</button>
                        </div>
                    </form>
                </div>
            )}

            <div className="grid gap-6">
                {simulations.map((sim) => (
                    <div key={sim.id} className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                        <div className="flex justify-between items-start mb-4">
                            <div>
                                <h3 className="text-lg font-semibold text-gray-900">{sim.name}</h3>
                                <p className="text-sm text-gray-500">Created: {new Date(sim.created_at).toLocaleDateString()}</p>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${sim.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'
                                    }`}>
                                    {sim.status.toUpperCase()}
                                </span>
                                <button
                                    onClick={() => triggerSend(sim.id)}
                                    className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg"
                                    title="Send Emails"
                                >
                                    <Send className="w-4 h-4" />
                                </button>
                            </div>
                        </div>

                        <div className="grid grid-cols-3 gap-4">
                            <div className="p-4 bg-gray-50 rounded-lg">
                                <div className="flex items-center gap-2 text-gray-600 mb-1">
                                    <Users className="w-4 h-4" />
                                    <span className="text-sm font-medium">Targets</span>
                                </div>
                                <p className="text-2xl font-bold text-gray-900">{sim.stats.total}</p>
                            </div>
                            <div className="p-4 bg-blue-50 rounded-lg">
                                <div className="flex items-center gap-2 text-blue-600 mb-1">
                                    <Mail className="w-4 h-4" />
                                    <span className="text-sm font-medium">Sent</span>
                                </div>
                                <p className="text-2xl font-bold text-blue-900">{sim.stats.sent}</p>
                            </div>
                            <div className="p-4 bg-red-50 rounded-lg">
                                <div className="flex items-center gap-2 text-red-600 mb-1">
                                    <BarChart2 className="w-4 h-4" />
                                    <span className="text-sm font-medium">Clicked</span>
                                </div>
                                <p className="text-2xl font-bold text-red-900">{sim.stats.clicked}</p>
                            </div>
                        </div>
                    </div>
                ))}

                {simulations.length === 0 && !isCreating && (
                    <div className="text-center py-12 bg-gray-50 rounded-xl border-2 border-dashed border-gray-200">
                        <p className="text-gray-500">No campaigns yet. Create one to get started!</p>
                    </div>
                )}
            </div>
        </div>
    );
}
