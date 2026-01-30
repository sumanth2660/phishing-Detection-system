import React, { useState, useEffect } from 'react';
import { Shield, Upload, Eye, CheckCircle, AlertTriangle, Zap, Lock, Search } from 'lucide-react';
import toast from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';

interface Brand {
    id: string;
    name: string;
    created_at: string;
}

export default function BrandProtection() {
    const [brands, setBrands] = useState<Brand[]>([]);
    const [newBrandName, setNewBrandName] = useState('');
    const [uploadFile, setUploadFile] = useState<File | null>(null);
    const [testFile, setTestFile] = useState<File | null>(null);
    const [testResult, setTestResult] = useState<any>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    useEffect(() => {
        fetchBrands();
    }, []);

    const fetchBrands = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:8000/api/v1/vision/brands', {
                headers: { Authorization: `Bearer ${token}` }
            });
            if (response.ok) {
                const data = await response.json();
                setBrands(data);
            }
        } catch (error) {
            console.error('Failed to fetch brands', error);
        }
    };

    const handleAddBrand = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!uploadFile || !newBrandName) return;

        const formData = new FormData();
        formData.append('name', newBrandName);
        formData.append('file', uploadFile);

        try {
            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:8000/api/v1/vision/brands', {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
                body: formData
            });

            if (response.ok) {
                toast.success('Brand protected successfully!');
                setNewBrandName('');
                setUploadFile(null);
                fetchBrands();
            } else {
                toast.error('Failed to add brand');
            }
        } catch (error) {
            toast.error('Error uploading brand');
        }
    };

    const handleTestImage = async () => {
        if (!testFile) return;
        setIsAnalyzing(true);
        setTestResult(null);

        const formData = new FormData();
        formData.append('file', testFile);

        try {
            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:8000/api/v1/vision/analyze-image', {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
                body: formData
            });

            // Fake delay for dramatic effect
            await new Promise(resolve => setTimeout(resolve, 1500));

            if (response.ok) {
                const data = await response.json();
                setTestResult(data);
            }
        } catch (error) {
            toast.error('Analysis failed');
        } finally {
            setIsAnalyzing(false);
        }
    };

    const { getRootProps: getBrandProps, getInputProps: getBrandInputProps } = useDropzone({
        accept: { 'image/*': [] },
        onDrop: acceptedFiles => setUploadFile(acceptedFiles[0]),
        multiple: false
    });

    const { getRootProps: getTestProps, getInputProps: getTestInputProps } = useDropzone({
        accept: { 'image/*': [] },
        onDrop: acceptedFiles => setTestFile(acceptedFiles[0]),
        multiple: false
    });

    return (
        <div className="min-h-screen bg-gray-50 p-8 space-y-12">
            {/* Hero Section */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-indigo-600 to-purple-700 p-12 text-white shadow-2xl"
            >
                <div className="relative z-10">
                    <h1 className="text-5xl font-extrabold tracking-tight mb-4 flex items-center gap-4">
                        <Shield className="w-12 h-12 text-blue-300" />
                        Visual Brand Defense
                    </h1>
                    <p className="text-xl text-indigo-100 max-w-2xl">
                        Protect your brand identity using advanced Computer Vision.
                        Detect unauthorized usage of your logos and assets across the web in real-time.
                    </p>
                </div>

                {/* Decorative Background Elements */}
                <div className="absolute top-0 right-0 -mt-20 -mr-20 w-96 h-96 bg-white opacity-10 rounded-full blur-3xl" />
                <div className="absolute bottom-0 left-0 -mb-20 -ml-20 w-72 h-72 bg-blue-500 opacity-20 rounded-full blur-3xl" />
            </motion.div>

            <div className="grid lg:grid-cols-2 gap-8">
                {/* Left Column: Management */}
                <div className="space-y-8">
                    {/* Add Brand Card */}
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.1 }}
                        className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden"
                    >
                        <div className="p-6 border-b border-gray-100 bg-gray-50/50">
                            <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                                <Lock className="w-5 h-5 text-indigo-600" />
                                Register New Asset
                            </h2>
                        </div>
                        <div className="p-6">
                            <form onSubmit={handleAddBrand} className="space-y-6">
                                <div>
                                    <label className="block text-sm font-semibold text-gray-700 mb-2">Brand Name</label>
                                    <input
                                        type="text"
                                        value={newBrandName}
                                        onChange={(e) => setNewBrandName(e.target.value)}
                                        className="w-full px-4 py-3 rounded-xl border border-gray-200 bg-white text-gray-900 placeholder-gray-400 focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
                                        placeholder="e.g., Google, Microsoft"
                                        required
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-semibold text-gray-700 mb-2">Reference Logo</label>
                                    <div
                                        {...getBrandProps()}
                                        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${uploadFile ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-indigo-500 hover:bg-indigo-50'
                                            }`}
                                    >
                                        <input {...getBrandInputProps()} />
                                        {uploadFile ? (
                                            <div className="flex items-center justify-center gap-2 text-green-700 font-medium">
                                                <CheckCircle className="w-5 h-5" />
                                                {uploadFile.name}
                                            </div>
                                        ) : (
                                            <div className="space-y-2">
                                                <Upload className="w-8 h-8 text-gray-400 mx-auto" />
                                                <p className="text-gray-500">Drag & drop logo here, or click to select</p>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                <button
                                    type="submit"
                                    className="w-full py-3.5 bg-gray-900 text-white rounded-xl font-semibold hover:bg-gray-800 transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg"
                                >
                                    Protect This Brand
                                </button>
                            </form>
                        </div>
                    </motion.div>

                    {/* Protected Brands List */}
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                        className="bg-white rounded-2xl shadow-xl border border-gray-100 p-6"
                    >
                        <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                            <Shield className="w-5 h-5 text-green-600" />
                            Protected Assets ({brands.length})
                        </h2>
                        <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                            {brands.map((brand) => (
                                <motion.div
                                    key={brand.id}
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="flex items-center justify-between p-4 bg-gray-50 rounded-xl border border-gray-100 hover:shadow-md transition-all"
                                >
                                    <div className="flex items-center gap-3">
                                        <div className="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-700 font-bold">
                                            {brand.name[0]}
                                        </div>
                                        <div>
                                            <h3 className="font-semibold text-gray-900">{brand.name}</h3>
                                            <p className="text-xs text-gray-500">Protected since {new Date(brand.created_at).toLocaleDateString()}</p>
                                        </div>
                                    </div>
                                    <div className="px-3 py-1 rounded-full bg-green-100 text-green-700 text-xs font-medium flex items-center gap-1">
                                        <CheckCircle className="w-3 h-3" />
                                        Active
                                    </div>
                                </motion.div>
                            ))}
                            {brands.length === 0 && (
                                <div className="text-center py-8 text-gray-400">
                                    <Shield className="w-12 h-12 mx-auto mb-2 opacity-20" />
                                    <p>No brands protected yet.</p>
                                </div>
                            )}
                        </div>
                    </motion.div>
                </div>

                {/* Right Column: Testing */}
                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden flex flex-col h-full"
                >
                    <div className="p-6 border-b border-gray-100 bg-gray-50/50">
                        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                            <Eye className="w-5 h-5 text-purple-600" />
                            Live Detection Lab
                        </h2>
                    </div>

                    <div className="p-8 flex-1 flex flex-col">
                        <div
                            {...getTestProps()}
                            className={`flex-1 border-3 border-dashed rounded-2xl transition-all duration-300 flex flex-col items-center justify-center p-8 cursor-pointer relative overflow-hidden group ${testFile
                                ? 'border-purple-500 bg-purple-50'
                                : 'border-gray-200 hover:border-purple-400 hover:bg-gray-50'
                                }`}
                        >
                            <input {...getTestInputProps()} />

                            {testFile ? (
                                <div className="relative z-10 text-center">
                                    <img
                                        src={URL.createObjectURL(testFile)}
                                        alt="Preview"
                                        className="h-48 object-contain mb-4 rounded-lg shadow-sm mx-auto"
                                    />
                                    <p className="text-purple-700 font-medium">{testFile.name}</p>
                                    <p className="text-purple-500 text-sm mt-1">Click to change</p>
                                </div>
                            ) : (
                                <div className="text-center space-y-4 relative z-10">
                                    <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto group-hover:scale-110 transition-transform">
                                        <Search className="w-10 h-10 text-purple-600" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-semibold text-gray-900">Upload Screenshot</h3>
                                        <p className="text-gray-500 mt-1">Drop an image here to scan for brand impersonation</p>
                                    </div>
                                </div>
                            )}
                        </div>

                        <button
                            onClick={handleTestImage}
                            disabled={!testFile || isAnalyzing}
                            className={`mt-6 w-full py-4 rounded-xl font-bold text-lg shadow-lg flex items-center justify-center gap-2 transition-all ${!testFile
                                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                                : isAnalyzing
                                    ? 'bg-purple-700 text-white cursor-wait'
                                    : 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white hover:shadow-xl hover:scale-[1.02]'
                                }`}
                        >
                            {isAnalyzing ? (
                                <>
                                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                    Analyzing Visual Features...
                                </>
                            ) : (
                                <>
                                    <Zap className="w-5 h-5" />
                                    Run Visual Analysis
                                </>
                            )}
                        </button>

                        <AnimatePresence>
                            {testResult && (
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: 20 }}
                                    className={`mt-6 p-6 rounded-2xl border-2 ${testResult.detected
                                        ? 'bg-red-50 border-red-100'
                                        : 'bg-green-50 border-green-100'
                                        }`}
                                >
                                    <div className="flex items-start gap-4">
                                        <div className={`p-3 rounded-full ${testResult.detected ? 'bg-red-100' : 'bg-green-100'
                                            }`}>
                                            {testResult.detected ? (
                                                <AlertTriangle className="w-8 h-8 text-red-600" />
                                            ) : (
                                                <CheckCircle className="w-8 h-8 text-green-600" />
                                            )}
                                        </div>
                                        <div>
                                            <h3 className={`text-xl font-bold ${testResult.detected ? 'text-red-900' : 'text-green-900'
                                                }`}>
                                                {testResult.detected ? 'Impersonation Detected!' : 'No Threats Found'}
                                            </h3>
                                            <p className={`mt-1 ${testResult.detected ? 'text-red-700' : 'text-green-700'
                                                }`}>
                                                {testResult.detected
                                                    ? `This image contains visual elements matching "${testResult.brand}".`
                                                    : "This image does not match any protected brand assets."
                                                }
                                            </p>
                                            {testResult.detected && (
                                                <div className="mt-3 flex items-center gap-2 text-sm text-red-600 bg-red-100/50 px-3 py-1 rounded-full w-fit">
                                                    <span className="font-semibold">Confidence Score:</span>
                                                    {(testResult.confidence * 100).toFixed(1)}%
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
