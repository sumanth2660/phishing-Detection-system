import React, { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import {
  Search,
  Image as ImageIcon,
  FileText,
  Globe,
  Loader2,
  AlertTriangle,
  CheckCircle,
  Shield,
  Activity
} from 'lucide-react'
import toast from 'react-hot-toast'

const urlSchema = z.object({
  url: z.string().url('Please enter a valid URL')
})

type URLFormData = z.infer<typeof urlSchema>

interface AnalysisResult {
  id: string;
  probability: number;
  risk_level: string;
  confidence: number;
  reasons: {
    feature: string;
    contribution: number;
    description: string;
  }[];
  domain_details?: {
    domain: string;
    age_days: number;
    dns_valid: boolean;
    registrar: string;
    country: string;
  };
  explain_html?: string;
  url?: string;
  timestamp: string;
  processing_time_ms: number;
  feature_scores?: {
    heuristic: number;
    text_analysis: number;
    transcript?: string;
    deepfake_analysis?: {
      is_deepfake: boolean;
      confidence: number;
      reasons: string[];
      pitch_volatility?: string;
      spectral_flatness?: string;
    };
  };
}

const URLScanner: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'url' | 'image' | 'document'>('url')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<AnalysisResult[]>([])

  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const { register, handleSubmit, getValues, formState: { errors } } = useForm<URLFormData>({
    resolver: zodResolver(urlSchema)
  })

  const onSubmit = async (data: URLFormData) => {
    setIsAnalyzing(true)
    setResults([])
    try {
      const response = await fetch('http://localhost:8000/api/v1/predict/url', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: data.url }),
      })

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const resultData = await response.json()
      // Wrap single result in array
      setResults([resultData])
      toast.success('Analysis completed')
    } catch (error) {
      console.error('Error:', error)
      toast.error('Analysis failed. Please check the backend connection.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0])
      setResults([]) // Clear previous results
    }
  }

  const handleAnalyzeFile = async (type: 'image' | 'document') => {
    if (!selectedFile) return

    setIsAnalyzing(true)
    setResults([])
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const endpoint = type === 'image'
        ? 'http://localhost:8000/api/v1/predict/image'
        : 'http://localhost:8000/api/v1/predict/document'

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const resultData = await response.json()
      // Backend now returns a list for these endpoints
      if (Array.isArray(resultData)) {
        setResults(resultData)
        if (resultData.length === 0) {
          toast('No URLs found in the file', { icon: 'ℹ️' })
        } else {
          toast.success(`Found and analyzed ${resultData.length} URLs`)
        }
      } else {
        // Fallback if backend returns single object (shouldn't happen with new backend)
        setResults([resultData])
      }
    } catch (error) {
      console.error('Error:', error)
      toast.error('Analysis failed. Please check the backend connection.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Advanced URL Analysis</h1>
        <p className="text-gray-500 mt-1">Multi-modal URL analysis with dual AI verification, image OCR, and document scanning</p>
      </div>


      {/* Tabs */}
      <div className="bg-white rounded-lg p-1 shadow-sm border border-gray-200 inline-flex w-full">
        <button
          onClick={() => { setActiveTab('url'); setResults([]); setSelectedFile(null); }}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${activeTab === 'url'
            ? 'bg-white text-gray-900 shadow-sm border border-gray-200'
            : 'text-gray-500 hover:text-gray-900'
            }`}
        >
          URL Analysis
        </button>
        <button
          onClick={() => { setActiveTab('image'); setResults([]); setSelectedFile(null); }}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${activeTab === 'image'
            ? 'bg-white text-gray-900 shadow-sm border border-gray-200'
            : 'text-gray-500 hover:text-gray-900'
            }`}
        >
          Image Analysis
        </button>
        <button
          onClick={() => { setActiveTab('document'); setResults([]); setSelectedFile(null); }}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${activeTab === 'document'
            ? 'bg-white text-gray-900 shadow-sm border border-gray-200'
            : 'text-gray-500 hover:text-gray-900'
            }`}
        >
          Document Analysis
        </button>
      </div>

      {/* Main Content */}
      <div className="card p-8 min-h-[400px]">
        <div className="max-w-3xl mx-auto space-y-8">
          <div className="text-center space-y-2">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center justify-center gap-2">
              <Search className="w-5 h-5" />
              Advanced URL Analysis with Dual AI Verification
            </h2>
          </div>

          {activeTab === 'url' && (
            <form onSubmit={handleSubmit(onSubmit)} className="flex gap-4">
              <div className="flex-1 relative">
                <input
                  {...register('url')}
                  type="text"
                  placeholder="Enter URL to analyze (e.g., https://example.com)"
                  className="w-full h-12 pl-4 pr-4 rounded-lg border border-gray-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all text-gray-900"
                  disabled={isAnalyzing}
                />
                {errors.url && (
                  <p className="absolute -bottom-6 left-0 text-red-500 text-sm">{errors.url.message}</p>
                )}
              </div>
              <button
                type="submit"
                disabled={isAnalyzing}
                className="h-12 px-6 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4" />
                    Analyzing...
                  </>
                ) : (
                  'Analyze URL'
                )}
              </button>
            </form>
          )}

          {activeTab === 'image' && (
            <div className="space-y-4">
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-500 transition-colors bg-gray-50">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="image-upload"
                  disabled={isAnalyzing}
                />
                <label htmlFor="image-upload" className="cursor-pointer flex flex-col items-center gap-4">
                  <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center">
                    <ImageIcon className="w-8 h-8" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      {selectedFile ? selectedFile.name : 'Upload Screenshot or QR Code'}
                    </h3>
                    <p className="text-gray-500 mt-1">Supports PNG, JPG, GIF up to 10MB</p>
                  </div>
                  <div className="px-6 py-2 bg-white border border-gray-300 rounded-lg text-gray-700 font-medium hover:bg-gray-50 transition-colors">
                    {selectedFile ? 'Change Image' : 'Select Image'}
                  </div>
                </label>
              </div>

              {selectedFile && (
                <div className="flex justify-center">
                  <button
                    onClick={() => handleAnalyzeFile('image')}
                    disabled={isAnalyzing}
                    className="h-12 px-8 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-200"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="w-4 h-4" />
                        Analyzing Image...
                      </>
                    ) : (
                      <>
                        <Search className="w-4 h-4" />
                        Analyze Image
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          )}

          {activeTab === 'document' && (
            <div className="space-y-4">
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-500 transition-colors bg-gray-50">
                <input
                  type="file"
                  accept=".pdf,.doc,.docx,.txt"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="doc-upload"
                  disabled={isAnalyzing}
                />
                <label htmlFor="doc-upload" className="cursor-pointer flex flex-col items-center gap-4">
                  <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center">
                    <FileText className="w-8 h-8" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      {selectedFile ? selectedFile.name : 'Upload Document'}
                    </h3>
                    <p className="text-gray-500 mt-1">Scan PDF or Word docs for malicious links</p>
                  </div>
                  <div className="px-6 py-2 bg-white border border-gray-300 rounded-lg text-gray-700 font-medium hover:bg-gray-50 transition-colors">
                    {selectedFile ? 'Change Document' : 'Select Document'}
                  </div>
                </label>
              </div>

              {selectedFile && (
                <div className="flex justify-center">
                  <button
                    onClick={() => handleAnalyzeFile('document')}
                    disabled={isAnalyzing}
                    className="h-12 px-8 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-200"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="w-4 h-4" />
                        Analyzing Document...
                      </>
                    ) : (
                      <>
                        <Search className="w-4 h-4" />
                        Analyze Document
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          )}


          {/* Results Section */}
          {results.length > 0 && (
            <div className="space-y-8 pt-8 border-t border-gray-100">
              {results.map((result, index) => (
                <div
                  key={result.id || index}
                  className="space-y-6"
                >
                  {/* URL Header if multiple results or if URL is available */}
                  {(results.length > 1 || result.url) && (
                    <div className="flex items-center gap-2 pb-2 border-b border-gray-100">
                      <span className="bg-gray-100 text-gray-600 px-2 py-1 rounded text-xs font-bold">#{index + 1}</span>
                      <h3 className="font-mono text-sm text-gray-700 truncate max-w-md" title={result.url || 'Analyzed URL'}>
                        {result.url || 'Analyzed URL'}
                      </h3>
                    </div>
                  )}

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Risk Score */}
                    <div className="p-6 bg-gray-50 rounded-xl text-center">
                      <p className="text-sm font-medium text-gray-500 mb-2">Risk Score</p>
                      <div className="flex items-center justify-center gap-2">
                        <span className={`text-3xl font-bold ${result.probability > 0.7 ? 'text-red-600' :
                          result.probability > 0.4 ? 'text-orange-600' : 'text-green-600'
                          }`}>
                          {Math.round(result.probability * 100)}%
                        </span>
                      </div>
                      <div className={`mt-2 inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium ${result.risk_level === 'critical' || result.risk_level === 'high'
                        ? 'bg-red-100 text-red-700'
                        : result.risk_level === 'medium'
                          ? 'bg-orange-100 text-orange-700'
                          : 'bg-green-100 text-green-700'
                        }`}>
                        {result.risk_level.toUpperCase()} RISK
                      </div>
                    </div>

                    {/* Confidence */}
                    <div className="p-6 bg-gray-50 rounded-xl text-center">
                      <p className="text-sm font-medium text-gray-500 mb-2">AI Confidence</p>
                      <div className="flex items-center justify-center gap-2">
                        <span className="text-3xl font-bold text-gray-900">
                          {Math.round(result.confidence * 100)}%
                        </span>
                      </div>
                      <p className="text-xs text-gray-400 mt-2">Model Certainty</p>
                    </div>

                    {/* Processing Time */}
                    <div className="p-6 bg-gray-50 rounded-xl text-center">
                      <p className="text-sm font-medium text-gray-500 mb-2">Processing Time</p>
                      <div className="flex items-center justify-center gap-2">
                        <span className="text-3xl font-bold text-gray-900">
                          {Math.round(result.processing_time_ms)}ms
                        </span>
                      </div>
                      <p className="text-xs text-gray-400 mt-2">Real-time Analysis</p>
                    </div>
                  </div>

                  {/* Analysis Details */}
                  <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Details</h3>
                    <div className="space-y-4">
                      {result.reasons.map((reason, rIndex) => (
                        <div key={rIndex} className="flex items-start gap-4 p-4 bg-gray-50 rounded-lg">
                          <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm
                          ${reason.contribution > 0 ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'}`}>
                            {rIndex + 1}
                          </div>
                          <div>
                            <h4 className="font-medium text-gray-900">
                              {reason.feature}
                              <span className="ml-2 text-xs text-gray-500">
                                ({Math.abs(Math.round(reason.contribution * 100))}% impact)
                              </span>
                            </h4>
                            <p className="text-sm text-gray-600 mt-1">{reason.description}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Active Defense (Poison Pill) */}
                  {result.probability > 0.8 && (
                    <div className="bg-red-50 border-2 border-red-200 rounded-xl p-6 relative overflow-hidden">
                      <div className="absolute top-0 right-0 p-4 opacity-10">
                        <Activity className="w-24 h-24" />
                      </div>

                      <div className="relative z-10">
                        <h3 className="text-lg font-bold text-red-800 flex items-center gap-2">
                          <Shield className="w-5 h-5" />
                          Active Cyber Defense Available
                        </h3>
                        <p className="text-red-700 mt-2 mb-4 text-sm max-w-xl">
                          This site has been identified as a <strong>High Risk Phishing Attack</strong>.
                          You can authorize an automated counter-measure to flood their database with fake credentials,
                          protecting real victims.
                        </p>

                        <button
                          onClick={async () => {
                            if (!confirm("⚠️ WARNING: Only use this on verified phishing sites.\n\nThis will flood the target server with fake data. Are you sure?")) return;

                            const toastId = toast.loading("Deploying Counter-Measures...");
                            try {
                              const currentUrl = result.url || getValues('url');
                              const res = await fetch('http://localhost:8000/api/v1/active-defense', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ url: currentUrl })
                              });
                              const d = await res.json();
                              if (d.sent > 0) {
                                toast.success(`Attack Successful! Injected ${d.sent} fake records.`, { id: toastId });
                              } else {
                                toast.error(`Defense Failed: ${d.reason || 'Target immune'}`, { id: toastId });
                              }
                            } catch (e) {
                              toast.error("Connection Failed", { id: toastId });
                            }
                          }}
                          className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-bold rounded-lg shadow-lg shadow-red-200 flex items-center gap-2"
                        >
                          <Activity className="w-5 h-5" />
                          DEPLOY POISON PILL
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Domain Intelligence */}
                  {result.domain_details && (
                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">Domain Intelligence</h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="p-4 bg-gray-50 rounded-lg">
                          <div className="text-sm text-gray-500 mb-1">Domain Age</div>
                          <div className="font-medium text-gray-900">
                            {result.domain_details.age_days > 0 ? `${result.domain_details.age_days} days` : 'Unknown'}
                          </div>
                        </div>
                        <div className="p-4 bg-gray-50 rounded-lg">
                          <div className="text-sm text-gray-500 mb-1">DNS Status</div>
                          <div className={`font-medium ${result.domain_details.dns_valid ? 'text-green-600' : 'text-red-600'}`}>
                            {result.domain_details.dns_valid ? 'Valid Records' : 'No Records Found'}
                          </div>
                        </div>
                        <div className="p-4 bg-gray-50 rounded-lg">
                          <div className="text-sm text-gray-500 mb-1">Registrar</div>
                          <div className="font-medium text-gray-900">{result.domain_details.registrar || 'Unknown'}</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

    </div>

  )
}

export default URLScanner