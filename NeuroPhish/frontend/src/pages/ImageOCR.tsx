import React, { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import {
  Image as ImageIcon,
  Upload,
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap,
  Maximize2,
  Type,
  Link as LinkIcon,
  Monitor,
  Copy,
  Download
} from 'lucide-react'
import toast from 'react-hot-toast'

interface ImageAnalysisResult {
  id: string
  probability: number
  risk_level: string
  confidence: number
  reasons: Array<{
    feature: string
    contribution: number
    description: string
  }>
  url?: string
  explain_html?: string
  timestamp: string
  processing_time_ms: number
  feature_scores?: {
    heuristic: number
    text_analysis: number
  }
}

const ImageOCR: React.FC = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<ImageAnalysisResult[]>([])
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      const reader = new FileReader()
      reader.onload = (event) => {
        setSelectedImage(event.target?.result as string)
      }
      reader.readAsDataURL(selectedFile)
      setResults([])
    }
  }

  const handleAnalyze = async () => {
    if (!file) {
      toast.error('Please select an image first')
      return
    }

    setIsAnalyzing(true)
    setResults([])

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://localhost:8000/api/v1/predict/image', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const data = await response.json()
      setResults(Array.isArray(data) ? data : [data])
      toast.success('Image analysis completed')
    } catch (error) {
      console.error('Error:', error)
      toast.error('Analysis failed. Please check the backend connection.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'critical': return 'text-red-400 bg-red-500/20 border-red-500/50'
      case 'high': return 'text-orange-400 bg-orange-500/20 border-orange-500/50'
      case 'medium': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/50'
      case 'low': return 'text-green-400 bg-green-500/20 border-green-500/50'
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/50'
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Visual Threat Inspector</h1>
          <p className="text-gray-400 mt-1">Extract and analyze hidden phishing URLs from images and screenshots</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-sm text-gray-400">Images Scanned Today</p>
            <p className="text-xl font-bold text-gray-900">156</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <motion.div
          className="card p-6"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Upload className="w-5 h-5 text-neon-blue" />
            Upload Evidence
          </h2>

          <div
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all ${selectedImage ? 'border-neon-purple bg-neon-purple/5' : 'border-dark-border hover:border-gray-600'
              }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              className="hidden"
            />
            {selectedImage ? (
              <div className="relative group">
                <img src={selectedImage} alt="Selected" className="max-h-[300px] mx-auto rounded-lg shadow-2xl" />
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity rounded-lg">
                  <p className="text-white text-sm font-medium">Click to change image</p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="w-16 h-16 bg-dark-card rounded-full flex items-center justify-center mx-auto shadow-inner">
                  <ImageIcon className="w-8 h-8 text-gray-500" />
                </div>
                <div>
                  <p className="text-gray-900 font-medium">Drop your image here or click to browse</p>
                  <p className="text-gray-400 text-sm mt-1">Supports PNG, JPEG, GIF and Screenshots</p>
                </div>
              </div>
            )}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!file || isAnalyzing}
            className={`btn-primary w-full py-4 mt-6 flex items-center justify-center gap-2 ${(!file || isAnalyzing) ? 'opacity-50 cursor-not-allowed' : ''
              }`}
          >
            {isAnalyzing ? (
              <>
                <Zap className="w-5 h-5 animate-spin text-white" />
                Processing Pixels...
              </>
            ) : (
              <>
                <Shield className="w-5 h-5" />
                Perform Deep Scanning
              </>
            )}
          </button>
        </motion.div>

        {/* Real-time Status */}
        <div className="space-y-4">
          <div className="card p-6 border-l-4 border-l-neon-blue h-full">
            <h3 className="text-lg font-semibold text-gray-900 mb-6 flex items-center gap-2">
              <Monitor className="w-5 h-5 text-neon-blue" />
              Extraction Monitor
            </h3>

            <div className="space-y-6">
              <div className="flex items-start gap-4 p-4 bg-dark-card rounded-lg relative overflow-hidden group">
                <div className="absolute inset-0 bg-blue-500/5 transform -translate-x-full group-hover:translate-x-0 transition-transform duration-500"></div>
                <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
                  <Type className="w-5 h-5 text-blue-400" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-1">OCR Engine</h4>
                  <p className="text-sm text-gray-300">Advanced character recognition active. Extracting textual patterns from visual data.</p>
                </div>
              </div>

              <div className="flex items-start gap-4 p-4 bg-dark-card rounded-lg relative overflow-hidden group">
                <div className="absolute inset-0 bg-purple-500/5 transform -translate-x-full group-hover:translate-x-0 transition-transform duration-500"></div>
                <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center flex-shrink-0">
                  <LinkIcon className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-1">URL Reconstructor</h4>
                  <p className="text-sm text-gray-300">De-obfuscating visual links and reconstructing full destination headers.</p>
                </div>
              </div>

              <div className="p-4 bg-neon-blue/5 rounded-lg border border-neon-blue/20">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-neon-blue uppercase">Analysis Pipeline Status</span>
                  <span className="text-[10px] text-gray-500 italic">Version 2.4.0-Stable</span>
                </div>
                <div className="flex gap-1 h-1.5">
                  <div className="flex-1 bg-neon-blue rounded-full"></div>
                  <div className={`flex-1 ${isAnalyzing ? 'bg-neon-blue animate-pulse' : 'bg-neon-blue'} rounded-full`}></div>
                  <div className={`flex-1 ${isAnalyzing ? 'bg-neon-purple animate-pulse' : 'bg-dark-border'} rounded-full`}></div>
                  <div className="flex-1 bg-dark-border rounded-full"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Results Matrix */}
      {results.length > 0 && (
        <motion.div
          className="space-y-6 pt-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-white">Analysis Matrix</h2>
            <span className="px-3 py-0.5 bg-dark-border rounded-full text-xs font-mono text-gray-400">
              {results.length} ENDPOINTS DETECTED
            </span>
          </div>

          <div className="grid grid-cols-1 gap-6">
            {results.map((result, index) => (
              <div key={index} className="card p-0 overflow-hidden group">
                <div className={`h-2 w-full ${result.risk_level === 'critical' ? 'bg-red-500' :
                    result.risk_level === 'high' ? 'bg-orange-500' : 'bg-yellow-500'
                  }`}></div>

                <div className="p-6">
                  <div className="flex flex-col md:flex-row gap-8 items-center">
                    <div className="flex-shrink-0">
                      <div className={`risk-meter ${result.risk_level}`}>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-gray-900">{Math.round(result.probability * 100)}%</div>
                          <div className="text-[8px] text-gray-400">THREAT LEVEL</div>
                        </div>
                      </div>
                    </div>

                    <div className="flex-1 space-y-4 w-full">
                      <div className="flex items-center justify-between">
                        <div className="space-y-1">
                          <p className="text-[10px] text-gray-500 uppercase font-bold tracking-widest">Target Endpoint</p>
                          <p className="text-lg font-mono text-neon-blue break-all">{result.url}</p>
                        </div>
                        <div className={`px-4 py-1 rounded-full border ${getRiskColor(result.risk_level)} text-xs font-bold uppercase`}>
                          {result.risk_level} Risk
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-3 bg-dark-card rounded-lg border border-dark-border">
                          <p className="text-[10px] text-gray-500 mb-1 font-bold">Confidence</p>
                          <p className="text-lg font-bold text-white">{Math.round(result.confidence * 100)}%</p>
                        </div>
                        <div className="p-3 bg-dark-card rounded-lg border border-dark-border">
                          <p className="text-[10px] text-gray-500 mb-1 font-bold">Heuristic</p>
                          <p className="text-lg font-bold text-white">{(result.feature_scores?.heuristic || 0).toFixed(2)}</p>
                        </div>
                        <div className="p-3 bg-dark-card rounded-lg border border-dark-border">
                          <p className="text-[10px] text-gray-500 mb-1 font-bold">Detection Ref</p>
                          <p className="text-xs font-mono text-gray-400 mt-1 uppercase">{result.id.split('-')[0]}</p>
                        </div>
                        <div className="p-3 bg-dark-card rounded-lg border border-dark-border flex items-center justify-center">
                          <button className="text-neon-blue hover:text-white transition-colors text-xs font-mono">
                            VIEW DETAILS →
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-8 pt-6 border-t border-dark-border">
                    <h4 className="text-xs font-bold text-gray-500 uppercase tracking-[0.2em] mb-4">Diagnostic Indicators</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {result.reasons.map((reason, rIdx) => (
                        <div key={rIdx} className="p-4 bg-dark-card/50 rounded-xl border border-dark-border/40 hover:border-neon-purple/30 transition-colors">
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-bold text-gray-200">{reason.feature}</span>
                            <span className="text-[10px] font-mono text-neon-purple">+{Math.round(reason.contribution * 100)}%</span>
                          </div>
                          <p className="text-xs text-gray-400 leading-relaxed font-medium">{reason.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  )
}

export default ImageOCR