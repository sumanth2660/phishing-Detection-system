import React, { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import {
  Mic,
  Upload,
  Play,
  Pause,
  AlertTriangle,
  CheckCircle,
  FileAudio,
  Loader2,
  X
} from 'lucide-react'
import toast from 'react-hot-toast'
import { AnimatePresence } from 'framer-motion'
import AROverlay from '../components/AROverlay'

// Mapped to match Backend PredictionResponse & AROverlay expectations
interface AudioAnalysisResult {
  id: string;
  probability: number;
  risk_level: string;
  confidence: number;
  reasons: {
    feature: string;
    contribution: number;
    description: string;
  }[];
  transcript?: string;
  detected_keywords?: string[];
  feature_scores?: {
    deepfake_analysis?: {
      is_deepfake: boolean;
      confidence: number;
      reasons: string[];
    };
  };
  explain_html?: string;
  processing_time_ms: number;
}

const AudioSTT = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<AudioAnalysisResult | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      if (selectedFile.size > 10 * 1024 * 1024) {
        toast.error('File size must be less than 10MB')
        return
      }
      setFile(selectedFile)
      setResult(null)
      setShowOverlay(false)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0]
      if (selectedFile.type.startsWith('audio/')) {
        setFile(selectedFile)
        setResult(null)
      } else {
        toast.error('Please upload an audio file')
      }
    }
  }

  const [showOverlay, setShowOverlay] = useState(false)

  const analyzeAudio = async () => {
    if (!file) return

    setIsAnalyzing(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      // Updated to correct backend endpoint
      const response = await fetch('http://localhost:8000/api/v1/predict/audio', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const data = await response.json()
      setResult(data)
      setShowOverlay(true)
      toast.success('Audio analysis completed')
    } catch (error: any) {
      console.error('Error:', error)
      toast.error(error.message || 'Failed to analyze audio')
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Voice Phishing Detection</h1>
        <p className="text-gray-500 mt-1">Analyze audio calls for vishing attempts and malicious intent</p>
      </div>

      <AnimatePresence>
        {showOverlay && result && (
          <AROverlay
            result={result}
            onClose={() => setShowOverlay(false)}
          />
        )}
      </AnimatePresence>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <motion.div
          className="card p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Audio
          </h2>

          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${file ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
              }`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="audio/*"
              className="hidden"
            />

            {file ? (
              <div className="flex flex-col items-center gap-4">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
                  <FileAudio className="w-8 h-8 text-blue-600" />
                </div>
                <div>
                  <p className="font-medium text-gray-900">{file.name}</p>
                  <p className="text-sm text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={analyzeAudio}
                    disabled={isAnalyzing}
                    className="btn-primary px-6 py-2 flex items-center gap-2"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Mic className="w-4 h-4" />
                        Analyze Audio
                      </>
                    )}
                  </button>
                  <button
                    onClick={() => setFile(null)}
                    className="btn-ghost p-2 text-red-500 hover:text-red-600 hover:bg-red-50"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4 cursor-pointer" onClick={() => fileInputRef.current?.click()}>
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center">
                  <Upload className="w-8 h-8 text-gray-400" />
                </div>
                <div>
                  <p className="font-medium text-gray-900">Click to upload or drag and drop</p>
                  <p className="text-sm text-gray-500">MP3, WAV, M4A (Max 10MB)</p>
                </div>
              </div>
            )}
          </div>
        </motion.div>

        {/* Results Section */}
        {result && (
          <motion.div
            className="card p-6"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <CheckCircle className="w-5 h-5" />
              Analysis Results
            </h2>

            <div className="space-y-6">
              {/* Risk Score */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <p className="text-sm text-gray-500">Risk Score</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className={`text-2xl font-bold ${result.probability > 0.7 ? 'text-red-600' :
                      result.probability > 0.4 ? 'text-orange-600' : 'text-green-600'
                      }`}>
                      {Math.round(result.probability * 100)}%
                    </span>
                    {result.probability > 0.7 ? (
                      <span className="px-2 py-1 bg-red-100 text-red-700 text-xs font-medium rounded-full">
                        THREAT DETECTED
                      </span>
                    ) : (
                      <span className="px-2 py-1 bg-green-100 text-green-700 text-xs font-medium rounded-full">
                        SAFE
                      </span>
                    )}
                  </div>
                </div>
                <div className={`p-3 rounded-full ${result.probability > 0.7 ? 'bg-red-100' : 'bg-green-100'
                  }`}>
                  {result.probability > 0.7 ? (
                    <AlertTriangle className={`w-6 h-6 ${result.probability > 0.7 ? 'text-red-600' : 'text-green-600'}`} />
                  ) : (
                    <CheckCircle className="w-6 h-6 text-green-600" />
                  )}
                </div>
              </div>

              {/* Deepfake Analysis Status */}
              {result.feature_scores?.deepfake_analysis && (
                <div className={`p-4 rounded-lg border ${result.feature_scores.deepfake_analysis.is_deepfake ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
                  <h3 className={`text-sm font-bold ${result.feature_scores.deepfake_analysis.is_deepfake ? 'text-red-800' : 'text-green-800'}`}>
                    Audio Deepfake DNA Analysis
                  </h3>
                  <p className="text-sm mt-1">
                    Verdict: <strong>{result.feature_scores.deepfake_analysis.is_deepfake ? 'SYNTHETIC / AI GENERATED' : 'HUMAN VOICE'}</strong>
                  </p>
                  {result.feature_scores.deepfake_analysis.is_deepfake && (
                    <div className="mt-2 text-xs text-red-600">
                      <strong>Detected Markers:</strong>
                      <ul className="list-disc pl-4 mt-1">
                        {result.feature_scores.deepfake_analysis.reasons.map((r, i) => <li key={i}>{r}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Transcript */}
              {(result.transcript || result.explain_html) && (
                <div>
                  <h3 className="text-sm font-medium text-gray-900 mb-2">Transcript & Analysis</h3>
                  <div className="p-4 bg-gray-50 rounded-lg text-sm text-gray-600 max-h-48 overflow-y-auto">
                    <p className="mb-2 font-semibold">Extracted Text:</p>
                    {result.transcript && <p className="mb-4">{result.transcript}</p>}

                    {result.explain_html && (
                      <div>
                        <p className="mb-1 font-semibold text-xs uppercase text-gray-500">Analysis Highlights:</p>
                        <div dangerouslySetInnerHTML={{ __html: result.explain_html }} />
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default AudioSTT
