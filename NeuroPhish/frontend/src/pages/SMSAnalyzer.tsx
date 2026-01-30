import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import {
  MessageSquare,
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock,
  Send,
  Smartphone,
  Copy,
  Download,
  Zap,
  Phone
} from 'lucide-react'
import toast from 'react-hot-toast'

const smsSchema = z.object({
  text: z.string().min(1, 'Message text is required').max(1000),
  sender: z.string().optional()
})

type SMSFormData = z.infer<typeof smsSchema>

interface SMSAnalysisResult {
  id: string
  probability: number
  risk_level: string
  confidence: number
  reasons: Array<{
    feature: string
    contribution: number
    description: string
  }>
  explain_html?: string
  timestamp: string
  processing_time_ms: number
  feature_scores?: {
    heuristic: number
    text_analysis: number
  }
  metadata?: {
    text_length: number
    url_count: number
    urgency_score: number
    sender_reputation?: number
    sender_is_shortcode?: boolean
    has_shortened_url?: boolean
  }
}

const SMSAnalyzer: React.FC = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<SMSAnalysisResult | null>(null)

  const { register, handleSubmit, setValue, watch, formState: { errors } } = useForm<SMSFormData>({
    resolver: zodResolver(smsSchema)
  })

  const onSubmit = async (data: SMSFormData) => {
    setIsAnalyzing(true)
    setResult(null)

    try {
      const response = await fetch('http://localhost:8000/api/v1/predict/sms', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: data.text,
          sender: data.sender || undefined
        }),
      })

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const resultData = await response.json()
      setResult(resultData)
      toast.success('SMS analysis completed')
    } catch (error) {
      console.error('Error:', error)
      toast.error('Analysis failed. Please check the backend connection.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const loadSampleSMS = (type: 'smishing' | 'legitimate') => {
    if (type === 'smishing') {
      setValue('text', 'URGENT: Your bank account has been compromised! Click here immediately to secure it: bit.ly/secure-bank-123. Act now or lose access forever!')
      setValue('sender', '87654')
    } else {
      setValue('text', 'Hi! Your package from Amazon is arriving tomorrow between 2-4 PM. To track your delivery, visit our website.')
      setValue('sender', 'AMZ-DLV')
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
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text">SMS Analyzer</h1>
          <p className="text-gray-400 mt-1">Detect smishing and malicious text messages using AI</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-sm text-gray-400">Messages Analyzed Today</p>
            <p className="text-xl font-bold text-gray-900">412</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Section */}
        <motion.div
          className="card p-6"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900">Message Content</h3>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => loadSampleSMS('smishing')}
                className="btn-secondary px-3 py-1 text-sm"
              >
                Load Smishing Sample
              </button>
              <button
                type="button"
                onClick={() => loadSampleSMS('legitimate')}
                className="btn-secondary px-3 py-1 text-sm"
              >
                Load Legitimate Sample
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">
                Sender ID/Number (Optional)
              </label>
              <div className="relative">
                <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  {...register('sender')}
                  type="text"
                  placeholder="e.g. +1234567890 or BANK-MSG"
                  className="input pl-12 w-full"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">
                SMS Text
              </label>
              <textarea
                {...register('text')}
                placeholder="Paste the SMS text here..."
                className="textarea w-full"
                rows={6}
              />
              {errors.text && (
                <p className="text-red-400 text-sm mt-1">{errors.text.message}</p>
              )}
            </div>

            <button
              type="submit"
              className={`btn-primary w-full py-4 flex items-center justify-center gap-2 ${isAnalyzing ? 'opacity-50 cursor-not-allowed' : ''}`}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Zap className="w-5 h-5 animate-spin" />
                  Analyzing Message...
                </>
              ) : (
                <>
                  <Shield className="w-5 h-5" />
                  Scan Message
                </>
              )}
            </button>
          </form>
        </motion.div>

        {/* Live Preview / Real-time Feedback */}
        <div className="space-y-4">
          <div className="card p-6 bg-gradient-to-br from-dark-card to-neon-purple/5">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Smartphone className="w-5 h-5 text-neon-blue" />
              Message Preview
            </h3>
            <div className="max-w-[300px] mx-auto">
              <div className="bg-[#1C1C1E] rounded-[2rem] p-4 shadow-2xl border-4 border-[#3A3A3C]">
                <div className="flex flex-col h-[400px]">
                  <div className="flex items-center justify-between mb-4 border-b border-gray-800 pb-2">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center">
                        <User className="w-4 h-4 text-gray-400" />
                      </div>
                      <span className="text-xs font-medium text-white">{watch('sender') || 'Unknown'}</span>
                    </div>
                  </div>
                  <div className="flex-1 overflow-y-auto space-y-2">
                    <div className="bg-[#3A3A3C] rounded-2xl p-3 text-sm text-white max-w-[80%]">
                      {watch('text') || 'Waiting for message input...'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {result && (
        <motion.div
          className="space-y-6 pt-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Risk Assessment */}
            <div className="card p-6 md:col-span-2">
              <div className="flex items-center justify-between mb-8">
                <h3 className="text-xl font-semibold text-gray-900">Message Risk Intelligence</h3>
                <div className={`px-4 py-1 rounded-full border ${getRiskColor(result.risk_level)} font-bold text-sm`}>
                  {result.risk_level.toUpperCase()} THREAT
                </div>
              </div>

              <div className="flex flex-col md:flex-row items-center gap-12">
                <div className={`risk-meter ${result.risk_level} scale-125`}>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-gray-900">{Math.round(result.probability * 100)}%</div>
                    <div className="text-[10px] text-gray-400">RISK INDEX</div>
                  </div>
                </div>

                <div className="flex-1 grid grid-cols-2 gap-y-6 gap-x-8 w-full">
                  <div className="space-y-1">
                    <p className="text-xs text-gray-400 uppercase tracking-wider">Urgency Score</p>
                    <p className="text-xl font-semibold text-gray-900">{Math.round((result.metadata?.urgency_score || 0) * 100)}%</p>
                    <div className="w-full bg-dark-border h-1 rounded-full">
                      <div className="bg-neon-blue h-1 rounded-full" style={{ width: `${(result.metadata?.urgency_score || 0) * 100}%` }}></div>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-gray-400 uppercase tracking-wider">Confidence Level</p>
                    <p className="text-xl font-semibold text-gray-900">{Math.round(result.confidence * 100)}%</p>
                    <div className="w-full bg-dark-border h-1 rounded-full">
                      <div className="bg-neon-purple h-1 rounded-full" style={{ width: `${result.confidence * 100}%` }}></div>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-gray-400 uppercase tracking-wider">Extracted URLs</p>
                    <p className="text-xl font-semibold text-gray-900">{result.metadata?.url_count || 0}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs text-gray-400 uppercase tracking-wider">Sender Reputation</p>
                    <p className={`text-xl font-semibold ${(result.metadata?.sender_reputation || 0) < 0.5 ? 'text-red-400' : 'text-green-400'}`}>
                      {result.metadata?.sender_reputation ? Math.round(result.metadata.sender_reputation * 100) : 'N/A'}%
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Content Highlighting */}
            <div className="card p-6 bg-dark-card/50">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Deep Analysis</h3>
              <div className="space-y-4">
                <div className="p-4 bg-black/30 rounded-lg border border-dark-border">
                  <p className="text-xs text-gray-400 mb-3 uppercase tracking-tighter">Heuristic Markers</p>
                  <div
                    className="text-sm leading-relaxed text-gray-300 font-sans"
                    dangerouslySetInnerHTML={{ __html: result.explain_html || '' }}
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2 rounded bg-dark-border/30 text-center">
                    <p className="text-[10px] text-gray-500">Shortened URL</p>
                    <p className="text-xs font-semibold text-gray-300">{result.metadata?.has_shortened_url ? 'YES' : 'NO'}</p>
                  </div>
                  <div className="p-2 rounded bg-dark-border/30 text-center">
                    <p className="text-[10px] text-gray-500">Short Code</p>
                    <p className="text-xs font-semibold text-gray-300">{result.metadata?.sender_is_shortcode ? 'YES' : 'NO'}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Breakdown Reasons */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="card p-6 border-l-4 border-l-neon-purple">
              <h3 className="text-sm font-bold text-gray-400 mb-4 uppercase tracking-widest">Neural Breakdown</h3>
              <div className="space-y-4">
                {result.reasons.map((reason, idx) => (
                  <div key={idx} className="flex items-start gap-4">
                    <div className="w-6 h-6 rounded-full bg-dark-border flex items-center justify-center text-[10px] font-bold text-neon-purple mt-1 flex-shrink-0">
                      {idx + 1}
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-gray-900">{reason.feature}</h4>
                      <p className="text-xs text-gray-400 leading-snug mt-1">{reason.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card p-6 bg-gradient-to-br from-dark-card to-blue-500/5">
              <h3 className="text-sm font-bold text-gray-400 mb-4 uppercase tracking-widest">Recommended Actions</h3>
              <div className="space-y-3">
                <button className="btn-secondary w-full py-2 text-xs flex items-center justify-center gap-2">
                  <Copy className="w-3 h-3" /> Copy Warning Details
                </button>
                <button className="btn-secondary w-full py-2 text-xs flex items-center justify-center gap-2">
                  <Download className="w-3 h-3" /> Export Forensic Log
                </button>
                <button className="btn-primary w-full py-3 text-xs flex items-center justify-center gap-2">
                  <Shield className="w-4 h-4" /> Block Sender & Report Phish
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

const User = ({ className, ...props }: any) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    {...props}
  >
    <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </svg>
)

export default SMSAnalyzer