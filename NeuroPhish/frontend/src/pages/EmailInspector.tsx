import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import {
  Mail,
  Shield,
  AlertTriangle,
  CheckCircle,
  Eye,
  EyeOff,
  Copy,
  Download,
  Clock,
  AtSign
} from 'lucide-react'
import toast from 'react-hot-toast'

const emailSchema = z.object({
  subject: z.string().min(1, 'Subject is required'),
  body: z.string().min(1, 'Email body is required'),
  sender: z.string().email('Invalid email format').optional().or(z.literal('')),
  headers: z.string().optional()
})

type EmailFormData = z.infer<typeof emailSchema>

interface EmailAnalysisResult {
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
    spf_pass: boolean
    dkim_valid: boolean
    dmarc_pass: boolean
    urgency_score: number
    link_count: number
    external_links: number
  }
}

const EmailInspector: React.FC = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<EmailAnalysisResult | null>(null)
  const [showRawHeaders, setShowRawHeaders] = useState(false)

  const { register, handleSubmit, setValue, formState: { errors } } = useForm<EmailFormData>({
    resolver: zodResolver(emailSchema)
  })

  const onSubmit = async (data: EmailFormData) => {
    setIsAnalyzing(true)
    setResult(null)

    try {
      const response = await fetch('http://localhost:8000/api/v1/predict/email', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          subject: data.subject,
          body: data.body,
          sender: data.sender || undefined,
          headers: data.headers ? JSON.parse(data.headers) : undefined
        }),
      })

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const resultData = await response.json()
      setResult(resultData)
      toast.success('Email analysis completed')
    } catch (error) {
      console.error('Error:', error)
      toast.error('Analysis failed. Please check the backend connection.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const loadSampleEmail = (type: 'phishing' | 'legitimate') => {
    if (type === 'phishing') {
      setValue('subject', 'URGENT: Your Account Will Be Suspended')
      setValue('body', `Dear Customer,

Your account has been flagged for suspicious activity and will be suspended within 24 hours unless you verify your identity immediately.

Click here to verify your account: http://suspicious-bank-verify.com/login

Failure to verify will result in permanent account closure.

Best regards,
Security Team`)
      setValue('sender', 'security@bank-alerts.com')
    } else {
      setValue('subject', 'Your Monthly Statement is Ready')
      setValue('body', `Dear Valued Customer,

Your monthly statement for October 2024 is now available in your online banking portal.

To view your statement, please log in to your account at our official website.

Thank you for banking with us.

Best regards,
Customer Service Team`)
      setValue('sender', 'statements@yourbank.com')
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
          <h1 className="text-3xl font-bold gradient-text">Email Inspector</h1>
          <p className="text-gray-400 mt-1">Analyze emails for phishing and social engineering attacks</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-sm text-gray-400">Emails Analyzed Today</p>
            <p className="text-xl font-bold text-gray-900">847</p>
          </div>
        </div>
      </div>

      {/* Email Input Form */}
      <motion.div
        className="card p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">Email Analysis</h3>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => loadSampleEmail('phishing')}
              className="btn-secondary px-3 py-1 text-sm"
            >
              Load Phishing Sample
            </button>
            <button
              type="button"
              onClick={() => loadSampleEmail('legitimate')}
              className="btn-secondary px-3 py-1 text-sm"
            >
              Load Legitimate Sample
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">
                Subject Line
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  {...register('subject')}
                  type="text"
                  placeholder="Enter email subject"
                  className="input pl-12 w-full"
                  disabled={isAnalyzing}
                />
              </div>
              {errors.subject && (
                <p className="text-red-400 text-sm mt-1">{errors.subject.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">
                Sender Email (Optional)
              </label>
              <div className="relative">
                <AtSign className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  {...register('sender')}
                  type="email"
                  placeholder="sender@example.com"
                  className="input pl-12 w-full"
                  disabled={isAnalyzing}
                />
              </div>
              {errors.sender && (
                <p className="text-red-400 text-sm mt-1">{errors.sender.message}</p>
              )}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-900 mb-2">
              Email Body
            </label>
            <textarea
              {...register('body')}
              placeholder="Paste the email content here..."
              className="textarea w-full"
              rows={8}
              disabled={isAnalyzing}
            />
            {errors.body && (
              <p className="text-red-400 text-sm mt-1">{errors.body.message}</p>
            )}
          </div>

          <div className="flex justify-end pt-4">
            <button
              type="submit"
              className={`btn-primary px-8 py-3 flex items-center gap-2 ${isAnalyzing ? 'opacity-50 cursor-not-allowed' : ''}`}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Clock className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Shield className="w-5 h-5" />
                  Analyze Email
                </>
              )}
            </button>
          </div>
        </form>
      </motion.div>

      {/* Results Section */}
      {result && (
        <motion.div
          className="space-y-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {/* Risk Assessment */}
          <div className="card p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-gray-900">Email Risk Assessment</h3>
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-400">{Math.round(result.processing_time_ms)}ms</span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Risk Score */}
              <div className="text-center">
                <div className={`risk-meter ${result.risk_level} mx-auto mb-4`}>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-gray-900">
                      {Math.round(result.probability * 100)}%
                    </div>
                    <div className="text-sm text-gray-300">Phishing Risk</div>
                  </div>
                </div>
                <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border ${getRiskColor(result.risk_level)}`}>
                  {result.risk_level === 'high' || result.risk_level === 'critical' ? (
                    <AlertTriangle className="w-4 h-4" />
                  ) : (
                    <CheckCircle className="w-4 h-4" />
                  )}
                  <span className="font-medium">{result.risk_level.toUpperCase()} RISK</span>
                </div>
              </div>

              {/* Authentication Status */}
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-gray-900">Authentication</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-300">SPF</span>
                    <span className={`text-sm ${result.metadata?.spf_pass ? 'text-green-400' : 'text-red-400'}`}>
                      {result.metadata?.spf_pass ? '✓ PASS' : '✗ FAIL'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-300">DKIM</span>
                    <span className={`text-sm ${result.metadata?.dkim_valid ? 'text-green-400' : 'text-red-400'}`}>
                      {result.metadata?.dkim_valid ? '✓ VALID' : '✗ INVALID'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-300">DMARC</span>
                    <span className={`text-sm ${result.metadata?.dmarc_pass ? 'text-green-400' : 'text-red-400'}`}>
                      {result.metadata?.dmarc_pass ? '✓ PASS' : '✗ FAIL'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Content Analysis */}
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-gray-900">Content Analysis</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-300">Urgency Score</span>
                    <span className="text-sm text-gray-900">
                      {Math.round((result.metadata?.urgency_score || 0) * 100)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-300">Total Links</span>
                    <span className="text-sm text-gray-900">{result.metadata?.link_count || 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-300">External Links</span>
                    <span className={`text-sm ${(result.metadata?.external_links || 0) > 0 ? 'text-orange-400' : 'text-green-400'}`}>
                      {result.metadata?.external_links || 0}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Highlighted Content */}
          <div className="card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Suspicious Content Analysis</h3>
            <div className="p-4 bg-dark-card rounded-lg border border-dark-border">
              <div className="text-sm text-gray-400 mb-2">Highlighted suspicious elements:</div>
              <div
                className="text-gray-900 leading-relaxed font-sans text-sm"
                dangerouslySetInnerHTML={{ __html: result.explain_html || 'No suspicious elements detected.' }}
              />
            </div>
          </div>

          {/* Detection Reasons */}
          <div className="card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Detection Analysis</h3>
            <div className="space-y-4">
              {result.reasons.map((reason, index) => (
                <div
                  key={index}
                  className="flex items-start gap-4 p-4 bg-dark-card rounded-lg border border-dark-border"
                >
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-gradient-to-r from-neon-purple to-neon-blue rounded-full flex items-center justify-center text-gray-900 text-sm font-bold">
                      {index + 1}
                    </div>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-gray-900">{reason.feature}</h4>
                      <span className="text-sm text-gray-400">
                        {Math.round(reason.contribution * 100)}% impact
                      </span>
                    </div>
                    <p className="text-gray-300 text-sm">{reason.description}</p>
                    <div className="w-full bg-dark-border rounded-full h-1 mt-2">
                      <div
                        className={`h-1 rounded-full ${result.risk_level === 'high' || result.risk_level === 'critical'
                          ? 'bg-gradient-to-r from-red-500 to-orange-500'
                          : 'bg-gradient-to-r from-green-500 to-blue-500'
                          }`}
                        style={{ width: `${reason.contribution * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

export default EmailInspector