import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import URLScanner from './pages/URLScanner'
import EmailInspector from './pages/EmailInspector'
import SMSAnalyzer from './pages/SMSAnalyzer'
import ImageOCR from './pages/ImageOCR'
import AudioSTT from './pages/AudioSTT'
import Reports from './pages/Reports'
import Simulation from './pages/Simulation'
import Settings from './pages/Settings'
import BrandProtection from './pages/BrandProtection'
import ThreatLedger from './pages/ThreatLedger'

function App() {
  return (
    <div className="min-h-screen bg-dark-bg">
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route
            path="dashboard"
            element={
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Dashboard />
              </motion.div>
            }
          />
          <Route
            path="url-scanner"
            element={
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <URLScanner />
              </motion.div>
            }
          />
          <Route
            path="email-inspector"
            element={
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <EmailInspector />
              </motion.div>
            }
          />
          <Route
            path="sms-analyzer"
            element={
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <SMSAnalyzer />
              </motion.div>
            }
          />
          <Route
            path="image-ocr"
            element={
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <ImageOCR />
              </motion.div>
            }
          />
          <Route
            path="audio-stt"
            element={
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <AudioSTT />
              </motion.div>
            }
          />
          <Route
            path="reports"
            element={
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Reports />
              </motion.div>
            }
          />
          <Route
            path="simulation"
            element={
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Simulation />
              </motion.div>
            }
          />
          <Route
            path="brand-protection"
            element={
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <BrandProtection />
              </motion.div>
            }
          />
          <Route
            path="threat-ledger"
            element={
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <ThreatLedger />
              </motion.div>
            }
          />
          <Route
            path="settings"
            element={
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Settings />
              </motion.div>
            }
          />
        </Route>
      </Routes>
    </div>
  )
}

export default App