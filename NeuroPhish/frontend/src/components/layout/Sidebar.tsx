import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  LayoutDashboard,
  Link,
  Mail,
  MessageSquare,
  Image,
  Mic,
  FileText,
  Users,
  Settings,
  Shield,
  Zap,
  Activity,
  Database
} from 'lucide-react'

const navigationItems = [
  {
    name: 'Dashboard',
    href: '/dashboard',
    icon: LayoutDashboard,
    description: 'Overview & Analytics'
  },
  {
    name: 'URL Scanner',
    href: '/url-scanner',
    icon: Link,
    description: 'Analyze suspicious URLs'
  },
  {
    name: 'Email Inspector',
    href: '/email-inspector',
    icon: Mail,
    description: 'Email threat analysis'
  },
  {
    name: 'SMS Analyzer',
    href: '/sms-analyzer',
    icon: MessageSquare,
    description: 'SMS phishing detection'
  },
  {
    name: 'Image OCR',
    href: '/image-ocr',
    icon: Image,
    description: 'Image-based threats'
  },
  {
    name: 'Audio STT',
    href: '/audio-stt',
    icon: Mic,
    description: 'Voice phishing analysis'
  },
  {
    name: 'Reports',
    href: '/reports',
    icon: FileText,
    description: 'Forensic reports'
  },
  {
    name: 'Simulation',
    href: '/simulation',
    icon: Users,
    description: 'Phishing training'
  },
  {
    name: 'Brand Protection',
    href: '/brand-protection',
    icon: Shield,
    description: 'Protect brand identity'
  },
  {
    name: 'Blockchain Ledger',
    href: '/threat-ledger',
    icon: Database,
    description: 'Immutable threat record'
  },
  {
    name: 'Settings',
    href: '/settings',
    icon: Settings,
    description: 'System configuration'
  }
]

const Sidebar: React.FC = () => {
  const location = useLocation()

  return (
    <div className="h-full bg-dark-surface border-r border-dark-border flex flex-col">
      {/* Logo Section */}
      <div className="p-6 border-b border-dark-border">
        <motion.div
          className="flex items-center gap-3"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-gray-900 text-lg">NeuroPhish</h1>
            <p className="text-xs text-gray-500">AI Security Platform</p>
          </div>
        </motion.div>
      </div>

      {/* System Status */}
      <div className="px-6 py-4 border-b border-dark-border">
        <div className="bg-dark-card rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-white">System Status</span>
            <div className="flex items-center gap-1">
              <Activity className="w-3 h-3 text-green-500" />
              <span className="text-xs text-green-500">Online</span>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">ML Models</span>
              <span className="text-green-500">✓ Active</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Threat Intel</span>
              <span className="text-green-500">✓ Updated</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">API Health</span>
              <span className="text-green-500">✓ 99.9%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 overflow-y-auto">
        <div className="space-y-2">
          {navigationItems.map((item, index) => {
            const isActive = location.pathname === item.href
            const Icon = item.icon

            return (
              <motion.div
                key={item.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <NavLink
                  to={item.href}
                  className={({ isActive }) =>
                    `sidebar-item group relative ${isActive ? 'active' : ''}`
                  }
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{item.name}</p>
                    <p className="text-xs text-gray-400 truncate">{item.description}</p>
                  </div>

                  {/* Active indicator */}
                  {isActive && (
                    <motion.div
                      className="absolute right-2 w-2 h-2 bg-neon-purple rounded-full"
                      layoutId="activeIndicator"
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    />
                  )}

                  {/* Hover glow effect */}
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-neon-purple/10 to-neon-blue/10 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                    initial={false}
                  />
                </NavLink>
              </motion.div>
            )
          })}
        </div>
      </nav>

      {/* Quick Actions */}
      <div className="p-4 border-t border-dark-border">
        <div className="space-y-2">
          <motion.button
            className="w-full btn-primary py-2 text-sm"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Zap className="w-4 h-4 mr-2" />
            Quick Scan
          </motion.button>

          <motion.button
            className="w-full btn-secondary py-2 text-sm"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <FileText className="w-4 h-4 mr-2" />
            Generate Report
          </motion.button>
        </div>
      </div>
    </div>
  )
}

export default Sidebar