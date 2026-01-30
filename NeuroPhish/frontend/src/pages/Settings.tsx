import React from 'react'

const Settings = () => {
    return (
        <div className="p-6">
            <h1 className="text-2xl font-bold text-white mb-4">Settings</h1>
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                    <h3 className="font-medium text-gray-900">NeuroPhish Version</h3>
                    <p className="text-sm text-gray-500">Current installed version</p>
                </div>
                <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">v2.4.0</span>
            </div>
        </div>
    )
}

export default Settings
