'use client'

import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar } from 'recharts'
import { RefreshCw, Zap } from 'lucide-react'

const API_BASE = 'http://localhost:8000'

export default function FraudDashboard() {
  const [metrics, setMetrics] = useState({ drift_scores: {} })
  const [registry, setRegistry] = useState({ versions: [] })
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(false)

  const fetchData = async () => {
    setLoading(true)
    try {
      const [metricsRes, registryRes, predictionsRes] = await Promise.all([
        fetch(`${API_BASE}/metrics`),
        fetch(`${API_BASE}/registry`),
        fetch(`${API_BASE}/predictions?limit=100`)
      ])
      setMetrics(await metricsRes.json())
      setRegistry(await registryRes.json())
      setPredictions(await predictionsRes.json())
    } catch (error) {
      console.error('Error fetching data:', error)
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [])

  const handleRetrain = async () => {
    await fetch(`${API_BASE}/retrain`, { method: 'POST' })
    fetchData()
  }

  const handlePromote = async () => {
    await fetch(`${API_BASE}/promote`, { method: 'POST' })
    fetchData()
  }

  const fraudRate = predictions.length > 0 ? (predictions.filter(p => p.prediction === 1).length / predictions.length * 100).toFixed(1) : 0

  const sparklineData = predictions.slice(-20).map((p, i) => ({ x: i, confidence: p.confidence }))

  const confidenceData = predictions.reduce((acc, p) => {
    if (p.shadow_confidence !== null) {
      acc.push({ production: p.confidence, shadow: p.shadow_confidence })
    }
    return acc
  }, [])

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold">FRAUD ML SYSTEM</h1>
          <div className="flex gap-4">
            <button onClick={handleRetrain} className="bg-blue-600 px-4 py-2 rounded flex items-center gap-2">
              <RefreshCw size={16} /> Retrain
            </button>
            <button onClick={handlePromote} className="bg-green-600 px-4 py-2 rounded flex items-center gap-2">
              <Zap size={16} /> Promote
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-gray-800 p-4 rounded">
            <h2 className="text-xl font-semibold mb-4">DRIFT SCORES</h2>
            <div className="space-y-2">
              <div>KL Amount: {metrics.drift_scores?.amount_kl?.toFixed(3) || '0.000'}</div>
              <div>PSI Amount: {metrics.drift_scores?.amount_psi?.toFixed(3) || '0.000'}</div>
              <div>KL Confidence: {metrics.drift_scores?.confidence_kl?.toFixed(3) || '0.000'}</div>
              <div>PSI Confidence: {metrics.drift_scores?.confidence_psi?.toFixed(3) || '0.000'}</div>
              <div>Status: STABLE</div>
            </div>
          </div>

          <div className="bg-gray-800 p-4 rounded">
            <h2 className="text-xl font-semibold mb-4">MODEL REGISTRY</h2>
            <div className="space-y-2">
              {registry.versions?.map(v => (
                <div key={v.version} className="flex justify-between">
                  <span>v{v.version}</span>
                  <span>{v.trigger_reason}</span>
                  <span className={v.status === 'production' ? 'text-green-400' : 'text-yellow-400'}>{v.status}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-gray-800 p-4 rounded">
            <h2 className="text-xl font-semibold mb-4">PREDICTION VOLUME (last 100)</h2>
            <LineChart width={400} height={200} data={sparklineData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="confidence" stroke="#8884d8" />
            </LineChart>
          </div>

          <div className="bg-gray-800 p-4 rounded flex items-center justify-center">
            <div className="text-center">
              <h2 className="text-xl font-semibold mb-4">FRAUD RATE %</h2>
              <div className="text-4xl font-bold">{fraudRate}%</div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded">
          <h2 className="text-xl font-semibold mb-4">SHADOW vs PRODUCTION CONFIDENCE DISTRIBUTION</h2>
          <BarChart width={800} height={300} data={confidenceData.slice(0, 20)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis />
            <YAxis />
            <Tooltip />
            <Bar dataKey="production" fill="#8884d8" />
            <Bar dataKey="shadow" fill="#82ca9d" />
          </BarChart>
        </div>
      </div>
    </div>
  )
}