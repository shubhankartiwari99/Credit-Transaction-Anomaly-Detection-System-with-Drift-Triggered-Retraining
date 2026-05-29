'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Area,
  AreaChart
} from 'recharts'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const CHART_COLORS = {
  production: '#4cd7f6', // primary
  shadow: '#bec6e0', // secondary
  danger: '#F43F5E', // danger-rose
  calm: '#06B6D4', // info-cyan
  grid: 'rgba(134, 147, 151, 0.1)', // outline with low opacity
}

function formatTimestamp(value) {
  if (!value) {
    return 'n/a'
  }

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return 'n/a'
  }

  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function normalizePredictions(payload) {
  if (!Array.isArray(payload)) {
    return []
  }

  return payload
    .map((item, index) => ({
      id: `${item.timestamp || 'prediction'}-${index}`,
      label: formatTimestamp(item.timestamp),
      timestamp: item.timestamp || null,
      prediction: Number(item.prediction ?? 0),
      confidence: Number(item.confidence ?? 0),
      shadowPrediction: item.shadow_prediction == null ? null : Number(item.shadow_prediction),
      shadowConfidence: item.shadow_confidence == null ? null : Number(item.shadow_confidence),
    }))
    .filter((item) => Number.isFinite(item.confidence))
}

function normalizeRegistry(payload) {
  const versions = Array.isArray(payload?.versions) ? payload.versions : []
  return versions.map((item) => ({
    version: item.version,
    triggerReason: item.trigger_reason || 'unknown',
    status: item.status || 'unknown',
    trainedAt: item.trained_at || null,
    aucPr: item.auc_pr,
    aucRoc: item.auc_roc,
    precision: item.precision,
    recall: item.recall,
    f1: item.f1,
  }))
}

function normalizeMetrics(payload) {
  return {
    amountKl: Number(payload?.drift_scores?.amount_kl ?? 0),
    amountPsi: Number(payload?.drift_scores?.amount_psi ?? 0),
    confidenceKl: Number(payload?.drift_scores?.confidence_kl ?? 0),
    confidencePsi: Number(payload?.drift_scores?.confidence_psi ?? 0),
  }
}

function getDriftStatus(metrics) {
  if (metrics.amountPsi > 0.2 || metrics.confidenceKl > 0.1) {
    return { label: 'Drift Watch', tone: 'text-danger-rose bg-danger-rose/10 border-danger-rose/30' }
  }
  if (metrics.amountPsi > 0.1 || metrics.confidenceKl > 0.05) {
    return { label: 'Monitor', tone: 'text-info-cyan bg-info-cyan/10 border-info-cyan/30' }
  }
  return { label: 'Stable', tone: 'text-success-emerald bg-success-emerald/10 border-success-emerald/30' }
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: 'no-store' })
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`)
  }
  return response.json()
}

export default function FraudDashboard() {
  const [metrics, setMetrics] = useState({
    amountKl: 0,
    amountPsi: 0,
    confidenceKl: 0,
    confidencePsi: 0,
  })
  const [registry, setRegistry] = useState([])
  const [predictions, setPredictions] = useState([])
  const [driftData, setDriftData] = useState({ drift_score: 0, status: 'LOW', threshold: 0.1, last_updated: null })
  const [driftHistory, setDriftHistory] = useState([])
  const [retrainStatus, setRetrainStatus] = useState({ status: 'idle', reason: null, new_model_version: null, timestamp: null, top_shifted_feature: null })
  const [loading, setLoading] = useState(true)
  const [retraining, setRetraining] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [errors, setErrors] = useState([])

  const fetchData = async () => {
    setLoading(true)
    const results = await Promise.allSettled([
      fetchJson(`${API_BASE}/metrics`),
      fetchJson(`${API_BASE}/registry`),
      fetchJson(`${API_BASE}/predictions?limit=100`),
      fetchJson(`${API_BASE}/drift`),
      fetchJson(`${API_BASE}/retrain/status`),
      fetchJson(`${API_BASE}/drift/history`),
    ])

    const [metricsResult, registryResult, predictionsResult, driftResult, retrainResult, historyResult] = results
    const nextErrors = []

    if (metricsResult.status === 'fulfilled') {
      setMetrics(normalizeMetrics(metricsResult.value))
    } else {
      nextErrors.push(`Metrics: ${metricsResult.reason.message}`)
    }

    if (registryResult.status === 'fulfilled') {
      setRegistry(normalizeRegistry(registryResult.value))
    } else {
      nextErrors.push(`Registry: ${registryResult.reason.message}`)
    }

    if (predictionsResult.status === 'fulfilled') {
      setPredictions(normalizePredictions(predictionsResult.value))
    } else {
      nextErrors.push(`Predictions: ${predictionsResult.reason.message}`)
    }

    if (driftResult.status === 'fulfilled') {
      setDriftData(driftResult.value)
    } else {
      nextErrors.push(`Drift: ${driftResult.reason.message}`)
    }

    if (retrainResult.status === 'fulfilled') {
      setRetrainStatus(retrainResult.value)
    } else {
      nextErrors.push(`Retrain Status: ${retrainResult.reason.message}`)
    }

    if (historyResult.status === 'fulfilled') {
      const historyFormatted = Array.isArray(historyResult.value) 
        ? historyResult.value.map((h, i) => ({
            label: formatTimestamp(h.timestamp),
            score: Number(h.drift_score?.toFixed(3))
          }))
        : []
      setDriftHistory(historyFormatted)
    } else {
      nextErrors.push(`Drift History: ${historyResult.reason.message}`)
    }

    if (nextErrors.length > 0) {
      console.error('Dashboard fetch errors', nextErrors)
    }

    setErrors(nextErrors)
    setLastUpdated(new Date())
    setLoading(false)
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [])

  const handleRetrain = async () => {
    setRetraining(true)
    try {
      await fetch(`${API_BASE}/retrain`, { method: 'POST' })
      await fetchData()
    } finally {
      setRetraining(false)
    }
  }

  const handlePromote = async () => {
    await fetch(`${API_BASE}/promote`, { method: 'POST' })
    fetchData()
  }

  const fraudRate = useMemo(() => {
    if (predictions.length === 0) {
      return 0
    }
    return Number(((predictions.filter((item) => item.prediction === 1).length / predictions.length) * 100).toFixed(1))
  }, [predictions])

  const predictionTrendData = useMemo(
    () =>
      predictions.slice(-24).map((item, index) => ({
        label: item.label || `${index + 1}`,
        confidence: Number((item.confidence * 100).toFixed(1)),
        shadowConfidence:
          item.shadowConfidence == null ? null : Number((item.shadowConfidence * 100).toFixed(1)),
      })),
    [predictions]
  )

  const predictionMix = useMemo(() => {
    const fraudCount = predictions.filter((item) => item.prediction === 1).length
    const normalCount = Math.max(predictions.length - fraudCount, 0)
    return [
      { name: 'Normal', value: normalCount, fill: CHART_COLORS.calm },
      { name: 'Fraud', value: fraudCount, fill: CHART_COLORS.danger },
    ]
  }, [predictions])

  const registrySummary = useMemo(
    () =>
      registry.map((item) => ({
        ...item,
        trainedLabel: item.trainedAt ? new Date(item.trainedAt).toLocaleString() : 'unknown',
      })),
    [registry]
  )

  const confidenceDistribution = useMemo(() => {
    const bins = Array(10).fill(0)
    predictions.forEach(p => {
      if (p.confidence != null) {
        const binIndex = Math.min(Math.floor(p.confidence * 10), 9)
        bins[binIndex]++
      }
    })
    return bins.map((count, i) => ({
      range: `${i * 10}-${(i + 1) * 10}%`,
      count
    }))
  }, [predictions])

  const driftStatus = getDriftStatus(metrics)
  const latestPrediction = predictions[predictions.length - 1] || null
  const registryCount = registrySummary.length

  const MaterialIcon = ({ icon, className = '' }) => (
    <span className={`material-symbols-outlined ${className}`} data-icon={icon}>{icon}</span>
  )

  return (
    <>
      <nav className="bg-surface-dim hidden lg:flex flex-col h-screen py-margin left-0 w-72 border-r border-outline/10 fixed z-40">
        <div className="px-6 mb-8 flex items-center gap-4">
          <div className="w-10 h-10 rounded-lg bg-surface-container flex items-center justify-center border border-primary/20">
            <MaterialIcon icon="shield" className="text-primary" />
          </div>
          <div>
            <h2 className="font-headline-sm text-headline-sm font-bold text-primary">Fraud Ops</h2>
            <p className="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider">Precision Monitoring</p>
          </div>
        </div>
        <div className="flex-1 px-4 space-y-1 overflow-y-auto">
          <a className="flex items-center gap-3 px-4 py-3 rounded-lg bg-primary/10 text-primary border-l-4 border-primary font-label-md text-label-md group hover:bg-surface-container-high transition-all duration-300 translate-x-1" href="#">
            <MaterialIcon icon="monitoring" /> Live Surveillance
          </a>
          <a className="flex items-center gap-3 px-4 py-3 rounded-lg text-on-surface-variant hover:bg-surface-variant/30 font-label-md text-label-md group hover:bg-surface-container-high transition-all duration-300" href="#">
            <MaterialIcon icon="database" /> Model Registry
          </a>
          <a className="flex items-center gap-3 px-4 py-3 rounded-lg text-on-surface-variant hover:bg-surface-variant/30 font-label-md text-label-md group hover:bg-surface-container-high transition-all duration-300" href="#">
            <MaterialIcon icon="analytics" /> System Health
          </a>
          <a className="flex items-center gap-3 px-4 py-3 rounded-lg text-on-surface-variant hover:bg-surface-variant/30 font-label-md text-label-md group hover:bg-surface-container-high transition-all duration-300" href="#">
            <MaterialIcon icon="history" /> History
          </a>
        </div>
        <div className="px-4 mt-auto space-y-1 pt-4 border-t border-outline/10">
          <a className="flex items-center gap-3 px-4 py-3 rounded-lg text-on-surface-variant hover:bg-surface-variant/30 font-label-md text-label-md group hover:bg-surface-container-high transition-all duration-300" href="#">
            <MaterialIcon icon="settings" /> Settings
          </a>
        </div>
      </nav>

      <div className="flex-1 lg:ml-72 flex flex-col min-h-screen">
        <header className="bg-surface-container/80 backdrop-blur-md flex justify-between items-center px-gutter w-full h-16 sticky top-0 z-50 border-b border-outline/10">
          <button className="lg:hidden text-on-surface-variant p-2 -ml-2 rounded-lg hover:bg-surface-variant/50">
            <MaterialIcon icon="menu" />
          </button>
          <div className="flex items-center gap-4">
            <h1 className="font-headline-md text-headline-md font-bold tracking-tighter text-primary">Neural Sentry</h1>
            <span className="hidden md:inline-block px-3 py-1 bg-surface-container-high rounded-full font-label-sm text-label-sm text-on-surface-variant border border-outline/10">
                Live Surveillance
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button className="p-2 text-on-surface-variant hover:text-primary transition-colors duration-200 rounded-full hover:bg-surface-variant/50 relative">
              <MaterialIcon icon="notifications" />
              {errors.length > 0 && <span className="absolute top-2 right-2 w-2 h-2 bg-danger-rose rounded-full"></span>}
            </button>
            <div className="ml-4 w-9 h-9 rounded-full bg-surface-container border border-primary/30 overflow-hidden relative group cursor-pointer">
              <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuCkh8HQGA2CMJfBGV_5TcCjWM46Z98xcqW_rZ34DX44ZsBYjNf3B4GZ2V8jmXQ-yd7srUXMvFxwTbIkUFx20d0vBZs2RQRrJ4STR19KoehEbhapQ9VRQ6epJvIHXznRVMHtKSUzi8kpKr3wiu6Z7-X5hSZZlesN6K4NBbin6m1qlCXafJi1ECZ5XGi0_SzbDfZrtY0wZvjyML-1ZRbvzrvUxZRC1oBD1yVH8VnnEZHhMY179EUYHdaSHHPNqUHIuoEoxx3ydiooAU89" className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300" />
            </div>
          </div>
        </header>

        <main className="flex-1 p-gutter max-w-container-max-width mx-auto w-full">
          <div className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4 animate-fade-in-up">
            <div>
              <h1 className="font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface mb-2">Fraud ML System</h1>
              <p className="font-body-lg text-body-lg text-on-surface-variant max-w-2xl">Live drift surveillance, model registry status, and seeded prediction telemetry from the production backend.</p>
            </div>
            <div className="flex gap-3">
              <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border font-label-md text-label-md ${driftStatus.tone}`}>
                <span className={`w-2 h-2 rounded-full animate-pulse ${driftStatus.tone.includes('emerald') ? 'bg-success-emerald' : driftStatus.tone.includes('rose') ? 'bg-danger-rose' : 'bg-info-cyan'}`}></span>
                {driftStatus.label}
              </span>
              <button onClick={handleRetrain} disabled={retraining} className="px-4 py-2 rounded-lg bg-surface-container border border-outline/20 font-label-md text-label-md text-on-surface hover:bg-surface-container-high transition-colors disabled:opacity-60 disabled:cursor-not-allowed">
                {retraining ? 'Retraining...' : 'Retrain'}
              </button>
              <button onClick={handlePromote} className="px-4 py-2 rounded-lg bg-primary/10 border border-primary/30 font-label-md text-label-md text-primary hover:bg-primary/20 transition-colors">Promote</button>
            </div>
          </div>

          {errors.length > 0 && (
            <div className="mb-6 flex items-start gap-3 rounded-lg border border-danger-rose/30 bg-danger-rose/10 p-4 text-sm text-danger-rose">
              <MaterialIcon icon="warning" className="mt-0.5 shrink-0" />
              <div>
                <div className="font-medium">Partial dashboard data unavailable</div>
                <div className="mt-1 opacity-80">{errors.join(' • ')}</div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-6">
            <div className="glass-panel p-6 flex flex-col animate-fade-in-up-1">
              <div className="flex justify-between items-start mb-4">
                <span className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wider">Observed Fraud Rate</span>
                <MaterialIcon icon="policy" className="text-outline" />
              </div>
              <div className="mt-auto">
                <span className={`font-headline-lg text-headline-lg ${fraudRate > 5 ? 'text-danger-rose' : 'text-primary'}`}>{fraudRate}%</span>
                <p className="font-body-md text-body-md text-on-surface-variant mt-1">{predictions.length} recent predictions sampled</p>
              </div>
            </div>

            <div className="glass-panel p-6 flex flex-col animate-fade-in-up-2">
              <div className="flex justify-between items-start mb-4">
                <span className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wider">Registry Versions</span>
                <MaterialIcon icon="layers" className="text-outline" />
              </div>
              <div className="mt-auto">
                <span className="font-headline-lg text-headline-lg text-on-surface">{registryCount}</span>
                <p className="font-body-md text-body-md text-on-surface-variant mt-1">{registryCount > 0 ? `Latest: v${registrySummary[registryCount - 1].version}` : 'No versions returned'}</p>
              </div>
            </div>

            <div className="glass-panel p-6 flex flex-col animate-fade-in-up-3">
              <div className="flex justify-between items-start mb-4">
                <span className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wider">Latest Confidence</span>
                <MaterialIcon icon="target" className="text-outline" />
              </div>
              <div className="mt-auto">
                <span className="font-headline-lg text-headline-lg text-on-surface-variant">{latestPrediction ? `${(latestPrediction.confidence * 100).toFixed(1)}%` : 'n/a'}</span>
                <p className="font-body-md text-body-md text-on-surface-variant mt-1">{latestPrediction ? `Recorded at ${latestPrediction.label}` : 'Waiting for prediction data'}</p>
              </div>
            </div>

            <div className="glass-panel p-6 flex flex-col animate-fade-in-up-4">
              <div className="flex justify-between items-start mb-4">
                <span className="font-label-md text-label-md text-on-surface-variant uppercase tracking-wider">Last Sync</span>
                <MaterialIcon icon="sync" className={`text-outline ${loading ? 'animate-spin' : ''}`} />
              </div>
              <div className="mt-auto">
                <span className="font-headline-lg text-headline-lg text-primary">{lastUpdated ? formatTimestamp(lastUpdated.toISOString()) : 'pending'}</span>
                <p className="font-body-md text-body-md text-on-surface-variant mt-1">{loading ? 'Refreshing now...' : 'Auto-refresh every 30s'}</p>
              </div>
            </div>
          </div>

          <div className="mb-6 grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
            <div className="glass-panel p-6 flex flex-col animate-fade-in-up-5 chart-grid">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <h2 className="font-headline-sm text-headline-sm text-on-surface">Prediction Confidence Trend</h2>
                  <p className="font-body-md text-body-md text-on-surface-variant mt-1">Latest production and shadow confidence values</p>
                </div>
                <MaterialIcon icon="monitoring" className="text-primary/70" />
              </div>
              <div className="h-72">
                {predictionTrendData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={predictionTrendData}>
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} vertical={false} />
                      <XAxis dataKey="label" tick={{ fill: '#bcc9cd', fontSize: 12, fontFamily: 'JetBrains Mono' }} />
                      <YAxis tick={{ fill: '#bcc9cd', fontSize: 12, fontFamily: 'JetBrains Mono' }} domain={[0, 100]} />
                      <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid rgba(6,182,212,0.2)', borderRadius: '8px', color: '#dce1fb' }} itemStyle={{ fontFamily: 'JetBrains Mono', fontSize: '12px' }} />
                      <Line type="monotone" dataKey="confidence" stroke={CHART_COLORS.production} strokeWidth={2} dot={false} name="Production %" />
                      <Line type="monotone" dataKey="shadowConfidence" stroke={CHART_COLORS.shadow} strokeWidth={2} strokeDasharray="4 4" dot={false} name="Shadow %" />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center border border-dashed border-outline/20 rounded-lg text-on-surface-variant">No data</div>
                )}
              </div>
            </div>
            
            <div className="glass-panel p-6 flex flex-col animate-fade-in-up-5">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <h2 className="font-headline-sm text-headline-sm text-on-surface">Prediction Mix</h2>
                  <p className="font-body-md text-body-md text-on-surface-variant mt-1">Fraud vs normal decisions</p>
                </div>
                <div className="rounded-full border border-danger-rose/30 bg-danger-rose/10 px-3 py-1 font-label-md text-label-md text-danger-rose">
                  {fraudRate}% fraud
                </div>
              </div>
              <div className="h-56">
                {predictions.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={predictionMix} dataKey="value" nameKey="name" innerRadius={60} outerRadius={90} paddingAngle={2} stroke="none">
                        {predictionMix.map((entry) => <Cell key={entry.name} fill={entry.fill} />)}
                      </Pie>
                      <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid rgba(6,182,212,0.2)', borderRadius: '8px', color: '#dce1fb' }} itemStyle={{ fontFamily: 'JetBrains Mono', fontSize: '12px' }} />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center border border-dashed border-outline/20 rounded-lg text-on-surface-variant">No data</div>
                )}
              </div>
              <div className="mt-4 grid grid-cols-2 gap-4">
                {predictionMix.map((item) => (
                  <div key={item.name} className="bg-surface-container rounded-lg p-3 border border-outline/10 text-center">
                    <div className="font-label-md text-label-md text-on-surface-variant">{item.name}</div>
                    <div className="mt-1 font-headline-md text-headline-md text-on-surface">{item.value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-[0.85fr_1.15fr] mb-6">
            <div className="glass-panel p-6 flex flex-col animate-fade-in-up-6">
              <h2 className="font-headline-sm text-headline-sm text-on-surface mb-6">System Health</h2>
              
              <div className="bg-surface-container border border-outline/10 rounded-lg p-4 mb-4">
                <div className="font-label-md text-label-md text-on-surface-variant mb-3">Real-Time Drift</div>
                <div className="flex items-end justify-between">
                  <div>
                    <div className="font-headline-lg text-headline-lg text-on-surface font-label-md">{driftData?.drift_score?.toFixed(3)}</div>
                    <div className="font-label-sm text-label-sm text-on-surface-variant mt-1">Threshold: {driftData?.threshold}</div>
                  </div>
                  <div className={`px-3 py-1 rounded-full border font-label-md text-label-md uppercase ${driftData?.status === 'HIGH' ? 'bg-danger-rose/10 border-danger-rose/30 text-danger-rose' : 'bg-success-emerald/10 border-success-emerald/30 text-success-emerald'}`}>
                    {driftData?.status === 'HIGH' ? 'High Drift' : 'Low Drift'}
                  </div>
                </div>
              </div>

              <div className="bg-surface-container border border-outline/10 rounded-lg p-4">
                <div className="font-label-md text-label-md text-on-surface-variant mb-4">Retrain Pipeline</div>
                <div className="space-y-3 font-body-md text-body-md">
                  <div className="flex justify-between">
                    <span className="text-on-surface-variant">Last Status</span>
                    <span className={`font-semibold ${retrainStatus?.status === 'failed' ? 'text-danger-rose' : 'text-success-emerald uppercase'}`}>{retrainStatus?.status || 'IDLE'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-on-surface-variant">Trigger</span>
                    <span className="text-on-surface">{retrainStatus?.reason || 'None'}</span>
                  </div>
                  {retrainStatus?.top_shifted_feature && (
                    <div className="flex justify-between">
                      <span className="text-on-surface-variant">Top Shifted Feature</span>
                      <span className="text-info-cyan font-label-md">{retrainStatus.top_shifted_feature}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-on-surface-variant">New Model</span>
                    <span className="text-on-surface">{retrainStatus?.new_model_version || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-on-surface-variant">Last Run</span>
                    <span className="text-on-surface font-label-md">{retrainStatus?.timestamp ? formatTimestamp(retrainStatus.timestamp) : 'N/A'}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="glass-panel p-6 flex flex-col animate-fade-in-up-6">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h2 className="font-headline-sm text-headline-sm text-on-surface">Model Registry</h2>
                  <p className="font-body-md text-body-md text-on-surface-variant mt-1">Version control and promotion workflow</p>
                </div>
              </div>
              
              <div className="mb-6">
                <div className="font-label-md text-label-md text-on-surface-variant mb-3">Active Production Model</div>
                {registrySummary.filter(r => r.status === 'production').map(item => (
                  <div key={item.version} className="bg-success-emerald/10 border border-success-emerald/30 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <div className="font-headline-sm text-headline-sm text-success-emerald">v{item.version}</div>
                      <div className="font-label-sm text-label-sm text-success-emerald/80">Trigger: {item.triggerReason}</div>
                    </div>
                    <div className="flex gap-6 font-label-md text-label-md text-success-emerald/90">
                      <div>AUC-ROC: <span className="font-bold">{item.aucRoc ?? 'N/A'}</span></div>
                      <div>F1 Score: <span className="font-bold">{item.f1 ?? 'N/A'}</span></div>
                    </div>
                  </div>
                ))}
                {registrySummary.filter(r => r.status === 'production').length === 0 && (
                  <div className="text-on-surface-variant italic font-body-md text-body-md">No active production model found</div>
                )}
              </div>

              <div>
                <div className="font-label-md text-label-md text-on-surface-variant mb-3">Available Candidate Models</div>
                <div className="space-y-3 max-h-64 overflow-y-auto pr-2">
                  {registrySummary.filter(r => r.status !== 'production').length > 0 ? (
                    registrySummary.filter(r => r.status !== 'production').map((item) => (
                      <div key={item.version} className="bg-surface-container border border-outline/10 rounded-lg p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-headline-sm text-headline-sm text-on-surface">v{item.version}</span>
                            <span className="px-2 py-0.5 rounded-full bg-surface-variant font-label-sm text-label-sm uppercase text-on-surface-variant border border-outline/20">{item.status}</span>
                          </div>
                          <div className="font-label-sm text-label-sm text-on-surface-variant mb-2">Trigger: {item.triggerReason} • {item.trainedLabel}</div>
                          <div className="flex gap-4 font-label-md text-label-md text-primary">
                            <div>AUC-ROC: <span className="font-bold">{item.aucRoc ?? 'N/A'}</span></div>
                            <div>F1 Score: <span className="font-bold">{item.f1 ?? 'N/A'}</span></div>
                          </div>
                        </div>
                        {item.status === 'shadow' && (
                          <button onClick={handlePromote} className="px-4 py-2 rounded-lg bg-success-emerald/10 border border-success-emerald/30 font-label-md text-label-md text-success-emerald hover:bg-success-emerald/20 transition-colors shrink-0">
                            Promote v{item.version}
                          </button>
                        )}
                      </div>
                    ))
                  ) : (
                    <div className="border border-dashed border-outline/20 rounded-lg p-6 text-center text-on-surface-variant font-body-md text-body-md">
                      No candidate models available.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-2 mb-6">
            <div className="glass-panel p-6 flex flex-col animate-fade-in-up-7 chart-grid">
              <div className="mb-5">
                <h2 className="font-headline-sm text-headline-sm text-on-surface">Drift Score Over Time</h2>
                <p className="font-body-md text-body-md text-on-surface-variant mt-1">Evolution of data distribution divergence</p>
              </div>
              <div className="h-72">
                {driftHistory.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={driftHistory}>
                      <defs>
                        <linearGradient id="colorDrift" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={CHART_COLORS.danger} stopOpacity={0.3}/>
                          <stop offset="95%" stopColor={CHART_COLORS.danger} stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} vertical={false} />
                      <XAxis dataKey="label" tick={{ fill: '#bcc9cd', fontSize: 12, fontFamily: 'JetBrains Mono' }} interval="preserveStartEnd" />
                      <YAxis tick={{ fill: '#bcc9cd', fontSize: 12, fontFamily: 'JetBrains Mono' }} />
                      <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid rgba(244,63,94,0.2)', borderRadius: '8px', color: '#dce1fb' }} itemStyle={{ fontFamily: 'JetBrains Mono', fontSize: '12px' }} />
                      <Area type="monotone" dataKey="score" stroke={CHART_COLORS.danger} strokeWidth={2} fillOpacity={1} fill="url(#colorDrift)" name="Drift Score" />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center border border-dashed border-outline/20 rounded-lg text-on-surface-variant">No data</div>
                )}
              </div>
            </div>

            <div className="glass-panel p-6 flex flex-col animate-fade-in-up-7 chart-grid">
              <div className="mb-5">
                <h2 className="font-headline-sm text-headline-sm text-on-surface">Confidence Distribution</h2>
                <p className="font-body-md text-body-md text-on-surface-variant mt-1">Density of production model probability scores</p>
              </div>
              <div className="h-72">
                {predictions.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={confidenceDistribution}>
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} vertical={false} />
                      <XAxis dataKey="range" tick={{ fill: '#bcc9cd', fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                      <YAxis tick={{ fill: '#bcc9cd', fontSize: 12, fontFamily: 'JetBrains Mono' }} />
                      <Tooltip contentStyle={{ background: '#0F172A', border: '1px solid rgba(6,182,212,0.2)', borderRadius: '8px', color: '#dce1fb' }} itemStyle={{ fontFamily: 'JetBrains Mono', fontSize: '12px' }} cursor={{fill: 'rgba(255,255,255,0.02)'}} />
                      <Bar dataKey="count" name="Predictions" fill={CHART_COLORS.calm} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center border border-dashed border-outline/20 rounded-lg text-on-surface-variant">No data</div>
                )}
              </div>
            </div>
          </div>
          
          <footer className="mt-8 border-t border-outline/10 pt-6 pb-4 text-center">
             <p className="font-label-md text-label-md text-on-surface-variant">
              Fraud ML System · Built by <a href="#" className="text-primary hover:underline">Shubhankar Tiwari</a>
             </p>
          </footer>
        </main>
      </div>
      
      <nav className="fixed bottom-0 left-0 w-full z-50 flex lg:hidden justify-around items-center py-2 pb-safe bg-surface-container-highest/90 backdrop-blur-xl rounded-t-xl border-t border-outline/20 shadow-lg">
        <a className="flex flex-col items-center justify-center text-primary font-bold active:bg-surface-variant scale-110 transition-transform duration-150 p-2 rounded-lg" href="#">
          <MaterialIcon icon="radar" />
          <span className="font-label-sm mt-1">Live</span>
        </a>
        <a className="flex flex-col items-center justify-center text-on-surface-variant active:bg-surface-variant p-2 rounded-lg" href="#">
          <MaterialIcon icon="inventory_2" />
          <span className="font-label-sm mt-1">Models</span>
        </a>
        <a className="flex flex-col items-center justify-center text-on-surface-variant active:bg-surface-variant p-2 rounded-lg" href="#">
          <MaterialIcon icon="vital_signs" />
          <span className="font-label-sm mt-1">Health</span>
        </a>
        <a className="flex flex-col items-center justify-center text-on-surface-variant active:bg-surface-variant p-2 rounded-lg" href="#">
          <MaterialIcon icon="settings" />
          <span className="font-label-sm mt-1">Settings</span>
        </a>
      </nav>
    </>
  )
}
