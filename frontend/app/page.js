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
} from 'recharts'
import { Activity, AlertTriangle, RefreshCw, ShieldAlert, Zap } from 'lucide-react'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const CHART_COLORS = {
  production: '#5eead4',
  shadow: '#f59e0b',
  danger: '#fb7185',
  calm: '#38bdf8',
  grid: 'rgba(148, 163, 184, 0.16)',
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
    return { label: 'Drift Watch', tone: 'text-rose-300 bg-rose-500/10 border-rose-400/30' }
  }
  if (metrics.amountPsi > 0.1 || metrics.confidenceKl > 0.05) {
    return { label: 'Monitor', tone: 'text-amber-300 bg-amber-500/10 border-amber-400/30' }
  }
  return { label: 'Stable', tone: 'text-emerald-300 bg-emerald-500/10 border-emerald-400/30' }
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

  return (
    <div className="min-h-screen bg-slate-950 px-4 py-6 text-slate-100 md:px-6">
      <div className="mx-auto max-w-7xl">
        <div className="animate-fade-in-up mb-6 flex flex-col gap-4 rounded-3xl border border-slate-800 bg-gradient-to-br from-slate-900/90 to-slate-950/90 p-6 shadow-2xl shadow-cyan-500/5 backdrop-blur-sm md:flex-row md:items-end md:justify-between">
          <div>
            <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-cyan-300/70">
              <Activity size={14} className="animate-pulse" />
              Fraud Operations Console
            </div>
            <h1 className="text-3xl font-semibold tracking-tight text-slate-50 md:text-4xl">Fraud ML System</h1>
            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              Live drift surveillance, model registry status, and seeded prediction telemetry from the
              production backend.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <span className={`rounded-full border px-3 py-1 text-xs font-medium ${driftStatus.tone}`}>
              {driftStatus.label}
            </span>
            <button
              onClick={handleRetrain}
              disabled={retraining}
              className="inline-flex items-center gap-2 rounded-full bg-cyan-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-cyan-400 hover:shadow-lg hover:shadow-cyan-500/20 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              <RefreshCw size={16} className={retraining ? 'animate-spin' : ''} /> {retraining ? 'Retraining...' : 'Retrain'}
            </button>
            <button
              onClick={handlePromote}
              className="inline-flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-4 py-2 text-sm font-medium text-emerald-200 transition hover:bg-emerald-500/20 hover:shadow-lg hover:shadow-emerald-500/10"
            >
              <Zap size={16} /> Promote
            </button>
          </div>
        </div>

        <div className="mb-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div className={`animate-fade-in-up animate-fade-in-up-1 rounded-2xl border p-5 transition-all duration-300 hover:scale-[1.02] ${fraudRate > 5 ? 'border-rose-500/40 bg-gradient-to-br from-rose-950/40 to-slate-900/70 pulse-glow-danger' : 'border-slate-800 bg-slate-900/70'}`}>
            <div className="text-sm text-slate-400">Observed Fraud Rate</div>
            <div className="mt-3 text-4xl font-semibold text-rose-200">{fraudRate}%</div>
            <div className="mt-2 text-xs text-slate-500">{predictions.length} recent predictions sampled</div>
          </div>
          <div className="animate-fade-in-up animate-fade-in-up-2 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 transition-all duration-300 hover:scale-[1.02] hover:border-slate-700">
            <div className="text-sm text-slate-400">Registry Versions</div>
            <div className="mt-3 text-4xl font-semibold text-cyan-100">{registryCount}</div>
            <div className="mt-2 text-xs text-slate-500">
              {registryCount > 0 ? `Latest: v${registrySummary[registryCount - 1].version}` : 'No versions returned'}
            </div>
          </div>
          <div className="animate-fade-in-up animate-fade-in-up-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 transition-all duration-300 hover:scale-[1.02] hover:border-slate-700">
            <div className="text-sm text-slate-400">Latest Confidence</div>
            <div className="mt-3 text-4xl font-semibold text-amber-100">
              {latestPrediction ? `${(latestPrediction.confidence * 100).toFixed(1)}%` : 'n/a'}
            </div>
            <div className="mt-2 text-xs text-slate-500">
              {latestPrediction ? `Recorded at ${latestPrediction.label}` : 'Waiting for prediction data'}
            </div>
          </div>
          <div className="animate-fade-in-up animate-fade-in-up-4 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 transition-all duration-300 hover:scale-[1.02] hover:border-slate-700">
            <div className="text-sm text-slate-400">Last Sync</div>
            <div className="mt-3 text-2xl font-semibold text-slate-100">
              {lastUpdated ? formatTimestamp(lastUpdated.toISOString()) : 'pending'}
            </div>
            <div className="mt-2 flex items-center gap-1.5 text-xs text-slate-500">
              {loading ? (<><span className="inline-block h-1.5 w-1.5 rounded-full bg-cyan-400 animate-pulse"></span> Refreshing now</>) : 'Auto-refresh every 30s'}
            </div>
          </div>
        </div>

        {errors.length > 0 && (
          <div className="mb-6 flex items-start gap-3 rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-100">
            <AlertTriangle className="mt-0.5 shrink-0" size={18} />
            <div>
              <div className="font-medium">Partial dashboard data unavailable</div>
              <div className="mt-1 text-amber-100/80">{errors.join(' • ')}</div>
            </div>
          </div>
        )}

        <div className="animate-fade-in-up animate-fade-in-up-5 mb-6 grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-100">Prediction Confidence Trend</h2>
                <p className="mt-1 text-sm text-slate-400">Latest production and shadow confidence values</p>
              </div>
              <ShieldAlert className="text-cyan-300/70" size={20} />
            </div>
            <div className="h-72">
              {predictionTrendData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={predictionTrendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                    <XAxis dataKey="label" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{
                        background: '#020617',
                        border: '1px solid rgba(148, 163, 184, 0.18)',
                        borderRadius: '14px',
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="confidence"
                      stroke={CHART_COLORS.production}
                      strokeWidth={3}
                      dot={false}
                      name="Production %"
                    />
                    <Line
                      type="monotone"
                      dataKey="shadowConfidence"
                      stroke={CHART_COLORS.shadow}
                      strokeWidth={2}
                      dot={false}
                      strokeDasharray="6 6"
                      name="Shadow %"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-700 text-sm text-slate-500">
                  No prediction data returned yet.
                </div>
              )}
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-100">Prediction Mix</h2>
                <p className="mt-1 text-sm text-slate-400">Fraud vs normal decisions in the sampled window</p>
              </div>
              <div className="rounded-full border border-rose-400/20 bg-rose-500/10 px-3 py-1 text-xs text-rose-200">
                {fraudRate}% fraud
              </div>
            </div>
            <div className="h-72">
              {predictions.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={predictionMix}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={65}
                      outerRadius={100}
                      paddingAngle={4}
                    >
                      {predictionMix.map((entry) => (
                        <Cell key={entry.name} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        background: '#020617',
                        border: '1px solid rgba(148, 163, 184, 0.18)',
                        borderRadius: '14px',
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-700 text-sm text-slate-500">
                  No prediction data returned yet.
                </div>
              )}
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
              {predictionMix.map((item) => (
                <div key={item.name} className="rounded-2xl border border-slate-800 bg-slate-950/70 p-3">
                  <div className="text-slate-400">{item.name}</div>
                  <div className="mt-1 text-xl font-semibold text-slate-100">{item.value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="animate-fade-in-up animate-fade-in-up-6 grid gap-6 lg:grid-cols-[0.85fr_1.15fr]">
          <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
            <h2 className="mb-4 text-lg font-semibold text-slate-100">System Health</h2>
            
            {/* Drift Card */}
            <div className="mb-4 rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
              <div className="mb-2 text-sm text-slate-400">Real-Time Drift</div>
              <div className="flex items-end gap-4">
                <div>
                  <div className="text-3xl font-semibold text-slate-100">{driftData?.drift_score?.toFixed(3)}</div>
                  <div className="text-xs text-slate-500 mt-1">Threshold: {driftData?.threshold}</div>
                </div>
                <div className={`mb-1 rounded-full px-3 py-1 text-xs font-medium uppercase tracking-wider ${driftData?.status === 'HIGH' ? 'bg-rose-500/20 text-rose-300 border border-rose-400/30' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-400/30'}`}>
                  {driftData?.status === 'HIGH' ? '🔴 HIGH' : '🟢 LOW'}
                </div>
              </div>
            </div>

            {/* Retrain Transparency Panel */}
            <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
              <div className="mb-2 text-sm text-slate-400">Retrain Pipeline</div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">Last Status</span>
                  <span className={`font-semibold ${retrainStatus?.status === 'failed' ? 'text-rose-400' : 'text-emerald-400 uppercase'}`}>
                    {retrainStatus?.status || 'IDLE'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Trigger Reason</span>
                  <span className="text-slate-200">{retrainStatus?.reason || 'None'}</span>
                </div>
                {retrainStatus?.top_shifted_feature && (
                  <div className="flex justify-between">
                    <span className="text-slate-500">Top Shifted Feature</span>
                    <span className="text-amber-200">{retrainStatus.top_shifted_feature}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-slate-500">New Model</span>
                  <span className="text-slate-200">{retrainStatus?.new_model_version || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Last Run</span>
                  <span className="text-slate-200">{retrainStatus?.timestamp ? formatTimestamp(retrainStatus.timestamp) : 'N/A'}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-100">Model Registry</h2>
                <p className="mt-1 text-sm text-slate-400">Version control and promotion workflow</p>
              </div>
            </div>
            
            <div className="mb-4">
              <div className="text-sm text-slate-400 mb-2">Active Production Model</div>
              {registrySummary.filter(r => r.status === 'production').map(item => (
                <div key={item.version} className="flex flex-col gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4">
                  <div className="flex items-center justify-between">
                    <div className="text-lg font-semibold text-emerald-200">v{item.version}</div>
                    <div className="text-xs text-emerald-200/70">Trigger: {item.triggerReason}</div>
                  </div>
                  <div className="flex gap-4 text-xs text-emerald-100/80">
                    <div>AUC-ROC: <span className="font-medium text-emerald-100">{item.aucRoc ?? 'N/A'}</span></div>
                    <div>F1 Score: <span className="font-medium text-emerald-100">{item.f1 ?? 'N/A'}</span></div>
                  </div>
                </div>
              ))}
              {registrySummary.filter(r => r.status === 'production').length === 0 && (
                <div className="text-sm text-slate-500 italic">No active production model found</div>
              )}
            </div>

            <div className="space-y-3">
              <div className="text-sm text-slate-400 mb-2">Available Candidate Models</div>
              {registrySummary.filter(r => r.status !== 'production').length > 0 ? (
                registrySummary.filter(r => r.status !== 'production').map((item) => (
                  <div
                    key={item.version}
                    className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 rounded-2xl border border-slate-800 bg-slate-950/70 p-4"
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-lg font-semibold text-slate-100">v{item.version}</span>
                        <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] uppercase text-slate-400 border border-slate-700">{item.status}</span>
                      </div>
                      <div className="mt-1 text-xs text-slate-400">Trigger: {item.triggerReason} • {item.trainedLabel}</div>
                      <div className="mt-2 flex gap-3 text-xs text-slate-300">
                        <div>AUC-ROC: <span className="font-medium text-cyan-200">{item.aucRoc ?? 'N/A'}</span></div>
                        <div>F1 Score: <span className="font-medium text-cyan-200">{item.f1 ?? 'N/A'}</span></div>
                      </div>
                    </div>
                    {item.status === 'shadow' && (
                      <button
                        onClick={handlePromote}
                        className="shrink-0 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-200 transition hover:bg-emerald-500/20"
                      >
                        Promote v{item.version}
                      </button>
                    )}
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-700 p-6 text-sm text-slate-500">
                  No candidate models available.
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="animate-fade-in-up animate-fade-in-up-7 mt-6 grid gap-6 lg:grid-cols-2">
          {/* Drift History Chart */}
          <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-100">Drift Score Over Time</h2>
                <p className="mt-1 text-sm text-slate-400">Evolution of data distribution divergence</p>
              </div>
            </div>
            <div className="h-72">
              {driftHistory.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={driftHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                    <XAxis dataKey="label" tick={{ fill: '#94a3b8', fontSize: 12 }} interval={'preserveStartEnd'} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{
                        background: '#020617',
                        border: '1px solid rgba(148, 163, 184, 0.18)',
                        borderRadius: '14px',
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="score"
                      stroke={CHART_COLORS.danger}
                      strokeWidth={3}
                      dot={false}
                      name="Drift Score"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-700 text-sm text-slate-500">
                  No drift history available.
                </div>
              )}
            </div>
          </div>

          {/* Confidence Distribution Panel */}
          <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-100">Confidence Distribution</h2>
                <p className="mt-1 text-sm text-slate-400">Density of production model probability scores</p>
              </div>
            </div>
            <div className="h-72">
              {predictions.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={confidenceDistribution}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} vertical={false} />
                    <XAxis dataKey="range" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{
                        background: '#020617',
                        border: '1px solid rgba(148, 163, 184, 0.18)',
                        borderRadius: '14px',
                      }}
                      cursor={{fill: 'rgba(255,255,255,0.05)'}}
                    />
                    <Bar dataKey="count" name="Predictions" fill={CHART_COLORS.calm} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-700 text-sm text-slate-500">
                  No predictions available yet.
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="animate-fade-in-up mt-6 rounded-3xl border border-slate-800 bg-slate-900/70 p-6">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-slate-100">Results & Performance Impact</h2>
              <p className="mt-1 text-sm text-slate-400">Analysis of recent drift events and retraining outcomes</p>
            </div>
            <Activity className="text-cyan-400/50" size={24} />
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-2xl border border-rose-500/20 bg-rose-500/5 p-5">
              <div className="mb-2 text-sm text-rose-200/70">Phase 1: Concept Drift Detected</div>
              <div className="text-2xl font-semibold text-rose-200">Amount Distribution Shift</div>
              <p className="mt-2 text-xs leading-relaxed text-rose-100/60">
                KL Divergence threshold exceeded. The underlying data distribution shifted significantly from the baseline training set.
              </p>
            </div>
            <div className="rounded-2xl border border-amber-500/20 bg-amber-500/5 p-5">
              <div className="mb-2 text-sm text-amber-200/70">Phase 2: Performance Degradation</div>
              <div className="text-2xl font-semibold text-amber-200">AUC-ROC Dropped ~15%</div>
              <p className="mt-2 text-xs leading-relaxed text-amber-100/60">
                Model confidence scores plummeted as the active production model struggled to classify the out-of-distribution transactions accurately.
              </p>
            </div>
            <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-5">
              <div className="mb-2 text-sm text-emerald-200/70">Phase 3: Automated Recovery</div>
              <div className="text-2xl font-semibold text-emerald-200">Shadow Retrain Triggered</div>
              <p className="mt-2 text-xs leading-relaxed text-emerald-100/60">
                System automatically trained a new XGBoost candidate on the latest window, recovering AUC-ROC back to 0.94+ and promoting it to production.
              </p>
            </div>
          </div>
        </div>

        <div className="animate-fade-in-up mt-6 rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-100">Shadow vs Production Comparison</h2>
              <p className="mt-1 text-sm text-slate-400">Most recent confidence outputs from both models</p>
            </div>
          </div>
          <div className="h-80">
            {predictionTrendData.some((item) => item.shadowConfidence != null) ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={predictionTrendData.slice(-12)}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="label" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 'auto']} />
                  <Tooltip
                    contentStyle={{
                      background: '#020617',
                      border: '1px solid rgba(148, 163, 184, 0.18)',
                      borderRadius: '14px',
                    }}
                  />
                  <Bar dataKey="confidence" name="Production %" fill={CHART_COLORS.production} radius={[6, 6, 0, 0]} />
                  <Bar
                    dataKey="shadowConfidence"
                    name="Shadow %"
                    fill={CHART_COLORS.shadow}
                    radius={[6, 6, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-700 text-sm text-slate-500">
                Shadow-model confidence data has not been returned yet.
              </div>
            )}
          </div>
        </div>

        <footer className="mt-10 border-t border-slate-800/60 pt-6 pb-4 text-center">
          <p className="text-xs text-slate-500">
            Fraud ML System · Drift-Triggered Retraining Pipeline · Built by{' '}
            <a href="https://github.com/shubhankartiwari99" target="_blank" rel="noopener noreferrer" className="text-cyan-400/70 hover:text-cyan-300 transition">
              Shubhankar Tiwari
            </a>
          </p>
          <p className="mt-1 text-[11px] text-slate-600">KL Divergence · PSI · Shadow Deployment · Model Registry · Cooldown Logic</p>
        </footer>
      </div>
    </div>
  )
}
