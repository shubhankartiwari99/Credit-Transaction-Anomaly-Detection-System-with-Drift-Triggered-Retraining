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
  const [loading, setLoading] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [errors, setErrors] = useState([])

  const fetchData = async () => {
    setLoading(true)
    const results = await Promise.allSettled([
      fetchJson(`${API_BASE}/metrics`),
      fetchJson(`${API_BASE}/registry`),
      fetchJson(`${API_BASE}/predictions?limit=100`),
    ])

    const [metricsResult, registryResult, predictionsResult] = results
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
    await fetch(`${API_BASE}/retrain`, { method: 'POST' })
    fetchData()
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

  const driftStatus = getDriftStatus(metrics)

  const latestPrediction = predictions[predictions.length - 1] || null

  const registryCount = registrySummary.length

  return (
    <div className="min-h-screen bg-slate-950 px-4 py-6 text-slate-100 md:px-6">
      <div className="mx-auto max-w-7xl">
        <div className="mb-6 flex flex-col gap-4 rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl shadow-slate-950/40 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-cyan-300/70">
              <Activity size={14} />
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
              className="inline-flex items-center gap-2 rounded-full bg-cyan-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-cyan-400"
            >
              <RefreshCw size={16} /> Retrain
            </button>
            <button
              onClick={handlePromote}
              className="inline-flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-4 py-2 text-sm font-medium text-emerald-200 transition hover:bg-emerald-500/20"
            >
              <Zap size={16} /> Promote
            </button>
          </div>
        </div>

        <div className="mb-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
            <div className="text-sm text-slate-400">Observed Fraud Rate</div>
            <div className="mt-3 text-4xl font-semibold text-rose-200">{fraudRate}%</div>
            <div className="mt-2 text-xs text-slate-500">{predictions.length} recent predictions sampled</div>
          </div>
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
            <div className="text-sm text-slate-400">Registry Versions</div>
            <div className="mt-3 text-4xl font-semibold text-cyan-100">{registryCount}</div>
            <div className="mt-2 text-xs text-slate-500">
              {registryCount > 0 ? `Latest: v${registrySummary[registryCount - 1].version}` : 'No versions returned'}
            </div>
          </div>
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
            <div className="text-sm text-slate-400">Latest Confidence</div>
            <div className="mt-3 text-4xl font-semibold text-amber-100">
              {latestPrediction ? `${(latestPrediction.confidence * 100).toFixed(1)}%` : 'n/a'}
            </div>
            <div className="mt-2 text-xs text-slate-500">
              {latestPrediction ? `Recorded at ${latestPrediction.label}` : 'Waiting for prediction data'}
            </div>
          </div>
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
            <div className="text-sm text-slate-400">Last Sync</div>
            <div className="mt-3 text-2xl font-semibold text-slate-100">
              {lastUpdated ? formatTimestamp(lastUpdated.toISOString()) : 'pending'}
            </div>
            <div className="mt-2 text-xs text-slate-500">{loading ? 'Refreshing now' : 'Auto-refresh every 30s'}</div>
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

        <div className="mb-6 grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
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

        <div className="grid gap-6 lg:grid-cols-[0.85fr_1.15fr]">
          <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
            <h2 className="mb-4 text-lg font-semibold text-slate-100">Drift Snapshot</h2>
            <div className="space-y-3">
              {[
                { label: 'Amount KL', value: metrics.amountKl },
                { label: 'Amount PSI', value: metrics.amountPsi },
                { label: 'Confidence KL', value: metrics.confidenceKl },
                { label: 'Confidence PSI', value: metrics.confidencePsi },
              ].map((item) => (
                <div key={item.label} className="rounded-2xl border border-slate-800 bg-slate-950/70 p-3">
                  <div className="text-sm text-slate-400">{item.label}</div>
                  <div className="mt-1 text-2xl font-semibold text-slate-100">{item.value.toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-100">Model Registry</h2>
                <p className="mt-1 text-sm text-slate-400">Version and promotion state returned by the backend</p>
              </div>
            </div>
            <div className="space-y-3">
              {registrySummary.length > 0 ? (
                registrySummary.map((item) => (
                  <div
                    key={item.version}
                    className="grid gap-3 rounded-2xl border border-slate-800 bg-slate-950/70 p-4 md:grid-cols-[auto_1fr_auto]"
                  >
                    <div className="text-lg font-semibold text-slate-100">v{item.version}</div>
                    <div>
                      <div className="text-sm text-slate-300">{item.triggerReason}</div>
                      <div className="mt-1 text-xs text-slate-500">{item.trainedLabel}</div>
                    </div>
                    <div
                      className={`rounded-full px-3 py-1 text-xs font-medium ${
                        item.status === 'production'
                          ? 'border border-emerald-400/30 bg-emerald-500/10 text-emerald-200'
                          : item.status === 'shadow'
                            ? 'border border-amber-400/30 bg-amber-500/10 text-amber-200'
                            : 'border border-slate-700 bg-slate-800 text-slate-300'
                      }`}
                    >
                      {item.status}
                    </div>
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-700 p-6 text-sm text-slate-500">
                  No registry versions were returned by the API.
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="mt-6 rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
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
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 100]} />
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
      </div>
    </div>
  )
}
