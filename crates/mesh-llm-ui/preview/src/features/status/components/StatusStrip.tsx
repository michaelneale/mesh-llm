import { createElement, useEffect, useMemo, useRef, useState } from 'react'
import { Activity, Cpu, HardDrive, Hash, Network, UserRound, type LucideIcon } from 'lucide-react'
import { Sparkline } from '@/components/ui/Sparkline'
import { cn } from '@/lib/cn'
import type { StatusBadgeTone, StatusMetric } from '@/features/app-tabs/types'

const METRIC_HISTORY_LIMIT = 60
const METRIC_HISTORY_INTERVAL_MS = 60_000

type MetricHistoryValues = {
  meshVram: number
  inflight: number
}

const EMPTY_METRIC_HISTORY_VALUES: MetricHistoryValues = {
  meshVram: 0,
  inflight: 0
}

type MetricHistorySample = MetricHistoryValues & {
  timestamp: number
}

type MetricHistoryState = {
  resetKey: string
  samples: MetricHistorySample[]
}

type StatusStripProps = {
  metrics: StatusMetric[]
  historyKey?: string
  historyPointCount?: number
  historyIntervalMs?: number
}

function metricIcon(Icon: LucideIcon) {
  return createElement(Icon, { className: 'size-[11px] shrink-0', 'aria-hidden': true })
}

const DEFAULT_METRIC_ICONS: Record<string, ReturnType<typeof metricIcon>> = {
  'node-id': metricIcon(Hash),
  owner: metricIcon(UserRound),
  'active-models': metricIcon(Cpu),
  'mesh-vram': metricIcon(HardDrive),
  nodes: metricIcon(Network),
  inflight: metricIcon(Activity)
}

function metricNumber(value: StatusMetric['value']): number {
  const parsed = typeof value === 'number' ? value : Number.parseFloat(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function metricHistorySample(values: MetricHistoryValues): MetricHistorySample {
  return { ...values, timestamp: Date.now() }
}

function normalizedHistoryPointCount(pointCount: number) {
  return Math.max(1, Math.floor(pointCount))
}

function seedMetricHistory(pointCount: number) {
  return Array.from({ length: normalizedHistoryPointCount(pointCount) }, () =>
    metricHistorySample(EMPTY_METRIC_HISTORY_VALUES)
  )
}

function useMetricHistory(metrics: StatusMetric[], resetKey: string, pointCount: number, intervalMs: number) {
  const historyPointCount = normalizedHistoryPointCount(pointCount)
  const values = useMemo<MetricHistoryValues>(
    () => ({
      meshVram: metricNumber(metrics.find((metric) => metric.id === 'mesh-vram')?.value ?? 0),
      inflight: metricNumber(metrics.find((metric) => metric.id === 'inflight')?.value ?? 0)
    }),
    [metrics]
  )
  const latestValues = useRef(values)
  const [history, setHistory] = useState<MetricHistoryState>(() => ({
    resetKey,
    samples: seedMetricHistory(historyPointCount)
  }))
  const resetHistory = useMemo<MetricHistoryState>(
    () => ({
      resetKey,
      samples: seedMetricHistory(historyPointCount)
    }),
    [historyPointCount, resetKey]
  )
  const activeHistory =
    history.resetKey === resetKey && history.samples.length === historyPointCount ? history : resetHistory

  useEffect(() => {
    latestValues.current = values
  }, [values])

  useEffect(() => {
    const timer = window.setInterval(() => {
      setHistory((current) => {
        const currentHistory =
          current.resetKey === resetKey && current.samples.length === historyPointCount ? current : resetHistory

        return {
          resetKey,
          samples: [
            ...currentHistory.samples.slice(-(historyPointCount - 1)),
            metricHistorySample(latestValues.current)
          ]
        }
      })
    }, intervalMs)

    return () => window.clearInterval(timer)
  }, [historyPointCount, intervalMs, resetHistory, resetKey])

  return activeHistory.samples
}

function mergeMetricHistory(metrics: StatusMetric[], history: MetricHistorySample[]): StatusMetric[] {
  const meshVramHistory = history.map((sample) => sample.meshVram)
  const inflightHistory = history.map((sample) => sample.inflight)
  const hasObservedSample = history.some((sample) => sample.meshVram !== 0 || sample.inflight !== 0)

  return metrics.map((metric) => {
    const defaultIcon = DEFAULT_METRIC_ICONS[metric.id]
    const icon = metric.icon ?? defaultIcon

    if (metric.id === 'mesh-vram') {
      return { ...metric, icon, sparkline: metric.sparkline && !hasObservedSample ? metric.sparkline : meshVramHistory }
    }

    if (metric.id === 'inflight') {
      return { ...metric, icon, sparkline: metric.sparkline && !hasObservedSample ? metric.sparkline : inflightHistory }
    }

    return { ...metric, icon }
  })
}

function toneClass(tone?: StatusBadgeTone) {
  if (tone === 'good') return 'text-good before:bg-good'
  if (tone === 'warn') return 'text-warn before:bg-warn'
  if (tone === 'bad') return 'text-bad before:bg-bad'
  if (tone === 'accent') return 'text-accent before:bg-accent'
  return 'text-muted-foreground before:bg-muted-foreground'
}

export function StatusTile({ metric }: { metric: StatusMetric }) {
  const { sparkline, badge } = metric
  return (
    <div
      className="grid min-w-0 flex-1 grid-rows-[14px_22px_18px] gap-y-1 border-r border-border-soft px-3.5 py-[12px] last:border-r-0"
      style={{ minHeight: 70 }}
    >
      <div className="flex min-w-0 items-center gap-1.5 text-[length:var(--density-type-label)] font-medium uppercase leading-none tracking-[0.6px] text-fg-faint">
        {metric.icon}
        <span className="truncate">{metric.label}</span>
      </div>
      <div className="flex min-w-0 items-baseline gap-1.5 overflow-hidden">
        <span
          className="truncate font-mono text-[length:var(--density-type-title)] font-medium leading-none tracking-tight"
          style={{ letterSpacing: -0.4 }}
        >
          {metric.value}
        </span>
        {metric.unit && (
          <span className="shrink-0 font-mono text-[length:var(--density-type-label)] uppercase leading-none text-fg-faint">
            {metric.unit}
          </span>
        )}
      </div>
      <div className="flex min-w-0 items-center gap-2 overflow-hidden">
        {sparkline && sparkline.length > 0 && (
          <Sparkline
            values={sparkline}
            color={metric.id === 'inflight' ? 'var(--color-warn)' : 'var(--color-accent)'}
          />
        )}
        {badge && (
          <span
            className={cn(
              'inline-flex items-center gap-[5px] text-[length:var(--density-type-label)] font-medium',
              'before:size-[5px] before:shrink-0 before:rounded-full before:content-[""]',
              toneClass(badge.tone)
            )}
          >
            {badge.label}
          </span>
        )}
        {metric.meta && <span className="text-[length:var(--density-type-label)] text-fg-faint">{metric.meta}</span>}
      </div>
    </div>
  )
}

export function StatusStrip({
  metrics,
  historyKey,
  historyPointCount = METRIC_HISTORY_LIMIT,
  historyIntervalMs = METRIC_HISTORY_INTERVAL_MS
}: StatusStripProps) {
  const resetKey =
    historyKey ??
    `${metrics.find((metric) => metric.id === 'node-id')?.value ?? ''}:${metrics.map((metric) => metric.id).join('|')}`
  const history = useMetricHistory(metrics, resetKey, historyPointCount, historyIntervalMs)
  const resolvedMetrics = useMemo(() => mergeMetricHistory(metrics, history), [history, metrics])

  return (
    <section
      aria-label="Network status"
      className="panel-shell grid grid-cols-2 overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel sm:grid-cols-3 lg:flex"
    >
      {resolvedMetrics.map((metric) => (
        <StatusTile key={metric.id} metric={metric} />
      ))}
    </section>
  )
}
