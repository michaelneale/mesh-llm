import { cn } from '@/lib/cn'
import type { StatusBadgeTone, StatusMetric } from '@/features/app-tabs/types'

function toneClass(tone?: StatusBadgeTone) {
  if (tone === 'good') return 'text-good before:bg-good'
  if (tone === 'warn') return 'text-warn before:bg-warn'
  if (tone === 'bad') return 'text-bad before:bg-bad'
  if (tone === 'accent') return 'text-accent before:bg-accent'
  return 'text-muted-foreground before:bg-muted-foreground'
}

function Sparkline({ values, color = 'var(--color-accent)' }: { values: number[]; color?: string }) {
  if (!values.length) return null
  const w = 72
  const h = 18
  const max = Math.max(1, ...values)
  const min = Math.min(...values)
  const range = Math.max(1, max - min)
  const denominator = Math.max(1, values.length - 1)
  const pts = values.map((v, i) => `${(i / denominator) * w},${h - ((v - min) / range) * (h - 2) - 1}`).join(' ')
  return (
    <svg width={w} height={h} style={{ display: 'block' }} aria-hidden="true">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <polyline points={`0,${h} ${pts} ${w},${h}`} fill={color} opacity="0.08" stroke="none" />
    </svg>
  )
}

export function StatusTile({ metric }: { metric: StatusMetric }) {
  const { sparkline, badge } = metric
  return (
    <div className="grid min-w-0 flex-1 grid-rows-[14px_22px_18px] gap-y-1 border-r border-border-soft px-3.5 py-[12px] last:border-r-0" style={{ minHeight: 70 }}>
      <div className="flex min-w-0 items-center gap-1.5 text-[length:var(--density-type-label)] font-medium uppercase leading-none tracking-[0.6px] text-fg-faint">
        {metric.icon}
        <span className="truncate">{metric.label}</span>
      </div>
      <div className="flex min-w-0 items-baseline gap-1.5 overflow-hidden">
        <span className="truncate font-mono text-[length:var(--density-type-title)] font-medium leading-none tracking-tight" style={{ letterSpacing: -0.4 }}>
          {metric.value}
        </span>
        {metric.unit && <span className="shrink-0 font-mono text-[length:var(--density-type-label)] uppercase leading-none text-fg-faint">{metric.unit}</span>}
      </div>
      <div className="flex min-w-0 items-center gap-2 overflow-hidden">
        {sparkline && sparkline.length > 0 && (
          <Sparkline values={sparkline} color={metric.id === 'inflight' ? 'var(--color-warn)' : 'var(--color-accent)'} />
        )}
        {badge && (
          <span className={cn(
            'inline-flex items-center gap-[5px] text-[length:var(--density-type-label)] font-medium',
            'before:size-[5px] before:shrink-0 before:rounded-full before:content-[""]',
            toneClass(badge.tone),
          )}>
            {badge.label}
          </span>
        )}
        {metric.meta && <span className="text-[length:var(--density-type-label)] text-fg-faint">{metric.meta}</span>}
      </div>
    </div>
  )
}

export function StatusStrip({ metrics }: { metrics: StatusMetric[] }) {
  return (
    <section aria-label="Network status" className="panel-shell flex overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
      {metrics.map((metric) => <StatusTile key={metric.id} metric={metric} />)}
    </section>
  )
}
