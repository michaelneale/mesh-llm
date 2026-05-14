import type { CSSProperties } from 'react'
import { Activity, Network } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { StatusPill } from '@/components/ui/status-pill'
import { formatReserveVram } from '@/features/reserves/lib/reserve-formatters'
import { getReserveStateMeta } from '@/features/reserves/lib/reserve-state'
import type { ReserveNode } from '@/features/reserves/lib/reserve-types'
import { cn } from '@/lib/utils'

type ReserveActiveWakeCardProps = {
  node: ReserveNode
}

type ReserveActiveWakeCardStyle = CSSProperties & {
  '--reserve-active-color': string
}

function clampProgress(progress: number | undefined): number {
  if (progress == null || !Number.isFinite(progress)) return 0
  return Math.min(100, Math.max(0, progress))
}

function formatCompactEta(seconds: number | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return 'pending'
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`
  const wholeMinutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.round(seconds % 60)
  return remainingSeconds > 0 ? `${wholeMinutes}m ${remainingSeconds}s` : `${wholeMinutes}m`
}

export function ReserveActiveWakeCard({ node }: ReserveActiveWakeCardProps) {
  const meta = getReserveStateMeta(node.state)
  const progress = clampProgress(node.progress)
  const stateColorVariable = node.state === 'waking' ? '--color-warn' : '--color-accent'
  const style: ReserveActiveWakeCardStyle = { '--reserve-active-color': `var(${stateColorVariable})` }

  return (
    <Card
      className="rounded-[var(--radius)] border-[color:color-mix(in_oklab,var(--reserve-active-color)_22%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--reserve-active-color)_4%,var(--color-panel-strong))] shadow-none"
      style={style}
    >
      <CardContent className="space-y-2.5 p-[11px]">
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-start gap-2.5">
            <span
              className={cn(
                'mt-0.5 inline-flex size-[26px] shrink-0 items-center justify-center rounded-[var(--radius)] border text-[11px] font-semibold',
                meta.dotClassName
              )}
            >
              {node.state === 'joining' ? (
                <Network aria-hidden="true" className="size-3" />
              ) : (
                <Activity aria-hidden="true" className="size-3" />
              )}
            </span>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-1.5 text-[13px] font-semibold leading-tight text-foreground [overflow-wrap:anywhere]">
                <span>{node.id}</span>
                <StatusPill label={meta.label} tone={meta.tone} dot />
              </div>
              <div className="mt-0.5 text-[11.5px] leading-tight text-fg-dim">{node.hw}</div>
              {node.location ? <div className="type-caption mt-1 text-fg-faint">{node.location}</div> : null}
            </div>
          </div>
          <div className="shrink-0 text-right">
            <div className="text-[13px] font-semibold leading-none text-foreground">{formatReserveVram(node.vram)}</div>
            <div className="mt-1 text-[9.5px] font-semibold uppercase leading-none tracking-[0.06em] text-fg-faint">
              VRAM
            </div>
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-[11px] leading-none text-fg-dim">
            <span>ETA {formatCompactEta(node.eta)}</span>
            <span>{progress}%</span>
          </div>
          <div
            className="h-[3px] overflow-hidden rounded-full bg-muted"
            role="progressbar"
            aria-label={`${node.id} wake progress`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={progress}
          >
            <div
              className="h-full rounded-full bg-[color:var(--reserve-active-color)]"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
        {node.note ? (
          <div className="type-caption rounded-[var(--radius)] border border-border/70 bg-panel px-2.5 py-2 text-fg-dim">
            {node.note}
          </div>
        ) : null}
        <div className="flex flex-wrap gap-1">
          {node.models.map((model) => (
            <Badge key={model} className="rounded-[var(--radius)] px-2 py-0.5 text-[11px] leading-none text-foreground">
              {model}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
