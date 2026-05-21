import { AlertTriangle, RotateCcw, Terminal, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { StatusPill } from '@/components/ui/status-pill'
import { formatReserveVram } from '@/features/reserves/lib/reserve-formatters'
import { getReserveStateMeta } from '@/features/reserves/lib/reserve-state'
import type { ReserveNode } from '@/features/reserves/lib/reserve-types'

type ReserveErrorCardProps = {
  node: ReserveNode
  onRetry?: (node: ReserveNode) => void
  onOpenLogs?: (node: ReserveNode) => void
  onDismiss?: (node: ReserveNode) => void
}

export function ReserveErrorCard({ node, onRetry, onOpenLogs, onDismiss }: ReserveErrorCardProps) {
  const meta = getReserveStateMeta(node.state)
  const timestamp = node.failedAt
    ? node.failedAt.startsWith('failed')
      ? node.failedAt
      : `failed ${node.failedAt}`
    : node.lastSeen
      ? node.lastSeen.startsWith('last seen')
        ? node.lastSeen
        : `last seen ${node.lastSeen}`
      : 'No timestamp reported'

  return (
    <Card className="rounded-[var(--radius)] border-[color:color-mix(in_oklab,var(--color-bad)_26%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-bad)_4%,var(--color-panel-strong))] shadow-none">
      <CardContent className="space-y-2 p-[10px]">
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-start gap-2.5">
            <span className="mt-0.5 inline-flex size-[26px] shrink-0 items-center justify-center rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-bad)_40%,transparent)] bg-[color:color-mix(in_oklab,var(--color-bad)_20%,var(--color-panel))] text-[color:var(--color-bad)]">
              <AlertTriangle className="size-[13px]" aria-hidden="true" />
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
        <div className="flex items-center gap-2 rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-bad)_18%,var(--color-border))] bg-panel px-2 py-1 text-[11.5px] leading-snug text-[color:var(--color-bad)]">
          <span
            className="size-[5px] shrink-0 rounded-full bg-[color:var(--color-bad)] shadow-[0_0_8px_color-mix(in_oklab,var(--color-bad)_70%,transparent)]"
            aria-hidden="true"
          />
          <span className="min-w-0 flex-1 font-mono text-[10.5px]">
            {node.error ?? 'Reserve node needs attention.'}
          </span>
          <span className="type-caption shrink-0 text-fg-faint">{timestamp}</span>
        </div>
        {node.note ? <div className="type-caption text-fg-dim">{node.note}</div> : null}
        <div className="flex flex-wrap items-center gap-1.5">
          <Button
            className="ui-control h-[24px] gap-1.5 rounded-[var(--radius)] border px-2.5 text-[11px]"
            disabled={!onRetry || node.retryable === false}
            onClick={() => onRetry?.(node)}
            size="sm"
            variant="outline"
          >
            <RotateCcw className="size-[11px]" aria-hidden="true" />
            Retry
          </Button>
          <Button
            className="ui-control h-[24px] gap-1.5 rounded-[var(--radius)] border px-2.5 text-[11px]"
            disabled={!onOpenLogs}
            onClick={() => onOpenLogs?.(node)}
            size="sm"
            variant="outline"
          >
            <Terminal className="size-[11px]" aria-hidden="true" />
            Logs
          </Button>
          <Button
            aria-label={`Dismiss ${node.id}`}
            className="ml-auto h-[24px] w-[24px] rounded-[var(--radius)] border-0 bg-transparent px-0 text-fg-faint hover:bg-panel hover:text-foreground"
            disabled={!onDismiss}
            onClick={() => onDismiss?.(node)}
            size="sm"
            variant="ghost"
          >
            <X className="size-[11px]" aria-hidden="true" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
