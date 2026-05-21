import { useMemo, useState } from 'react'
import { AlertTriangle, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ReserveActiveWakeCard } from '@/features/reserves/components/ReserveActiveWakeCard'
import { ReserveErrorCard } from '@/features/reserves/components/ReserveErrorCard'
import { ReserveNodeDot } from '@/features/reserves/components/ReserveNodeDot'
import { ReserveOnlineSummary } from '@/features/reserves/components/ReserveOnlineSummary'
import { ReserveProviderHeader } from '@/features/reserves/components/ReserveProviderHeader'
import type { ReserveNode, ReserveProvider } from '@/features/reserves/lib/reserve-types'
import { getReserveProviderTotals } from '@/features/reserves/lib/reserve-totals'
import { cn } from '@/lib/utils'

type ReserveProviderGroupProps = {
  provider: ReserveProvider
  onDismissNode?: (provider: ReserveProvider, node: ReserveNode) => void
  onOpenLogs?: (provider: ReserveProvider, node: ReserveNode) => void
  onRetryAll?: (provider: ReserveProvider) => void
  onRetryNode?: (provider: ReserveProvider, node: ReserveNode) => void
  onWakeProvider?: (provider: ReserveProvider) => void
}

export function ReserveProviderGroup({
  provider,
  onDismissNode,
  onOpenLogs,
  onRetryAll,
  onRetryNode,
  onWakeProvider
}: ReserveProviderGroupProps) {
  const [expanded, setExpanded] = useState(false)
  const totals = useMemo(() => getReserveProviderTotals(provider), [provider])
  const activeNodes = provider.nodes.filter((node) => node.state === 'waking' || node.state === 'joining')
  const errorNodes = provider.nodes.filter((node) => node.state === 'failed' || node.state === 'unreachable')
  const onlineNodes = provider.nodes.filter((node) => node.state === 'online')
  const allOnline = totals.totalNodes > 0 && totals.counts.online === totals.totalNodes

  return (
    <section
      className={cn(
        'overflow-hidden rounded-[var(--radius)] border shadow-none',
        allOnline
          ? 'border-[color:color-mix(in_oklab,var(--color-good)_24%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-good)_3%,var(--color-panel-strong))]'
          : errorNodes.length > 0
            ? 'border-[color:color-mix(in_oklab,var(--color-bad)_22%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-bad)_2%,var(--color-panel-strong))]'
            : 'border-border/70 bg-panel-strong'
      )}
      data-reserve-entrance
      data-testid="reserve-provider-group"
    >
      <ReserveProviderHeader
        expanded={expanded}
        onToggle={() => setExpanded((value) => !value)}
        provider={provider}
        totals={totals}
        onWakeProvider={() => onWakeProvider?.(provider)}
      />
      {expanded ? (
        <div className="border-t border-border/60 bg-[color:color-mix(in_oklab,var(--color-panel)_70%,var(--color-panel-strong))]">
          {activeNodes.length > 0 ? (
            <div
              className={cn(
                'grid gap-2 border-b border-border/60 bg-[color:color-mix(in_oklab,var(--color-panel)_50%,var(--color-panel-strong))] px-[14px] py-[10px]',
                activeNodes.length > 1 ? 'md:grid-cols-2' : ''
              )}
            >
              {activeNodes.map((node) => (
                <ReserveActiveWakeCard key={node.id} node={node} />
              ))}
            </div>
          ) : null}

          {errorNodes.length > 0 ? (
            <div className="border-b border-[color:color-mix(in_oklab,var(--color-bad)_15%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-bad)_4%,var(--color-panel-strong))] px-[14px] py-[10px]">
              <div className="mb-2 flex items-center gap-2">
                <AlertTriangle className="size-[11px] shrink-0 text-[color:var(--color-bad)]" aria-hidden="true" />
                <div className="type-label text-[color:var(--color-bad)]">{errorNodes.length} nodes need attention</div>
                <span className="min-w-0 flex-1" />
                <Button
                  className="ui-control h-[22px] gap-1.5 rounded-[var(--radius)] border px-2.5 text-[11px]"
                  onClick={() => onRetryAll?.(provider)}
                  size="sm"
                  variant="outline"
                >
                  <RotateCcw className="size-[10px]" aria-hidden="true" />
                  Retry all
                </Button>
              </div>
              <div className="grid gap-2 md:grid-cols-2">
                {errorNodes.map((node) => (
                  <ReserveErrorCard
                    key={node.id}
                    node={node}
                    onDismiss={(dismissedNode) => onDismissNode?.(provider, dismissedNode)}
                    onOpenLogs={(selectedNode) => onOpenLogs?.(provider, selectedNode)}
                    onRetry={(retryNode) => onRetryNode?.(provider, retryNode)}
                  />
                ))}
              </div>
            </div>
          ) : null}

          {onlineNodes.length > 0 ? (
            <div className="border-b border-border/60 bg-[color:color-mix(in_oklab,var(--color-panel)_50%,var(--color-panel-strong))] px-[14px] py-[10px]">
              <ReserveOnlineSummary nodes={onlineNodes} />
            </div>
          ) : null}
          <div className="px-[14px] py-3">
            <div className="flex flex-wrap items-center gap-[4px]">
              {provider.nodes.map((node) => (
                <ReserveNodeDot key={node.id} node={node} />
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </section>
  )
}
