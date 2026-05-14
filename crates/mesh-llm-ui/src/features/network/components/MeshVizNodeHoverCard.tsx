import * as HoverCardPrimitive from '@radix-ui/react-hover-card'
import { Network } from 'lucide-react'
import { StatusBadge } from '@/components/ui/StatusBadge'
import type { MeshNode, Peer } from '@/features/app-tabs/types'
import { meshNodeSecondaryLabel } from '@/features/network/lib/mesh-node-labels'
import { meshNodeStatusSource, meshStatusLabel, meshStatusTone } from '@/features/network/lib/mesh-status'
import { hoverCardPlacement, nodeMetrics, roleLabel } from '@/features/network/components/MeshViz.helpers'

type MeshVizNodeHoverCardProps = {
  node: MeshNode
  peer?: Peer
}

export function MeshVizNodeHoverCard({ node, peer }: MeshVizNodeHoverCardProps) {
  const metrics = nodeMetrics(node, peer)
  const ageMetric = metrics.find((metric) => metric.id === 'age')
  const detailMetrics = metrics.filter((metric) => metric.id !== 'age')
  const AgeIcon = ageMetric?.icon
  const title = peer?.hostname ?? node.label
  const nodeId = peer?.shortId ?? node.id
  const role = roleLabel(node, peer)
  const secondaryLabel = meshNodeSecondaryLabel(node, peer)
  const showSecondaryLabel = secondaryLabel.trim().toLowerCase() !== role.trim().toLowerCase()
  const statusSource = meshNodeStatusSource(peer, node)
  const status = meshStatusLabel(statusSource)
  const { side, align } = hoverCardPlacement(node)

  return (
    <HoverCardPrimitive.Portal>
      <HoverCardPrimitive.Content
        id={`mesh-node-popover-${node.id}`}
        role="tooltip"
        side={side}
        align={align}
        sideOffset={22}
        collisionPadding={12}
        className="surface-popover-panel pointer-events-none z-50 w-[264px] rounded-[var(--radius-lg)] px-3 py-2.5 text-left"
      >
        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3 gap-y-2">
          <div className="min-w-0 flex-1">
            <div className="truncate font-mono text-[length:var(--density-type-body)] font-semibold uppercase leading-5 tracking-[0.04em] text-foreground">
              {title}
            </div>
            <div className="mt-0.5 truncate font-mono text-[length:var(--density-type-annotation)] uppercase tracking-[0.08em] text-fg-faint">
              {nodeId}
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              <span className="inline-flex items-center rounded-full border border-border bg-background px-2 py-0.5 text-[length:var(--density-type-label)] font-medium text-fg-dim">
                {role}
              </span>
              {showSecondaryLabel ? (
                <span className="inline-flex items-center rounded-full border border-border bg-background px-2 py-0.5 text-[length:var(--density-type-label)] font-medium text-fg-dim">
                  {secondaryLabel}
                </span>
              ) : null}
            </div>
          </div>
          <div className="flex h-full shrink-0 flex-col items-end justify-between gap-2 text-right">
            <StatusBadge className="mt-0.5 shrink-0" dot tone={meshStatusTone(statusSource)}>
              {status}
            </StatusBadge>
            {ageMetric && AgeIcon ? (
              <div className="inline-flex min-w-[3.25rem] items-center justify-end gap-1.5 font-mono text-[length:var(--density-type-annotation)] uppercase tracking-[0.08em] text-fg-faint">
                <AgeIcon className="size-3 shrink-0" aria-hidden="true" strokeWidth={1.8} />
                <span className="text-[length:var(--density-type-caption-lg)] text-foreground">{ageMetric.value}</span>
              </div>
            ) : null}
          </div>
        </div>

        <div className="mt-3 border-t border-border-soft pt-2.5">
          <div className="grid grid-cols-2 gap-x-3 gap-y-2">
            {detailMetrics.map((metric) => {
              const Icon = metric.icon

              return (
                <div key={metric.id} className="min-w-0">
                  <div className="flex items-center gap-1.5 font-mono text-[length:var(--density-type-micro)] font-medium uppercase tracking-[0.16em] text-fg-faint">
                    <Icon className="size-3 shrink-0" aria-hidden="true" strokeWidth={1.8} />
                    {metric.label}
                  </div>
                  <div className="mt-0.5 truncate font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
                    {metric.value}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {peer ? (
          <div className="mt-2.5 flex items-center gap-1.5 border-t border-border-soft pt-2 font-mono text-[length:var(--density-type-annotation)] uppercase tracking-[0.1em] text-fg-faint">
            <Network className="size-3" aria-hidden="true" strokeWidth={1.8} />
            {peer.sharePct}% mesh share · {peer.hostedModels.length} models
          </div>
        ) : null}
      </HoverCardPrimitive.Content>
    </HoverCardPrimitive.Portal>
  )
}
