import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { WakeableNode } from '@/features/app-shell/lib/status-types'
import { StatusPill } from '@/features/dashboard/components/details'

const WAKEABLE_NODE_STATE_LABELS = {
  sleeping: 'Sleeping',
  waking: 'Waking'
} as const

export function ReservesPanel({ wakeableNodes }: { wakeableNodes?: WakeableNode[] }) {
  const reserveCount = wakeableNodes?.length ?? 0

  return (
    <Card data-testid="reserves-section">
      <CardHeader className="pb-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="type-label">Reserves</CardTitle>
          <div className="type-caption text-fg-dim">
            {reserveCount} node{reserveCount === 1 ? '' : 's'}
          </div>
        </div>
        <div className="type-caption text-fg-dim">
          Provider-backed reserve capacity that can be woken on demand. These nodes are kept separate from the live
          topology until they rejoin the mesh.
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {reserveCount > 0 ? (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {wakeableNodes?.map((node) => {
              const etaLabel = node.wake_eta_secs == null ? null : formatWakeEta(node.wake_eta_secs)
              return (
                <div key={node.logical_id} className="rounded-[var(--radius)] border bg-panel-strong p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="type-label font-medium [overflow-wrap:anywhere]">{node.logical_id}</div>
                      <div className="mt-1 flex flex-wrap items-center gap-1.5">
                        <StatusPill
                          label={WAKEABLE_NODE_STATE_LABELS[node.state]}
                          tone={node.state === 'waking' ? 'warn' : 'cold'}
                          dot
                        />
                        {node.provider ? (
                          <Badge className="h-5 rounded-full px-2 text-[10px] font-medium text-foreground">
                            {node.provider}
                          </Badge>
                        ) : null}
                      </div>
                    </div>
                    <div className="text-right text-fg-dim">
                      <div className="type-caption font-medium text-foreground">{node.vram_gb.toFixed(1)} GB</div>
                      <div className="type-caption">VRAM</div>
                    </div>
                  </div>

                  <div className="mt-3 grid gap-3 sm:grid-cols-2">
                    <div className="space-y-1 sm:col-span-2">
                      <div className="type-caption font-medium text-fg-dim">Models</div>
                      {node.models.length > 0 ? (
                        <div className="flex flex-wrap gap-1.5">
                          {node.models.map((model) => (
                            <Badge
                              key={model}
                              className="max-w-full rounded-[var(--radius)] px-2 py-0.5 text-[11px] text-foreground"
                            >
                              <span className="truncate [overflow-wrap:anywhere]">{model}</span>
                            </Badge>
                          ))}
                        </div>
                      ) : (
                        <div className="type-caption text-fg-faint">No advertised models</div>
                      )}
                    </div>

                    {etaLabel ? (
                      <div className="space-y-1">
                        <div className="type-caption font-medium text-fg-dim">ETA</div>
                        <div className="type-caption text-foreground">{etaLabel}</div>
                      </div>
                    ) : null}
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="rounded-[var(--radius)] border border-dashed bg-panel-strong p-4 type-caption text-fg-faint">
            No reserve nodes are advertised yet. Add provider-backed nodes or wait for live status to report wakeable
            capacity.
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function formatWakeEta(seconds: number) {
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} min`
  if (seconds < 86_400) return `${Math.ceil(seconds / 3600)} hr`
  return `${Math.ceil(seconds / 86_400)} d`
}
