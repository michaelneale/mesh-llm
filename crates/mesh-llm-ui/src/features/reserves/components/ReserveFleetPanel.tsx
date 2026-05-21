import { Card } from '@/components/ui/card'
import { Activity } from 'lucide-react'
import { ReserveProviderGroup } from '@/features/reserves/components/ReserveProviderGroup'
import { ReserveStateLegend } from '@/features/reserves/components/ReserveStateLegend'
import { ReserveSummaryStrip } from '@/features/reserves/components/ReserveSummaryStrip'
import type { ReserveNode, ReserveProvider, ReserveWakePolicySettings } from '@/features/reserves/lib/reserve-types'
import { getReserveFleetTotals } from '@/features/reserves/lib/reserve-totals'

type ReserveFleetPanelProps = {
  providers: ReserveProvider[]
  liveMeshVramGB?: number
  wakePolicySettings?: ReserveWakePolicySettings
  onDismissNode?: (provider: ReserveProvider, node: ReserveNode) => void
  onOpenLogs?: (provider: ReserveProvider, node: ReserveNode) => void
  onRetryAll?: (provider: ReserveProvider) => void
  onRetryNode?: (provider: ReserveProvider, node: ReserveNode) => void
  onWakeProvider?: (provider: ReserveProvider) => void
}

export function ReserveFleetPanel({
  providers,
  liveMeshVramGB,
  wakePolicySettings,
  onDismissNode,
  onOpenLogs,
  onRetryAll,
  onRetryNode,
  onWakeProvider
}: ReserveFleetPanelProps) {
  const totals = getReserveFleetTotals(providers)

  return (
    <Card
      className="overflow-hidden rounded-[10px] bg-panel shadow-none"
      data-reserve-entrance
      data-testid="reserves-section"
    >
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/60 bg-[color-mix(in_oklab,var(--color-panel-strong)_50%,var(--color-panel))] px-[14px] py-[10px]">
        <h3 className="text-[12px] font-semibold leading-none tracking-[0.0125em] text-foreground">Reserve fleet</h3>
        <ReserveStateLegend />
      </div>
      <ReserveSummaryStrip
        autoWakeEnabled={wakePolicySettings?.autoWakeEnabled}
        liveMeshVramGB={liveMeshVramGB}
        policyLabel={
          wakePolicySettings ? `policy: ${wakePolicySettings.providerOrder.join(' → ').toLowerCase()}` : undefined
        }
        totals={totals}
      />
      <div className="space-y-3 px-[14px] py-[14px]">
        <div className="space-y-2">
          {providers.length > 0 ? (
            providers.map((provider) => (
              <ReserveProviderGroup
                key={provider.id}
                provider={provider}
                onDismissNode={onDismissNode}
                onOpenLogs={onOpenLogs}
                onRetryAll={onRetryAll}
                onRetryNode={onRetryNode}
                onWakeProvider={onWakeProvider}
              />
            ))
          ) : (
            <div className="flex min-h-[154px] flex-col items-center justify-center rounded-[var(--radius)] border border-dashed border-[color:color-mix(in_oklab,var(--color-accent)_22%,var(--color-border))] bg-[radial-gradient(circle_at_50%_0%,color-mix(in_oklab,var(--color-accent)_10%,transparent),transparent_42%),color-mix(in_oklab,var(--color-panel-strong)_70%,var(--color-panel))] px-5 py-7 text-center">
              <div className="inline-flex size-10 items-center justify-center rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-accent)_34%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_12%,transparent)] text-[color:var(--color-accent)]">
                <Activity className="size-4" aria-hidden="true" />
              </div>
              <div className="mt-3 text-[13px] font-semibold leading-tight text-foreground">
                No reserve providers are configured yet.
              </div>
              <p className="mt-1 max-w-[420px] text-[11.5px] leading-[1.45] text-fg-dim">
                Add cloud VMs, colocated hosts, or office workstations here so they can wake when mesh demand rises.
              </p>
            </div>
          )}
        </div>
      </div>
      <div className="type-caption flex flex-col gap-2 border-t border-border/60 bg-[color-mix(in_oklab,var(--color-panel-strong)_40%,var(--color-panel))] px-[14px] py-[10px] text-fg-dim sm:flex-row sm:items-center sm:justify-between">
        <span>
          Once a node finishes joining it leaves this list and appears in{' '}
          <span className="text-foreground">Connected peers</span> on the Network tab.
        </span>
        <a className="ui-link shrink-0 whitespace-nowrap sm:ml-auto" href="#reserve-policy">
          Configure auto-wake →
        </a>
      </div>
    </Card>
  )
}
