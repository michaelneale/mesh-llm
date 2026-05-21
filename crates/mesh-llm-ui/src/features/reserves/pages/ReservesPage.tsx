import { DASHBOARD_HARNESS } from '@/features/app-tabs/data'
import type { DashboardHarnessData } from '@/features/app-tabs/types'
import { useStatusQuery } from '@/features/network/api/use-status-query'
import { ReservesSurface } from '@/features/reserves/components/ReservesSurface'
import { RESERVE_PROVIDER_FIXTURES, reserveProvidersFromWakeableNodes } from '@/features/reserves/lib/reserve-fixtures'
import type { StatusPayload } from '@/lib/api/types'
import { useDataMode } from '@/lib/data-mode'
import { useBooleanFeatureFlag } from '@/lib/feature-flags'

type ReservesPageProps = { data?: DashboardHarnessData }
type VramNode = { my_vram_gb?: number; vram_gb?: number; gpus?: StatusPayload['gpus'] }

function finiteVramGB(value: number | undefined): number {
  return Number.isFinite(value) && value != null && value > 0 ? value : 0
}

function gpuInventoryVramGB(gpus: VramNode['gpus'] | undefined): number {
  return (gpus ?? []).reduce((sum, gpu) => {
    const fromTotal = finiteVramGB(gpu.total_vram_gb)
    if (fromTotal > 0) return sum + fromTotal

    const fromBytes = finiteVramGB(gpu.vram_bytes) / 1024 ** 3
    return sum + fromBytes
  }, 0)
}

function nodeVramGB(node: VramNode): number {
  return finiteVramGB(node.my_vram_gb) || finiteVramGB(node.vram_gb) || gpuInventoryVramGB(node.gpus)
}

function statusMeshVramGB(status: StatusPayload | undefined): number | undefined {
  if (!status) return undefined
  return status.peers.reduce((sum, peer) => sum + nodeVramGB(peer), nodeVramGB(status))
}

export function ReservesPageContent({ data = DASHBOARD_HARNESS }: ReservesPageProps = {}) {
  const newReservesPageEnabled = useBooleanFeatureFlag('global/newReservesPage')
  const wakePolicyConfigurationEnabled = useBooleanFeatureFlag('configuration/wakePolicyConfiguration')
  const { mode } = useDataMode()
  const liveMode = mode === 'live'
  const statusQuery = useStatusQuery({ enabled: liveMode && newReservesPageEnabled })
  const liveReserveProviders = reserveProvidersFromWakeableNodes(statusQuery.data?.wakeable_nodes)
  const reserveProviders = liveMode ? (liveReserveProviders ?? []) : RESERVE_PROVIDER_FIXTURES
  const harnessMeshVramGB = data.peers.reduce((sum, peer) => sum + (peer.vramGB ?? 0), 0)
  const liveMeshVramGB = liveMode ? statusMeshVramGB(statusQuery.data) : harnessMeshVramGB

  if (!newReservesPageEnabled) {
    return (
      <section className="panel-shell mx-auto max-w-3xl rounded-[var(--radius-lg)] border border-border bg-panel p-6">
        <div className="type-label text-fg-faint">Feature flag disabled</div>
        <h1 className="type-display mt-1 text-foreground">Reserves is gated</h1>
        <p className="type-body mt-2 max-w-[68ch] text-fg-dim">
          Enable <span className="font-mono text-foreground">global/newReservesPage</span> in the developer playground
          to expose this app surface.
        </p>
      </section>
    )
  }

  return (
    <ReservesSurface
      configurationHref={wakePolicyConfigurationEnabled ? '/configuration/wake-policy' : undefined}
      liveMeshVramGB={liveMeshVramGB}
      providers={reserveProviders}
    />
  )
}

export function ReservesPage(props: ReservesPageProps = {}) {
  return <ReservesPageContent {...props} />
}
