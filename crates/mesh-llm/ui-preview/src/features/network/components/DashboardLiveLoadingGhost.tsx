import { LiveLoadingGhostRoot } from '@/components/ui/LiveLoadingGhostRoot'
import { DashboardLayout } from '@/features/network/layouts/DashboardLayout'
import { MeshVizTopologyGhost } from '@/features/network/components/MeshVizTopologyGhost'
import {
  ConnectBlockLoadingGhost,
  DashboardHeroLoadingGhost,
  DashboardStatusLoadingGhost,
  ModelCatalogLoadingGhost,
  PeersTableLoadingGhost
} from '@/features/network/components/DashboardLoadingGhostSections'

export function DashboardLiveLoadingGhost() {
  return (
    <LiveLoadingGhostRoot>
      <DashboardLayout
        hero={<DashboardHeroLoadingGhost />}
        status={<DashboardStatusLoadingGhost />}
        topology={<MeshVizTopologyGhost />}
        catalog={<ModelCatalogLoadingGhost />}
        peers={<PeersTableLoadingGhost />}
        connect={<ConnectBlockLoadingGhost />}
        drawers={null}
      />
    </LiveLoadingGhostRoot>
  )
}
