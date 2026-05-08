import type { ReactNode } from 'react'
import {
  DASHBOARD_TOPOLOGY_PANEL_HEIGHT_CLASS,
  DASHBOARD_TOPOLOGY_ROW_HEIGHT_CLASS
} from '@/features/network/components/MeshViz.layout'

type DashboardLayoutProps = {
  hero: ReactNode
  status: ReactNode
  topology: ReactNode
  catalog: ReactNode
  peers: ReactNode
  connect: ReactNode
  drawers: ReactNode
}

export function DashboardLayout({ hero, status, topology, catalog, peers, connect, drawers }: DashboardLayoutProps) {
  return (
    <div className="flex min-w-0 flex-col gap-[14px]">
      {hero}
      {status}
      <div
        className={`grid min-w-0 gap-[14px] xl:grid-cols-[minmax(0,1fr)_minmax(300px,360px)] xl:items-stretch ${DASHBOARD_TOPOLOGY_ROW_HEIGHT_CLASS}`}
      >
        <div className={`flex min-h-0 min-w-0 flex-col ${DASHBOARD_TOPOLOGY_PANEL_HEIGHT_CLASS}`}>{topology}</div>
        <div className={`flex min-h-0 min-w-0 flex-col ${DASHBOARD_TOPOLOGY_PANEL_HEIGHT_CLASS}`}>{catalog}</div>
      </div>
      <div className="min-w-0">{peers}</div>
      <div className="min-w-0">{connect}</div>
      {drawers}
    </div>
  )
}
