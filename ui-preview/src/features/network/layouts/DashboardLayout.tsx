import type { ReactNode } from 'react'

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
    <div className="flex flex-col gap-[14px]">
      {hero}
      {status}
      <div className="grid gap-[14px]" style={{ gridTemplateColumns: '1fr 360px' }}>
        {topology}
        {catalog}
      </div>
      {peers}
      {connect}
      {drawers}
    </div>
  )
}
