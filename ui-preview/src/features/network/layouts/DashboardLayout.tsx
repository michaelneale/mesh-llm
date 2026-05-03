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
    <div className="flex min-w-0 flex-col gap-[14px]">
      {hero}
      {status}
      <div className="grid min-w-0 gap-[14px] xl:grid-cols-[minmax(0,1fr)_minmax(300px,360px)]">
        <div className="min-w-0">{topology}</div>
        <div className="min-w-0">{catalog}</div>
      </div>
      <div className="min-w-0">{peers}</div>
      <div className="min-w-0">{connect}</div>
      {drawers}
    </div>
  )
}
