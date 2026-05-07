import { LoadingGhostBlock } from '@/components/ui/LoadingGhostBlock'

const STATUS_GHOST_ITEMS = [
  { id: 'node-id', labelWidth: 'w-16', valueWidth: 'w-24', metaWidth: 'w-12', meta: 'badge' },
  { id: 'owner', labelWidth: 'w-14', valueWidth: 'w-20', metaWidth: 'w-36', meta: 'badge' },
  { id: 'nodes', labelWidth: 'w-14', valueWidth: 'w-10', metaWidth: 'w-24', meta: 'text' },
  { id: 'active-models', labelWidth: 'w-24', valueWidth: 'w-8', metaWidth: 'w-32', meta: 'text' },
  { id: 'mesh-vram', labelWidth: 'w-20', valueWidth: 'w-16', metaWidth: 'w-16', meta: 'sparkline' },
  { id: 'inflight', labelWidth: 'w-16', valueWidth: 'w-8', metaWidth: 'w-16', meta: 'sparkline' }
]
const CATALOG_GHOST_ROWS = ['model-a', 'model-b', 'model-c', 'model-d']
const PEER_GHOST_ROWS = ['peer-a', 'peer-b', 'peer-c', 'peer-d']

export function DashboardHeroLoadingGhost() {
  return (
    <section className="panel-shell flex flex-wrap items-start gap-4 rounded-[var(--radius-lg)] border border-border bg-panel px-[19px] py-[15px] sm:flex-nowrap sm:items-center">
      <LoadingGhostBlock className="size-[34px] shrink-0" shimmer />
      <div className="min-w-0 flex-1">
        <div className="type-label mb-1 text-fg-faint">Live API</div>
        <div className="flex flex-wrap items-center gap-2">
          <LoadingGhostBlock className="h-4 w-40" shimmer />
        </div>
        <div className="mt-1">
          <LoadingGhostBlock className="h-3 w-[28rem] max-w-full" shimmer />
        </div>
      </div>
      <div className="flex shrink-0 basis-full items-center justify-start self-center pl-[50px] pt-1 sm:basis-auto sm:justify-end sm:pl-0 sm:pt-0">
        <div className="flex items-center gap-3">
          <LoadingGhostBlock className="h-4 w-20" shimmer />
          <span aria-hidden="true" className="text-border">
            ·
          </span>
          <LoadingGhostBlock className="h-4 w-16" shimmer />
        </div>
      </div>
    </section>
  )
}

export function DashboardStatusLoadingGhost() {
  return (
    <section className="panel-shell grid grid-cols-2 overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel sm:grid-cols-3 lg:flex">
      {STATUS_GHOST_ITEMS.map((item) => (
        <div
          key={item.id}
          className="grid min-w-0 flex-1 grid-rows-[14px_22px_18px] gap-y-1 border-r border-border-soft px-3.5 py-[12px] last:border-r-0"
          style={{ minHeight: 70 }}
        >
          <div className="flex min-w-0 items-center gap-1.5">
            <LoadingGhostBlock className="size-[11px] shrink-0" shimmer />
            <LoadingGhostBlock className={`h-2.5 ${item.labelWidth}`} shimmer />
          </div>
          <div className="flex min-w-0 items-baseline gap-1.5 overflow-hidden">
            <LoadingGhostBlock className={`h-4 ${item.valueWidth}`} shimmer />
          </div>
          <div className="flex min-w-0 items-center gap-2 overflow-hidden">
            {item.meta === 'badge' ? (
              <>
                <span aria-hidden="true" className="size-[5px] shrink-0 rounded-full bg-panel-strong" />
                <LoadingGhostBlock className={`h-2.5 ${item.metaWidth}`} shimmer />
              </>
            ) : item.meta === 'sparkline' ? (
              <LoadingGhostBlock className={`h-[2px] ${item.metaWidth} rounded-full`} shimmer />
            ) : (
              <LoadingGhostBlock className={`h-2.5 ${item.metaWidth}`} shimmer />
            )}
          </div>
        </div>
      ))}
    </section>
  )
}

export function ModelCatalogLoadingGhost() {
  return (
    <section className="panel-shell flex h-full flex-col rounded-[var(--radius-lg)] border border-border bg-panel p-3.5">
      <LoadingGhostBlock className="h-4 w-36 shrink-0" shimmer />
      <div className="mt-3 min-h-0 flex-1 space-y-2">
        {CATALOG_GHOST_ROWS.map((row) => (
          <LoadingGhostBlock key={row} className="h-14" shimmer />
        ))}
      </div>
    </section>
  )
}

export function PeersTableLoadingGhost() {
  return (
    <section className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-3.5">
      <LoadingGhostBlock className="h-4 w-28" shimmer />
      <div className="mt-3 space-y-2">
        {PEER_GHOST_ROWS.map((row) => (
          <LoadingGhostBlock key={row} className="h-10" shimmer />
        ))}
      </div>
    </section>
  )
}

export function ConnectBlockLoadingGhost() {
  return (
    <section className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-3.5">
      <LoadingGhostBlock className="h-4 w-24" shimmer />
      <div className="mt-3 grid grid-cols-2 gap-3.5">
        <LoadingGhostBlock className="h-[72px]" shimmer />
        <LoadingGhostBlock className="h-[72px]" shimmer />
      </div>
      <LoadingGhostBlock className="mt-3 h-11" shimmer />
    </section>
  )
}
