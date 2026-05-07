import { LoadingGhostBlock } from '@/components/ui/LoadingGhostBlock'

const RAIL_GHOST_ROWS = ['node-a', 'node-b', 'node-c', 'node-d']
const NODE_CARD_GHOST_ROWS = ['gpu-a', 'gpu-b', 'gpu-c']
const SETTINGS_GHOST_ROWS = ['setting-a', 'setting-b', 'setting-c']

export function ConfigurationHeaderLoadingGhost() {
  return (
    <header className="sticky top-0 z-20 bg-transparent">
      <div className="flex min-h-[76px] flex-wrap items-center justify-between gap-3 px-5 py-3">
        <div className="min-w-0">
          <div className="type-label text-fg-faint">Live API</div>
          <LoadingGhostBlock className="mt-2 h-6 w-56" shimmer />
          <LoadingGhostBlock className="mt-2 h-3 w-80" shimmer />
        </div>
        <div className="flex items-center gap-1.5">
          <LoadingGhostBlock className="h-[30px] w-[30px]" shimmer />
          <LoadingGhostBlock className="h-[30px] w-[30px]" shimmer />
          <LoadingGhostBlock className="h-[30px] w-24" shimmer />
          <LoadingGhostBlock className="h-[30px] w-28" shimmer />
        </div>
      </div>
    </header>
  )
}

export function ConfigurationRailLoadingGhost() {
  return (
    <aside className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-3">
      <LoadingGhostBlock className="h-4 w-28" shimmer />
      <div className="mt-3 space-y-2">
        {RAIL_GHOST_ROWS.map((row) => (
          <LoadingGhostBlock key={row} className="h-12" shimmer />
        ))}
      </div>
    </aside>
  )
}

export function ConfigurationDeploymentLoadingGhost() {
  return (
    <div className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-4">
      <div className="flex items-center justify-between">
        <LoadingGhostBlock className="h-5 w-44" shimmer />
        <LoadingGhostBlock className="h-7 w-32 rounded-full" shimmer />
      </div>
      <div className="mt-4 grid gap-3">
        {NODE_CARD_GHOST_ROWS.map((card) => (
          <div key={card} className="rounded-[var(--radius)] border border-border-soft bg-background p-3">
            <div className="flex items-center justify-between">
              <LoadingGhostBlock className="h-4 w-36" shimmer />
              <LoadingGhostBlock className="h-4 w-20" shimmer />
            </div>
            <LoadingGhostBlock className="mt-3 h-2 w-full rounded-full" shimmer />
            <LoadingGhostBlock className="mt-2 h-2 w-2/3 rounded-full" shimmer />
          </div>
        ))}
      </div>
    </div>
  )
}

export function ConfigurationSettingsLoadingGhost() {
  return (
    <div className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-4">
      <LoadingGhostBlock className="h-4 w-40" shimmer />
      <div className="mt-3 space-y-2">
        {SETTINGS_GHOST_ROWS.map((row) => (
          <LoadingGhostBlock key={row} className="h-10" shimmer />
        ))}
      </div>
    </div>
  )
}
