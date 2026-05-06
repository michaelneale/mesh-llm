import { cn } from '@/lib/cn'

type LoadingGhostBlockProps = {
  className?: string
  panelShell?: boolean
  shimmer?: boolean
}

export function LoadingGhostBlock({ className, panelShell = false, shimmer = false }: LoadingGhostBlockProps) {
  return (
    <div
      aria-hidden="true"
      className={cn(
        'rounded-[var(--radius)] border',
        shimmer && 'relative overflow-hidden',
        panelShell ? 'panel-shell border-border bg-panel' : 'border-border-soft bg-panel-strong',
        className
      )}
    >
      {shimmer ? (
        <span
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-[-1px] left-0 w-1/2 -translate-x-[130%] bg-[linear-gradient(90deg,transparent,color-mix(in_oklab,var(--color-accent)_7%,transparent),transparent)] opacity-0"
          data-loading-ghost-shimmer
        />
      ) : null}
    </div>
  )
}
