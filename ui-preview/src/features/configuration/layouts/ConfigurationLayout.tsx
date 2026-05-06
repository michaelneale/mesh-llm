import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'

type ConfigurationLayoutProps = {
  header: ReactNode
  children: ReactNode
}

export function ConfigurationLayout({ header, children }: ConfigurationLayoutProps) {
  return (
    <div className="-mt-[18px] md:-mx-5">
      {header}
      {children}
    </div>
  )
}

export function ConfigurationDeploymentLayout({ rail, children }: { rail: ReactNode; children: ReactNode }) {
  return (
    <div className="grid min-w-0 gap-3.5 lg:grid-cols-[220px_minmax(0,1fr)]">
      {rail}
      <div className="min-w-0 space-y-3">{children}</div>
    </div>
  )
}

export function ConfigurationPlaceholderPanel({
  title,
  icon: Icon,
  children
}: {
  title: string
  icon: LucideIcon
  children: ReactNode
}) {
  const headingId = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-heading`

  return (
    <section
      className="panel-shell grid min-h-[220px] place-items-center rounded-[var(--radius-lg)] border border-border bg-panel px-6 py-10 text-center shadow-surface-panel"
      aria-labelledby={headingId}
      data-panel-soft-elevation="none"
    >
      <div className="flex max-w-[46rem] flex-col items-center">
        <div
          className="grid size-12 place-items-center rounded-full border border-border bg-panel-strong text-accent shadow-[0_0_18px_color-mix(in_oklab,var(--color-accent)_10%,transparent)]"
          aria-hidden="true"
        >
          <Icon className="size-5" strokeWidth={1.8} />
        </div>
        <p className="mt-4 type-label text-fg-faint">Planned section</p>
        <h2 id={headingId} className="mt-1 text-[length:var(--density-type-title)] font-semibold text-foreground">
          {title}
        </h2>
        <p className="mt-3 max-w-[58ch] text-[length:var(--density-type-control-lg)] leading-relaxed text-fg-dim">
          {children}
        </p>
      </div>
    </section>
  )
}
