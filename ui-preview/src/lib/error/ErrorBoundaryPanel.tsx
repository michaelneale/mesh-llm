import { useId } from 'react'
import { env } from '@/lib/env'
import { formatErrorDiagnostics } from '@/lib/error/format-error-diagnostics'

type ErrorBoundaryPanelProps = {
  title: string
  description: string
  error?: Error
  componentStack?: string
  scopeLabel: string
  recoveryActionLabel?: string
}

export function ErrorBoundaryPanel({
  title,
  description,
  error,
  componentStack,
  scopeLabel,
  recoveryActionLabel = 'Reload app'
}: ErrorBoundaryPanelProps) {
  const titleId = useId()
  const errorMessage = env.isDevelopment ? error?.message : undefined
  const diagnostics = env.isDevelopment ? formatErrorDiagnostics(error, componentStack) : undefined

  return (
    <section
      aria-labelledby={titleId}
      className="panel-shell w-full max-w-6xl overflow-hidden rounded-[var(--radius-lg)] border border-[color:color-mix(in_oklab,var(--color-bad)_36%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-bad)_7%,var(--color-panel))] shadow-surface-low"
      role="alert"
    >
      <div className="grid gap-5 border-b border-border-soft px-6 py-5 md:grid-cols-[1fr_auto] md:items-start">
        <div>
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[color:color-mix(in_oklab,var(--color-bad)_42%,var(--color-border))] bg-background px-2.5 py-1 font-mono text-[length:var(--density-type-label)] font-semibold uppercase tracking-[0.12em] text-bad">
            <span
              className="size-1.5 rounded-full bg-bad shadow-[0_0_8px_color-mix(in_oklab,var(--color-bad)_70%,transparent)]"
              aria-hidden="true"
            />
            Render fault
          </div>
          <h1 id={titleId} className="type-display text-foreground">
            {title}
          </h1>
          <p className="type-body mt-2 max-w-[96ch] text-muted-foreground">{description}</p>
        </div>
        <button
          className="inline-flex h-9 items-center justify-center rounded-[var(--radius)] border border-border bg-panel-strong px-3 font-medium text-foreground hover:border-[color:color-mix(in_oklab,var(--color-accent)_42%,var(--color-border))] hover:bg-[color:color-mix(in_oklab,var(--color-accent)_7%,var(--color-panel-strong))] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          type="button"
          onClick={() => window.location.reload()}
        >
          {recoveryActionLabel}
        </button>
        {errorMessage ? (
          <div className="rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-bad)_28%,var(--color-border))] bg-background px-3 py-2.5 md:col-span-2">
            <div className="type-label text-fg-faint">Error message</div>
            <p className="mt-1 font-mono text-[length:var(--density-type-caption-lg)] leading-relaxed text-foreground">
              {errorMessage}
            </p>
          </div>
        ) : null}
      </div>

      <div className="grid gap-4 px-6 py-5">
        <div className="grid gap-3 md:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
          <div className="rounded-[var(--radius)] border border-border-soft bg-panel-strong px-3 py-2.5">
            <div className="type-label text-fg-faint">Boundary scope</div>
            <div className="mt-1 font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
              {scopeLabel}
            </div>
          </div>
          <div className="rounded-[var(--radius)] border border-border-soft bg-panel-strong px-3 py-2.5">
            <div className="type-label text-fg-faint">Suggested recovery</div>
            <div className="mt-1 text-[length:var(--density-type-caption-lg)] text-foreground">
              Refresh the page, then retry the action. If it repeats, capture the diagnostics below.
            </div>
          </div>
        </div>

        {diagnostics ? (
          <div className="overflow-hidden rounded-[var(--radius)] border border-border bg-background">
            <div className="flex items-center justify-between gap-3 border-b border-border-soft bg-panel-strong px-3 py-2">
              <div>
                <div className="type-label text-fg-faint">Development diagnostics</div>
                <div className="mt-0.5 text-[length:var(--density-type-caption)] text-muted-foreground">
                  Stack trace shown when the runtime exposes it.
                </div>
              </div>
              <span className="shrink-0 rounded-full border border-border-soft bg-background px-2 py-0.5 font-mono text-[length:var(--density-type-label)] uppercase tracking-[0.08em] text-fg-faint">
                Stack trace
              </span>
            </div>
            <pre className="max-h-[42vh] min-h-44 overflow-auto whitespace-pre-wrap p-4 font-mono text-[length:var(--density-type-label)] leading-relaxed text-foreground">
              {diagnostics}
            </pre>
          </div>
        ) : null}
      </div>
    </section>
  )
}
