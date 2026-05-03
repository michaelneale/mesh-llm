import type { ReactNode } from 'react'
import { env } from '@/lib/env'

type LiveDataUnavailableOverlayProps = {
  children: ReactNode
  debugDescription: string
  productionDescription: string
  title: string
  debugTitle?: string
  onRetry: () => void
  onSwitchToTestData?: () => void
}

export function LiveDataUnavailableOverlay({ children, debugDescription, debugTitle, onRetry, onSwitchToTestData, productionDescription, title }: LiveDataUnavailableOverlayProps) {
  const isDebug = env.isDevelopment

  return (
    <div className="relative isolate">
      <div aria-hidden="true" className="pointer-events-none select-none opacity-55 blur-[2.5px] saturate-75">
        {children}
      </div>
      <div className="absolute inset-0 z-10 grid min-h-[min(74vh,760px)] place-items-center bg-[color:color-mix(in_oklab,var(--color-background)_34%,transparent)] px-4 py-8">
        <section
          className="panel-shell w-full max-w-[34rem] rounded-[var(--radius-lg)] border border-[color:color-mix(in_oklab,var(--color-bad)_42%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-bad)_10%,var(--color-panel))] px-5 py-4.5 shadow-[0_24px_80px_color-mix(in_oklab,var(--color-background)_72%,transparent)]"
          role="alert"
        >
          <div className="inline-flex items-center gap-2 rounded-full border border-[color:color-mix(in_oklab,var(--color-bad)_38%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-bad)_12%,transparent)] px-2 py-1 type-label text-bad">
            <span aria-hidden="true" className="size-1.5 rounded-full bg-bad shadow-[0_0_10px_color-mix(in_oklab,var(--color-bad)_42%,transparent)]" />
            Live API unavailable
          </div>
          <h2 className="mt-3 type-panel-title text-foreground">{isDebug && debugTitle ? debugTitle : title}</h2>
          <p className="mt-2 text-[length:var(--density-type-caption-lg)] leading-relaxed text-fg-dim">
            {isDebug ? debugDescription : productionDescription}
          </p>
          {isDebug ? (
            <div className="mt-4 rounded-[var(--radius)] border border-border-soft bg-background px-3 py-2 font-mono text-[length:var(--density-type-caption)] text-fg-faint">
              API target: <span className="text-foreground">{env.apiUrl}</span>
            </div>
          ) : null}
          <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
            {isDebug && onSwitchToTestData ? (
              <button
                className="ui-control inline-flex h-8 items-center justify-center rounded-[var(--radius)] border border-border bg-panel-strong px-3 text-[length:var(--density-type-control)] font-medium leading-none text-foreground outline-none hover:border-[color:color-mix(in_oklab,var(--color-accent)_38%,var(--color-border))] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                type="button"
                onClick={onSwitchToTestData}
              >
                Switch to test data
              </button>
            ) : <span aria-hidden="true" />}
            <button
              className="ui-control-primary inline-flex h-8 items-center justify-center rounded-[var(--radius)] px-3 text-[length:var(--density-type-control)] font-semibold leading-none outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              type="button"
              onClick={onRetry}
            >
              Retry live data
            </button>
          </div>
        </section>
      </div>
    </div>
  )
}
