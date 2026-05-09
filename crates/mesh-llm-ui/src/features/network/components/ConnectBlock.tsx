import { Copy } from 'lucide-react'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import { copyStateLabel } from '@/lib/copyStateLabel'
import type { ClipboardCopyState } from '@/lib/useClipboardCopy'

export type ConnectApiTargetLiveness = 'configured' | 'checking' | 'live' | 'unavailable'

type ConnectBlockProps = {
  installHref: string
  apiUrl: string
  apiStatus: string
  apiTargetLiveness?: ConnectApiTargetLiveness
  runCommand: string
  description: string
  onCopy?: () => void
  copyState?: ClipboardCopyState
}

function displayInstallHref(installHref: string) {
  return installHref.replace(/^https?:\/\//, '').replace(/\/$/, '')
}

function apiTargetBadgeTone(liveness: ConnectApiTargetLiveness): StatusBadgeTone {
  if (liveness === 'configured') return 'warn'
  if (liveness === 'live') return 'good'
  if (liveness === 'unavailable') return 'bad'
  return 'muted'
}

export function ConnectBlock({
  installHref,
  apiUrl,
  apiStatus,
  apiTargetLiveness = 'checking',
  runCommand,
  description,
  onCopy,
  copyState
}: ConnectBlockProps) {
  const label = copyStateLabel(copyState)
  const installDisplay = displayInstallHref(installHref)
  const apiTone = apiTargetBadgeTone(apiTargetLiveness)

  return (
    <section className="panel-shell overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border-soft px-3.5 py-2.5">
        <div className="flex min-w-0 flex-wrap items-baseline gap-2">
          <h2 className="type-panel-title">Connect</h2>
          <span className="type-caption text-fg-faint">· {description}</span>
        </div>
        <a href={installHref} className="ui-link text-[length:var(--density-type-caption-lg)]">
          Full docs →
        </a>
      </header>
      <div className="p-3.5">
        <div className="grid gap-3.5 md:grid-cols-2">
          <div>
            <div className="type-label mb-1.5 text-fg-faint">1 · Install</div>
            <div className="flex min-w-0 items-center justify-between gap-2 rounded-[var(--radius)] border border-border bg-panel-strong px-3 py-[9px]">
              <span className="min-w-0 truncate font-mono text-[length:var(--density-type-control)]">
                {installDisplay}
              </span>
              <a href={installHref} className="ui-link text-[length:var(--density-type-caption-lg)]">
                open
              </a>
            </div>
          </div>
          <div>
            <div className="type-label mb-1.5 text-fg-faint">API endpoint</div>
            <div className="flex min-w-0 flex-wrap items-center justify-between gap-2 rounded-[var(--radius)] border border-border bg-panel-strong px-3 py-[9px]">
              <span className="min-w-0 truncate font-mono text-[length:var(--density-type-control)]">{apiUrl}</span>
              <StatusBadge dot tone={apiTone}>
                {apiStatus}
              </StatusBadge>
            </div>
          </div>
        </div>
        <div className="mt-3.5">
          <div className="type-label mb-1.5 text-fg-faint">2 · Run</div>
          <div className="flex items-center gap-2 overflow-hidden rounded-[var(--radius)] border border-border bg-panel-strong px-3 py-2.5">
            <span className="font-mono text-[length:var(--density-type-control)] text-accent">$</span>
            <span className="flex-1 truncate font-mono text-[length:var(--density-type-control)]">{runCommand}</span>
            <button
              onClick={onCopy}
              type="button"
              disabled={!onCopy}
              className="ui-control inline-flex items-center gap-1.5 rounded-[var(--radius)] border px-[9px] py-1 font-sans text-[length:var(--density-type-caption)]"
            >
              <Copy className="size-[11px]" />
              {label}
            </button>
          </div>
          <p className="type-caption mt-2 text-fg-faint">
            Previews the command that will auto-select a model, join the mesh, and serve an OpenAI-compatible API at{' '}
            <span className="font-mono text-fg-dim">{apiUrl}</span> once the system backend is connected.
          </p>
        </div>
      </div>
    </section>
  )
}
