import { cn } from '@/lib/cn'
import { formatLatency } from '@/lib/format-latency'
import type { Peer, PeerSummary } from '@/features/app-tabs/types'

type PeersTableProps = { peers: Peer[]; summary: PeerSummary; selectedPeerId?: string; onSelect?: (peer: Peer) => void }

const gridCols = '1.4fr 0.7fr 0.7fr 0.8fr 1.8fr 0.6fr 0.7fr 0.6fr 0.6fr'

function RolePill({ role }: { role: NonNullable<Peer['role']> }) {
  const isYou = role === 'you'
  return (
    <span
      className="inline-flex items-center rounded-full px-2 py-px text-[length:var(--density-type-caption)] font-medium"
      style={{
        background: isYou ? 'color-mix(in oklab, var(--color-accent) 16%, var(--color-background))' : 'transparent',
        color: isYou ? 'var(--color-accent)' : 'var(--color-fg-faint)',
        border: isYou ? '1px solid color-mix(in oklab, var(--color-accent) 28%, var(--color-background))' : '1px solid var(--color-border)',
      }}
    >
      {role === 'you' ? 'You' : role === 'host' ? 'Host' : 'Peer'}
    </span>
  )
}

function StatusPill() {
  return (
    <span
      className="inline-flex items-center gap-[5px] rounded-full px-2 py-px text-[length:var(--density-type-caption)] font-medium"
      style={{
        background: 'color-mix(in oklab, var(--color-good) 18%, var(--color-background))',
        color: 'var(--color-good)',
        border: '1px solid color-mix(in oklab, var(--color-good) 30%, var(--color-background))',
      }}
    >
      <span className="size-[5px] rounded-full bg-good" />
      Serving
    </span>
  )
}

function ShareMeter({ sharePct, compact = false }: { sharePct: number; compact?: boolean }) {
  if (compact) {
    return (
      <div className="flex min-w-0 items-center justify-end gap-1.5">
        <span className="w-7 shrink-0 text-right font-mono text-[length:var(--density-type-caption)] text-fg-dim">{sharePct}%</span>
        <div
          className="h-[3px] w-14 shrink-0 overflow-hidden rounded-[3px] sm:w-20"
          style={{ background: 'color-mix(in oklab, var(--color-accent) 15%, transparent)' }}
        >
          <div className="h-full rounded-[3px] bg-accent" style={{ width: `${sharePct}%` }} />
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-1.5">
      <div
        className="h-[3px] shrink-0 flex-1 overflow-hidden rounded-[3px]"
        style={{ background: 'color-mix(in oklab, var(--color-accent) 15%, transparent)' }}
      >
        <div className="h-full rounded-[3px] bg-accent" style={{ width: `${sharePct}%` }} />
      </div>
      <span className="w-7 shrink-0 text-right font-mono text-[length:var(--density-type-caption)] text-fg-dim">{sharePct}%</span>
    </div>
  )
}

type PeerRowProps = { peer: Peer; active: boolean; isLast: boolean; onSelect?: (peer: Peer) => void }

function PeerRow({ peer, active, isLast, onSelect }: PeerRowProps) {
  const hostedModels = peer.hostedModels.join(' ')

  return (
    <button
      aria-label={`View ${peer.hostname} node${active ? ' (selected)' : ''}`}
      data-active={active ? 'true' : undefined}
      onClick={() => onSelect?.(peer)}
      type="button"
      className={cn(
        'ui-row-action w-full min-w-0 px-3.5 py-[11px] text-left lg:grid lg:min-w-[760px] lg:items-center lg:gap-0',
        !isLast && 'border-b border-border-soft',
        active ? 'bg-[color-mix(in_oklab,var(--color-accent)_10%,var(--color-panel))]' : 'bg-transparent',
      )}
      style={{ gridTemplateColumns: gridCols }}
    >
      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-2 lg:contents">
        <div className="flex min-w-0 items-center gap-2 lg:contents">
          <div className="flex min-w-0 items-center gap-2">
            <span className="shrink-0 font-mono text-[length:var(--density-type-control)]">{peer.shortId ?? peer.id}</span>
            <span className="text-[length:var(--density-type-caption)] text-fg-faint">·</span>
            <span className="min-w-0 truncate font-mono text-[length:var(--density-type-caption)] text-fg-dim">{peer.hostname}</span>
          </div>
          <div className="hidden lg:block">{peer.role && <RolePill role={peer.role} />}</div>
          <div className="hidden font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim lg:block">{peer.version ?? '—'}</div>
          <div className="hidden lg:block"><StatusPill /></div>
          <div className="hidden flex-wrap gap-1.5 font-mono text-[length:var(--density-type-caption)] text-fg-dim lg:flex">
            {peer.hostedModels.map((m) => <span key={m}>{m}</span>)}
          </div>
        </div>
        <div className="flex items-center justify-end gap-1.5 lg:hidden">
          {peer.role && <RolePill role={peer.role} />}
          <StatusPill />
        </div>
        <div className="col-span-2 hidden min-w-0 truncate font-mono text-[length:var(--density-type-caption)] text-fg-dim min-[401px]:block lg:hidden">
          {hostedModels || '—'}
        </div>
        <div className="col-span-2 grid min-w-0 grid-cols-[auto_auto_auto_minmax(92px,1fr)] items-center gap-x-3 gap-y-1 lg:hidden">
          <div className="font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim">{formatLatency(peer.latencyMs)} ms</div>
          <div className="font-mono text-[length:var(--density-type-caption-lg)]">{peer.vramGB?.toFixed(1) ?? '—'} GB</div>
          <div className="flex items-baseline justify-end gap-1 font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim">
            <span className="text-[length:var(--density-type-annotation)] uppercase tracking-[0.04em] text-fg-faint">Tok/s</span>
            <span>{peer.toksPerSec?.toFixed(1) ?? '—'}</span>
          </div>
          <ShareMeter sharePct={peer.sharePct} compact />
        </div>
        <div className="hidden lg:contents">
          <div className="font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim">{formatLatency(peer.latencyMs)} ms</div>
          <div className="font-mono text-[length:var(--density-type-caption-lg)]">{peer.vramGB?.toFixed(1) ?? '—'} GB</div>
          <ShareMeter sharePct={peer.sharePct} />
          <div className="text-right font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim">{peer.toksPerSec?.toFixed(1) ?? '—'}</div>
        </div>
      </div>
    </button>
  )
}

export function PeersTable({ peers, summary, selectedPeerId, onSelect }: PeersTableProps) {
  return (
    <section className="panel-shell min-w-0 overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel [contain:inline-size]">
      <header className="flex items-center justify-between border-b border-border-soft px-3.5 py-2.5">
        <h2 className="type-panel-title">Connected peers</h2>
        <span className="type-caption text-fg-faint">
          {summary.total} total · <span className="text-good">● {summary.capacity}</span>
        </span>
      </header>
      <div className="max-w-full overflow-x-auto [contain:inline-size]">
        <div
          className="type-label hidden min-w-[760px] border-b border-border-soft px-3.5 py-2 text-fg-faint lg:grid"
          style={{ gridTemplateColumns: gridCols }}
        >
          <div>ID</div><div>Role</div><div>Version</div><div>Status</div><div>Hosted</div><div>Latency</div><div>VRAM</div><div>Share</div><div className="text-right">Tok/s</div>
        </div>
        {peers.map((peer, i) => (
          <PeerRow
            key={peer.id}
            peer={peer}
            active={peer.id === selectedPeerId}
            isLast={i === peers.length - 1}
            onSelect={onSelect}
          />
        ))}
      </div>
    </section>
  )
}
