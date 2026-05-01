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

type PeerRowProps = { peer: Peer; active: boolean; isLast: boolean; onSelect?: (peer: Peer) => void }

function PeerRow({ peer, active, isLast, onSelect }: PeerRowProps) {
  return (
    <button
      aria-label={`View ${peer.hostname} node${active ? ' (selected)' : ''}`}
      data-active={active ? 'true' : undefined}
      onClick={() => onSelect?.(peer)}
      type="button"
      className={cn(
        'ui-row-action grid w-full items-center px-3.5 py-[11px] text-left',
        !isLast && 'border-b border-border-soft',
        active ? 'bg-[color-mix(in_oklab,var(--color-accent)_10%,var(--color-panel))]' : 'bg-transparent',
      )}
      style={{ gridTemplateColumns: gridCols }}
    >
      <div className="flex items-center gap-2">
        <span className="font-mono text-[length:var(--density-type-control)]">{peer.shortId ?? peer.id}</span>
        <span className="text-[length:var(--density-type-caption)] text-fg-faint">·</span>
        <span className="font-mono text-[length:var(--density-type-caption)] text-fg-dim">{peer.hostname}</span>
      </div>
      <div>{peer.role && <RolePill role={peer.role} />}</div>
      <div className="font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim">{peer.version ?? '—'}</div>
      <div><StatusPill /></div>
      <div className="flex flex-wrap gap-1.5 font-mono text-[length:var(--density-type-caption)] text-fg-dim">
        {peer.hostedModels.map((m) => <span key={m}>{m}</span>)}
      </div>
      <div className="font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim">{formatLatency(peer.latencyMs)} ms</div>
      <div className="font-mono text-[length:var(--density-type-caption-lg)]">{peer.vramGB?.toFixed(1) ?? '—'} GB</div>
      <div className="flex items-center gap-1.5">
        <div className="h-[3px] flex-1 overflow-hidden rounded-[3px]" style={{ background: 'color-mix(in oklab, var(--color-accent) 15%, transparent)' }}>
          <div className="h-full rounded-[3px] bg-accent" style={{ width: `${peer.sharePct}%` }} />
        </div>
        <span className="w-7 text-right font-mono text-[length:var(--density-type-caption)] text-fg-dim">{peer.sharePct}%</span>
      </div>
      <div className="text-right font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim">{peer.toksPerSec?.toFixed(1) ?? '—'}</div>
    </button>
  )
}

export function PeersTable({ peers, summary, selectedPeerId, onSelect }: PeersTableProps) {
  return (
    <section className="panel-shell overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
      <header className="flex items-center justify-between border-b border-border-soft px-3.5 py-2.5">
        <h2 className="type-panel-title">Connected peers</h2>
        <span className="type-caption text-fg-faint">
          {summary.total} total · <span className="text-good">● {summary.capacity}</span>
        </span>
      </header>
      <div>
        <div
          className="type-label grid border-b border-border-soft px-3.5 py-2 text-fg-faint"
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
