import { Hash, Copy, Send } from 'lucide-react'
import { DecisionRow } from '@/features/chat/components/transparency/DecisionRow'
import { RouteMiniMap } from '@/features/chat/components/transparency/RouteMiniMap'
import { SectionLabel } from '@/features/chat/components/transparency/SectionLabel'
import type { OutboundTransparencyMessage, TransparencyNode } from '@/features/app-tabs/types'

function Metric({ label, value, unit, small }: { label: string; value: string; unit?: string; small?: boolean }) {
  return (
    <div className="rounded-[var(--radius)] border border-border-soft bg-panel-strong px-2.5 py-[9px]">
      <div className="text-[length:var(--density-type-micro)] uppercase tracking-[0.05em] text-fg-faint">{label}</div>
      <div className="mt-0.5 flex items-baseline gap-1 overflow-hidden">
        <span className={`truncate font-mono font-medium ${small ? 'text-[length:var(--density-type-caption)]' : 'text-[length:var(--density-type-headline)]'}`}>{value}</span>
        {unit && <span className="text-[length:var(--density-type-annotation)] text-fg-faint">{unit}</span>}
      </div>
    </div>
  )
}

export function OutboundTransparency({ message, nodes }: { message: OutboundTransparencyMessage; nodes: TransparencyNode[] }) {
  const picked = nodes.find((node) => node.id === message.dispatch.picked)
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-[var(--radius-lg)] border border-border-soft bg-panel-strong p-3.5 pb-3">
        <SectionLabel>Outbound dispatch</SectionLabel>
        <RouteMiniMap nodes={nodes} pickId={message.dispatch.picked} direction="out" />
        <div className="mt-3 flex items-center gap-2.5 border-t border-border-soft pt-3">
          <span
            className="inline-flex size-7 items-center justify-center rounded-[var(--radius)] text-accent"
            style={{ background: 'color-mix(in oklab, var(--color-accent) 25%, transparent)' }}
          >
            <Send className="size-[13px]" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="font-mono text-[length:var(--density-type-control-lg)] font-medium">→ {picked?.label ?? message.dispatch.picked}</div>
            <div className="truncate font-mono text-[length:var(--density-type-label)] text-fg-faint">{message.dispatch.picked}</div>
          </div>
          <span
            className="inline-flex items-center rounded-full px-2 py-px text-[length:var(--density-type-label)] font-medium"
            style={{
              background: 'color-mix(in oklab, var(--color-accent) 16%, var(--color-background))',
              color: 'var(--color-accent)',
              border: '1px solid color-mix(in oklab, var(--color-accent) 28%, var(--color-background))',
            }}
          >
            dispatched
          </span>
        </div>
      </div>

      <div>
        <SectionLabel>Payload</SectionLabel>
        <div className="grid grid-cols-2 gap-2">
          <Metric label="Bytes" value={`${message.dispatch.bytes}`} unit="B" />
          <Metric label="Tokens" value={`${message.dispatch.tokens}`} />
          <Metric label="Candidates" value={`${message.dispatch.candidates}`} unit="nodes" />
          <Metric label="Model" value={message.dispatch.model} small />
        </div>
      </div>

      <div>
        <SectionLabel>Security</SectionLabel>
        <div className="rounded-[var(--radius)] border border-border-soft bg-background px-3 py-1">
          {message.security.map((decision) => <DecisionRow key={decision.id} ok={decision.ok} label={decision.label} detail={decision.detail} />)}
        </div>
      </div>

      <div className="flex items-center justify-between rounded-[var(--radius)] border border-dashed border-border bg-background px-3 py-2.5">
        <div className="flex items-center gap-2">
          <Hash className="size-[11px] text-fg-dim" />
          <span className="text-[length:var(--density-type-caption-lg)] text-fg-dim">Request ID · {message.requestId}</span>
        </div>
        <button type="button" className="inline-flex items-center gap-[5px] text-[length:var(--density-type-caption)] font-medium text-accent">
          <Copy className="size-[11px]" />Copy
        </button>
      </div>
    </div>
  )
}
