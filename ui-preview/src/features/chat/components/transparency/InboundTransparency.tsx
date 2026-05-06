import { Hash, Copy, Server } from 'lucide-react'
import { DecisionRow } from '@/features/chat/components/transparency/DecisionRow'
import { RouteMiniMap } from '@/features/chat/components/transparency/RouteMiniMap'
import { SectionLabel } from '@/features/chat/components/transparency/SectionLabel'
import { TraceBar } from '@/features/chat/components/transparency/TraceBar'
import type { InboundTransparencyMessage, TransparencyNode } from '@/features/app-tabs/types'

function Metric({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="rounded-[var(--radius)] border border-border-soft bg-panel-strong px-2.5 py-[9px]">
      <div className="text-[length:var(--density-type-micro)] uppercase tracking-[0.05em] text-fg-faint">{label}</div>
      <div className="mt-0.5 flex items-baseline gap-1">
        <span className="font-mono text-[length:var(--density-type-headline)] font-medium">{value}</span>
        {unit && <span className="text-[length:var(--density-type-annotation)] text-fg-faint">{unit}</span>}
      </div>
    </div>
  )
}

export function InboundTransparency({
  message,
  nodes
}: {
  message: InboundTransparencyMessage
  nodes: TransparencyNode[]
}) {
  const servedNode = nodes.find((node) => node.id === message.servedBy)
  const isLocal = servedNode?.id === nodes[0]?.id
  const totalMs = message.trace.reduce((sum, seg) => sum + seg.ms, 0)
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-[var(--radius-lg)] border border-border-soft bg-panel-strong p-3.5 pb-3">
        <SectionLabel>Inbound route</SectionLabel>
        <RouteMiniMap nodes={nodes} pickId={message.servedBy} direction="in" />
        <div className="mt-3 flex items-center gap-2.5 border-t border-border-soft pt-3">
          <span
            className="inline-flex size-7 items-center justify-center rounded-[var(--radius)] text-accent"
            style={{ background: 'color-mix(in oklab, var(--color-accent) 25%, transparent)' }}
          >
            <Server className="size-[13px]" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="font-mono text-[length:var(--density-type-control-lg)] font-medium">
              {servedNode?.label ?? message.servedBy}
            </div>
            <div className="truncate font-mono text-[length:var(--density-type-label)] text-fg-faint">
              {message.servedBy}
            </div>
          </div>
          <span
            className="inline-flex items-center rounded-full px-2 py-px text-[length:var(--density-type-label)] font-medium"
            style={{
              background: isLocal
                ? 'color-mix(in oklab, var(--color-accent) 16%, var(--color-background))'
                : 'color-mix(in oklab, var(--color-good) 18%, var(--color-background))',
              color: isLocal ? 'var(--color-accent)' : 'var(--color-good)',
              border: isLocal
                ? '1px solid color-mix(in oklab, var(--color-accent) 28%, var(--color-background))'
                : '1px solid color-mix(in oklab, var(--color-good) 30%, var(--color-background))'
            }}
          >
            {isLocal ? 'local' : 'peer'}
          </span>
        </div>
      </div>

      <div>
        <SectionLabel>Metrics</SectionLabel>
        <div className="grid grid-cols-2 gap-2">
          <Metric label="RTT" value={`${message.metrics.rttMs}`} unit="ms" />
          <Metric label="TTFT" value={`${message.metrics.ttftMs}`} unit="ms" />
          <Metric label="Throughput" value={message.metrics.throughput} unit="tok/s" />
          <Metric label="Tokens" value={`${message.metrics.tokens}`} />
        </div>
      </div>

      <div>
        <SectionLabel>Why this node</SectionLabel>
        <div className="rounded-[var(--radius)] border border-border-soft bg-background px-3 py-1">
          {message.decisions.map((decision) => (
            <DecisionRow key={decision.id} ok={decision.ok} label={decision.label} detail={decision.detail} />
          ))}
        </div>
      </div>

      <div>
        <SectionLabel right={`total ${totalMs} ms`}>Trace</SectionLabel>
        <TraceBar segs={message.trace} />
      </div>

      <div className="flex items-center justify-between rounded-[var(--radius)] border border-dashed border-border bg-background px-3 py-2.5">
        <div className="flex items-center gap-2">
          <Hash className="size-[11px] text-fg-dim" />
          <span className="text-[length:var(--density-type-caption-lg)] text-fg-dim">Signed routing receipt</span>
        </div>
        <button
          type="button"
          className="inline-flex items-center gap-[5px] text-[length:var(--density-type-caption)] font-medium text-accent"
        >
          <Copy className="size-[11px]" />
          Verify
        </button>
      </div>
    </div>
  )
}
