import { MessageSquare, Send } from 'lucide-react'
import { InboundTransparency } from '@/features/chat/components/transparency/InboundTransparency'
import { OutboundTransparency } from '@/features/chat/components/transparency/OutboundTransparency'
import type { TransparencyMessage, TransparencyNode } from '@/features/app-tabs/types'

function MeshIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width={11}
      height={11}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="5" cy="6" r="2.2" />
      <circle cx="19" cy="6" r="2.2" />
      <circle cx="12" cy="18" r="2.2" />
      <path d="M6.8 7.3L10.7 16.3M17.2 7.3L13.3 16.3M7 6h10" />
    </svg>
  )
}

export function TransparencyPane({ message, nodes }: { message?: TransparencyMessage; nodes: TransparencyNode[] }) {
  if (!message) {
    return (
      <div className="flex min-h-[132px] flex-col items-center justify-center px-6 py-9 text-center text-[length:var(--density-type-control)] leading-[1.55] text-fg-faint">
        <div
          className="mb-3 inline-flex size-8 items-center justify-center rounded-[var(--radius)]"
          style={{
            background: 'color-mix(in oklab, var(--color-accent) 12%, transparent)',
            border: '1px solid color-mix(in oklab, var(--color-accent) 22%, var(--color-border-soft))',
            color: 'var(--color-accent)'
          }}
        >
          <MessageSquare className="size-[15px]" strokeWidth={1.7} aria-hidden="true" />
        </div>
        <div className="mb-1.5 text-[length:var(--density-type-body)] font-medium text-fg-dim">No message selected</div>
        <p className="max-w-[18rem] text-pretty">
          Click any message on the right to see how it was routed through the mesh.
        </p>
      </div>
    )
  }
  const isUser = message.kind === 'user'
  return (
    <div className="flex flex-col gap-3">
      <div
        className="flex items-center gap-2 rounded-[var(--radius)] px-3 py-[9px]"
        style={{
          background: 'color-mix(in oklab, var(--color-accent) 10%, transparent)',
          border: '1px solid color-mix(in oklab, var(--color-accent) 25%, var(--color-border-soft))'
        }}
      >
        <span
          className="inline-flex size-[18px] items-center justify-center rounded-[var(--radius)]"
          style={{
            background: isUser
              ? 'var(--color-panel-strong)'
              : 'color-mix(in oklab, var(--color-accent) 30%, transparent)',
            color: isUser ? 'var(--color-fg-dim)' : 'var(--color-accent)'
          }}
        >
          {isUser ? <Send className="size-[11px]" /> : <MeshIcon />}
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-[length:var(--density-type-caption-lg)] font-medium">
            {isUser ? 'Your message' : 'Assistant reply'}{' '}
            <span className="font-normal text-fg-faint">· {message.at}</span>
          </div>
          <div className="truncate text-[length:var(--density-type-label)] text-fg-dim">{message.text}</div>
        </div>
      </div>
      {message.kind === 'assistant' ? (
        <InboundTransparency message={message} nodes={nodes} />
      ) : (
        <OutboundTransparency message={message} nodes={nodes} />
      )}
    </div>
  )
}
