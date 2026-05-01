import type { CSSProperties } from 'react'
import { Send, User } from 'lucide-react'
import type { MessageRole } from '@/features/app-tabs/types'
import { cn } from '@/lib/cn'

function MeshIcon() {
  return (
    <svg viewBox="0 0 24 24" width={10} height={10} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="5" cy="6" r="2.2" /><circle cx="19" cy="6" r="2.2" /><circle cx="12" cy="18" r="2.2" />
      <path d="M6.8 7.3L10.7 16.3M17.2 7.3L13.3 16.3M7 6h10" />
    </svg>
  )
}

type MessageRowProps = { messageRole: MessageRole; body: string; timestamp: string; model?: string; route?: string; tokens?: string; inspect?: () => void; inspectLabel?: string; inspected?: boolean; routeNode?: string; showRouteMetadata?: boolean; tokPerSec?: string; ttft?: string }

export function MessageRow({ messageRole, body, timestamp, model, route, tokens, inspect, inspectLabel, inspected, routeNode, showRouteMetadata = true, tokPerSec, ttft }: MessageRowProps) {
  const isUser = messageRole === 'user'
  const isResponse = !isUser
  const canInspect = inspect != null
  const routeMetadata = showRouteMetadata && ((isUser && routeNode) || (isResponse && route))
  const rowClassName = cn(
    'relative -mx-2 mb-5 block w-[calc(100%+16px)] select-none rounded-[var(--radius-lg)] border-0 bg-transparent px-2 py-1 text-left transition-[background,box-shadow] duration-150',
    canInspect && 'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
  )
  const rowStyle: CSSProperties = {
    ...(canInspect ? { cursor: 'pointer' } : {}),
    ...(inspected ? { background: 'color-mix(in oklab, var(--color-accent) 4%, transparent)' } : {}),
  }
  const userContentStyle: CSSProperties = {
    background: 'var(--chat-user-message-background)',
    borderLeft: '1px solid var(--color-accent)',
    padding: '8px 12px 8px 14px',
  }
  const accessibleInspectLabel = inspectLabel ?? `Inspect ${isUser ? 'user' : 'assistant'} message from ${timestamp}`
  const messageContent = (
    <>
      <span className="mb-1.5 flex select-none items-center gap-2 text-[length:var(--density-type-caption)] text-fg-faint">
        <span
          className="inline-flex size-4 items-center justify-center rounded-[var(--radius)]"
          style={{
            background: isUser ? 'var(--color-panel-strong)' : 'color-mix(in oklab, var(--color-accent) 25%, transparent)',
            color: isUser ? 'var(--color-fg-dim)' : 'var(--color-accent)',
          }}
        >
          {isUser ? <User className="size-[10px]" /> : <MeshIcon />}
        </span>
        <span className="font-medium text-foreground">{isUser ? 'You' : 'Assistant'}</span>
        <span>·</span>
        {model && <><span className="font-mono">{model}</span><span>·</span></>}
        <span className="font-mono">{timestamp}</span>
        {routeMetadata ? (
          <>
            <span>·</span>
            {isUser && routeNode ? (
              <span className="inline-flex items-center gap-[5px]" style={{ color: inspected ? 'var(--color-accent)' : 'var(--color-fg-dim)' }}>
                <Send className="size-[10px]" /> sent to <span className="font-mono">{routeNode}</span>
              </span>
            ) : isResponse && route ? (
              <span className="inline-flex items-center gap-[5px]" style={{ color: inspected ? 'var(--color-accent)' : 'var(--color-fg-dim)' }}>
                <span className="size-[5px] rounded-full bg-accent" />
                routed via <span className="font-mono">{routeNode ?? route}</span>
              </span>
            ) : null}
          </>
        ) : null}
      </span>
      <span
        className="block select-text text-[length:var(--density-type-body-lg)] leading-[1.55]"
        style={{
          padding: isUser ? '6px 0 6px 14px' : '12px 16px',
          background: 'transparent',
          border: isUser ? '0 solid transparent' : '1px solid transparent',
          maxWidth: isUser ? 720 : 'none',
        }}
      >
        {body}
      </span>
      {isResponse && (tokens || tokPerSec || ttft) && (
        <span className="mt-2 flex select-none items-center gap-3 text-[length:var(--density-type-label)] text-fg-faint">
          {tokens && <span className="font-mono">{tokens}</span>}
          {tokPerSec && <><span>·</span><span className="font-mono">{tokPerSec}</span></>}
          {ttft && <><span>·</span><span className="font-mono">TTFT {ttft}</span></>}
        </span>
      )}
    </>
  )
  const content = isUser ? (
    <span
      className="block"
      style={userContentStyle}
    >
      {messageContent}
    </span>
  ) : messageContent

  if (!canInspect) {
    return (
      <article className={rowClassName} style={rowStyle}>
        {content}
      </article>
    )
  }

  return (
    <button
      aria-label={accessibleInspectLabel}
      onClick={inspect}
      type="button"
      className={rowClassName}
      style={rowStyle}
    >
      {content}
    </button>
  )
}
