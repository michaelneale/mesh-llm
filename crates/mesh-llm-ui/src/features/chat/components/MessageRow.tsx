import type { CSSProperties } from 'react'
import {
  BrainCircuit,
  Eye,
  FileIcon,
  FileImage,
  FileText,
  Loader2,
  MessageSquareX,
  Music,
  Send,
  User,
  X
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { MessageRole } from '@/features/app-tabs/types'
import { cn } from '@/lib/cn'
import { ResponseStatsBar } from '@/features/chat/components/ResponseStatsBar'
import { splitAssistantThinking } from '@/features/chat/components/thinking-segments'

export type MessageAttachmentAction = {
  id: string
  label: string
  kind: 'image' | 'pdf' | 'audio' | 'file'
  fileName: string
  onOpen: () => void
}

function MeshIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width={10}
      height={10}
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

type MessageRowProps = {
  messageRole: MessageRole
  body: string
  timestamp: string
  model?: string
  state?: 'default' | 'queued' | 'streaming' | 'stopped' | 'error'
  route?: string
  tokens?: string
  inspect?: () => void
  inspectLabel?: string
  inspected?: boolean
  routeNode?: string
  showRouteMetadata?: boolean
  tokPerSec?: string
  ttft?: string
  onStopStreaming?: () => void
  onRemoveQueued?: () => void
  attachments?: MessageAttachmentAction[]
}

function AttachmentIcon({ kind }: { kind: MessageAttachmentAction['kind'] }) {
  if (kind === 'image') return <FileImage className="size-3.5" aria-hidden={true} />
  if (kind === 'pdf') return <FileText className="size-3.5" aria-hidden={true} />
  if (kind === 'audio') return <Music className="size-3.5" aria-hidden={true} />
  return <FileIcon className="size-3.5" aria-hidden={true} />
}

function AssistantMarkdown({ text, linksEnabled }: { text: string; linksEnabled: boolean }) {
  return (
    <div className="block select-text break-words [&_code]:rounded-[calc(var(--radius)-2px)] [&_code]:bg-panel-strong [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.93em] [&_em]:text-fg-dim [&_strong]:font-semibold [&_strong]:text-foreground">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a(props) {
            const { node, ...anchorProps } = props
            void node

            if (!linksEnabled) {
              return (
                <span className="text-accent underline underline-offset-2" title={anchorProps.href}>
                  {anchorProps.children}
                </span>
              )
            }

            return <a {...anchorProps} rel="noreferrer noopener" target="_blank" />
          },
          blockquote(props) {
            const { node, ...blockquoteProps } = props
            void node
            return <blockquote {...blockquoteProps} className="my-2 block border-l border-border pl-3 text-fg-dim" />
          },
          h1(props) {
            const { node, ...headingProps } = props
            void node
            return (
              <h1
                {...headingProps}
                className="mb-2 mt-3 block text-[length:var(--density-type-title)] font-semibold first:mt-0"
              />
            )
          },
          h2(props) {
            const { node, ...headingProps } = props
            void node
            return (
              <h2
                {...headingProps}
                className="mb-2 mt-3 block text-[length:var(--density-type-control-lg)] font-semibold first:mt-0"
              />
            )
          },
          h3(props) {
            const { node, ...headingProps } = props
            void node
            return (
              <h3
                {...headingProps}
                className="mb-1.5 mt-3 block text-[length:var(--density-type-body-lg)] font-semibold first:mt-0"
              />
            )
          },
          hr(props) {
            const { node, ...separatorProps } = props
            void node
            return <hr {...separatorProps} className="my-3 block border-t border-border-soft" />
          },
          li(props) {
            const { node, ...itemProps } = props
            void node
            return (
              <li {...itemProps} className="my-0.5 block pl-1">
                <span aria-hidden={true} className="mr-2 text-fg-faint">
                  •
                </span>
                <span className="inline [&>span]:my-0 [&>span]:inline">{itemProps.children}</span>
              </li>
            )
          },
          ol(props) {
            const { node, ...listProps } = props
            void node
            return <ol {...listProps} className="my-2 block list-none pl-4" />
          },
          p(props) {
            const { node, ...paragraphProps } = props
            void node
            return <p {...paragraphProps} className="my-2 block first:mt-0 last:mb-0" />
          },
          pre(props) {
            const { node, ...preProps } = props
            void node
            return (
              <pre
                {...preProps}
                className="my-2 block max-w-full overflow-x-auto whitespace-pre rounded-[var(--radius)] border border-border-soft bg-panel p-3 [&_code]:bg-transparent [&_code]:p-0"
              />
            )
          },
          table(props) {
            const { className, node, ...tableProps } = props
            void node
            return (
              <div className="my-2 max-w-full overflow-x-auto">
                <table
                  {...tableProps}
                  className={cn('w-full border-collapse text-[length:var(--density-type-caption)]', className)}
                />
              </div>
            )
          },
          tbody(props) {
            const { node, ...tbodyProps } = props
            void node
            return <tbody {...tbodyProps} />
          },
          td(props) {
            const { className, node, ...cellProps } = props
            void node
            return (
              <td
                {...cellProps}
                className={cn('border border-border-soft px-2 py-1 align-top text-fg-dim', className)}
              />
            )
          },
          th(props) {
            const { className, node, ...cellProps } = props
            void node
            return (
              <th
                {...cellProps}
                className={cn(
                  'border border-border-soft bg-panel-strong px-2 py-1 text-left font-semibold text-foreground',
                  className
                )}
              />
            )
          },
          thead(props) {
            const { node, ...theadProps } = props
            void node
            return <thead {...theadProps} />
          },
          tr(props) {
            const { node, ...rowProps } = props
            void node
            return <tr {...rowProps} />
          },
          ul(props) {
            const { node, ...listProps } = props
            void node
            return <ul {...listProps} className="my-2 block list-none pl-4" />
          }
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}

function AssistantMessageContent({
  body,
  linksEnabled,
  streaming
}: {
  body: string
  linksEnabled: boolean
  streaming: boolean
}) {
  const segments = splitAssistantThinking(body, { streaming })

  if (segments.length === 0) return null

  return (
    <div className="block space-y-3">
      {segments.map((segment, index) => {
        const key = `${segment.kind}-${index}`

        if (segment.kind === 'thinking') {
          const active = streaming && segment.open

          return (
            <div
              className="block rounded-[var(--radius)] border px-3 py-2.5 text-[length:var(--density-type-body)] leading-[1.5] text-fg-dim"
              data-thinking-state={active ? 'active' : 'complete'}
              key={key}
              style={{
                background: 'color-mix(in oklab, var(--color-accent) 5%, var(--color-panel))',
                borderColor: 'color-mix(in oklab, var(--color-accent) 22%, var(--color-border-soft))'
              }}
            >
              <span className="mb-1.5 flex select-none items-center gap-1.5 font-mono text-[length:var(--density-type-label)] uppercase tracking-[0.07em] text-fg-faint">
                {active ? (
                  <Loader2 className="size-3 animate-spin" aria-hidden={true} />
                ) : (
                  <BrainCircuit className="size-3" aria-hidden={true} strokeWidth={1.7} />
                )}
                <span>{active ? 'Thinking' : 'Thinking trace'}</span>
              </span>
              <span className="block select-text whitespace-pre-wrap break-words">{segment.text}</span>
            </div>
          )
        }

        return (
          <div className="block select-text break-words" key={key} style={{ border: '1px solid transparent' }}>
            <AssistantMarkdown text={segment.text} linksEnabled={linksEnabled} />
          </div>
        )
      })}
    </div>
  )
}

export function MessageRow({
  messageRole,
  body,
  timestamp,
  model,
  state = 'default',
  route,
  tokens,
  inspect,
  inspectLabel,
  inspected,
  routeNode,
  showRouteMetadata = true,
  tokPerSec,
  ttft,
  onStopStreaming,
  onRemoveQueued,
  attachments = []
}: MessageRowProps) {
  const isUser = messageRole === 'user'
  const isResponse = !isUser
  const isQueued = state === 'queued'
  const isStreamingPlaceholder = state === 'streaming'
  const isStopped = state === 'stopped'
  const isError = state === 'error'
  const hasAttachmentActions = isUser && attachments.length > 0
  const canInspect = inspect != null
  const canRemoveQueued = isQueued && onRemoveQueued != null
  const routeMetadata = showRouteMetadata && ((isUser && routeNode) || (isResponse && route))
  const displayModel = isQueued ? 'Queued' : model
  const rowClassName = cn(
    'relative -mx-2 mb-5 block w-[calc(100%+16px)] select-none rounded-[var(--radius-lg)] border-0 bg-transparent px-2 py-1 text-left transition-[background,box-shadow] duration-150',
    canInspect &&
      'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent'
  )
  const rowStyle: CSSProperties = {
    ...(inspected ? { background: 'color-mix(in oklab, var(--color-accent) 4%, transparent)' } : {})
  }
  const userContentStyle: CSSProperties = {
    background: 'var(--chat-user-message-background)',
    borderLeft: isError
      ? '1px solid var(--color-bad)'
      : isQueued
        ? '1px solid color-mix(in oklab, var(--color-fg-faint) 42%, var(--color-border))'
        : '1px solid var(--color-accent)',
    padding: '8px 12px 8px 14px'
  }
  const errorContentStyle: CSSProperties = {
    ...userContentStyle,
    background: 'color-mix(in oklab, var(--color-bad) 8%, var(--chat-user-message-background))'
  }
  const accessibleInspectLabel = inspectLabel ?? `Inspect ${isUser ? 'user' : 'assistant'} message from ${timestamp}`
  const headerMetadata = displayModel ? [displayModel] : []
  const messageContent = (
    <>
      <div className="mb-1.5 flex select-none items-center gap-2 text-[length:var(--density-type-caption)] text-fg-faint">
        <span
          className="inline-flex size-4 items-center justify-center rounded-[var(--radius)]"
          style={{
            background: isUser
              ? 'var(--color-panel-strong)'
              : 'color-mix(in oklab, var(--color-accent) 25%, transparent)',
            color: isUser ? 'var(--color-fg-dim)' : 'var(--color-accent)'
          }}
        >
          {isUser ? <User className="size-[10px]" /> : <MeshIcon />}
        </span>
        <span className="font-medium text-foreground">{isUser ? 'You' : 'Assistant'}</span>
        {headerMetadata.map((metadata) => (
          <span className="contents" key={metadata}>
            <span>·</span>
            <span className="font-mono">{metadata}</span>
          </span>
        ))}
        {routeMetadata ? (
          <>
            <span>·</span>
            {isUser && routeNode ? (
              <span
                className="inline-flex items-center gap-[5px]"
                style={{ color: inspected ? 'var(--color-accent)' : 'var(--color-fg-dim)' }}
              >
                <Send className="size-[10px]" /> sent to <span className="font-mono">{routeNode}</span>
              </span>
            ) : isResponse && route ? (
              <span
                className="inline-flex items-center gap-[5px]"
                style={{ color: inspected ? 'var(--color-accent)' : 'var(--color-fg-dim)' }}
              >
                <span className="size-[5px] rounded-full bg-accent" />
                routed via <span className="font-mono">{routeNode ?? route}</span>
              </span>
            ) : null}
          </>
        ) : null}
        {canInspect && !hasAttachmentActions ? (
          <button
            type="button"
            className="ui-control ml-auto inline-flex h-7 items-center gap-1.5 rounded-[var(--radius)] border px-2 text-[length:var(--density-type-caption)] font-medium text-fg-muted outline-none transition-[background,color,box-shadow,transform] hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            aria-label={accessibleInspectLabel}
            title={accessibleInspectLabel}
            onClick={inspect}
          >
            <Eye className="size-3.5" aria-hidden={true} />
            <span>Inspect</span>
          </button>
        ) : null}
      </div>
      <div
        className="block select-text text-[length:var(--density-type-body-lg)] leading-[1.55]"
        style={{
          padding: isUser || isError ? '6px 0 6px 14px' : '12px 16px',
          background: 'transparent',
          border: isUser || isError ? '0 solid transparent' : '1px solid transparent',
          maxWidth: isUser || isError ? 720 : 'none'
        }}
      >
        {isError ? (
          <div className="block space-y-1">
            <span className="block font-medium text-bad">Message failed to send</span>
            <span className="block break-words text-fg-muted">{body}</span>
            <span className="block text-[length:var(--density-type-caption)] text-fg-faint">
              Try selecting another model, then send the prompt again.
            </span>
          </div>
        ) : isResponse ? (
          <AssistantMessageContent body={body} linksEnabled={true} streaming={isStreamingPlaceholder} />
        ) : (
          body
        )}
        {isStreamingPlaceholder ? (
          <span className="mt-3 block">
            <button
              aria-label="Stop streaming"
              className="ui-control ui-control-destructive-hover group/stop inline-flex items-center gap-2 rounded-[var(--radius)] border px-2.5 py-1.5 font-mono text-[length:var(--density-type-caption)] outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-destructive"
              onClick={(event) => {
                event.stopPropagation()
                onStopStreaming?.()
              }}
              type="button"
            >
              <Loader2 className="size-3.5 animate-spin" />
              <span className="group-hover/stop:hidden">Streaming response...</span>
              <span className="hidden group-hover/stop:inline">Stop Streaming</span>
            </button>
          </span>
        ) : null}
      </div>
      {isResponse && !isError ? (
        <ResponseStatsBar tokens={tokens} tokPerSec={tokPerSec} ttft={ttft} stopped={isStopped} />
      ) : null}
    </>
  )
  const content = isError ? (
    <div className="flex items-center justify-between gap-5" role="alert" style={errorContentStyle}>
      <div className="min-w-0">{messageContent}</div>
      <MessageSquareX aria-hidden={true} className="size-8 shrink-0 text-bad" strokeWidth={1.6} />
    </div>
  ) : isUser ? (
    <div
      className={cn('relative block', hasAttachmentActions && 'flex items-center justify-between gap-4')}
      style={userContentStyle}
    >
      <div className="min-w-0">{messageContent}</div>
      {canRemoveQueued ? (
        <button
          type="button"
          className="ui-control absolute right-3 top-1/2 z-10 inline-flex size-8 -translate-y-1/2 items-center justify-center rounded-[var(--radius)] border text-fg-muted outline-none transition-[background,color,box-shadow,transform] hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          aria-label="Remove queued message"
          title="Remove queued message"
          onClick={(event) => {
            event.stopPropagation()
            onRemoveQueued()
          }}
        >
          <X className="size-4" aria-hidden={true} strokeWidth={1.8} />
        </button>
      ) : null}
      {hasAttachmentActions ? (
        <span className="flex shrink-0 flex-col items-end justify-center gap-1.5">
          {inspect ? (
            <button
              type="button"
              className="ui-control inline-flex h-8 max-w-[12rem] items-center gap-1.5 rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-caption)] font-medium text-fg-muted outline-none transition-[background,color,box-shadow,transform] hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              aria-label={accessibleInspectLabel}
              title={accessibleInspectLabel}
              onClick={(event) => {
                event.stopPropagation()
                inspect()
              }}
            >
              <Eye className="size-3.5" aria-hidden={true} />
              <span className="truncate">Inspect</span>
            </button>
          ) : null}
          {attachments.map((attachment) => (
            <button
              key={attachment.id}
              type="button"
              className="ui-control inline-flex h-8 max-w-[12rem] items-center gap-1.5 rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-caption)] font-medium text-fg-muted outline-none transition-[background,color,box-shadow,transform] hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              aria-label={`Open ${attachment.fileName}`}
              title={attachment.fileName}
              onClick={(event) => {
                event.stopPropagation()
                attachment.onOpen()
              }}
            >
              <AttachmentIcon kind={attachment.kind} />
              <span className="truncate">{attachment.label}</span>
            </button>
          ))}
        </span>
      ) : null}
    </div>
  ) : (
    messageContent
  )

  return (
    <article className={rowClassName} style={rowStyle}>
      {content}
    </article>
  )
}
