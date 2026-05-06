import { useCallback, useRef, type Ref } from 'react'
import { Code2, ListEnd, MessageSquareX, Paperclip, RotateCcw, Send, Square } from 'lucide-react'
import { Tooltip } from '../../../components/ui/Tooltip'
import { cn } from '@/lib/cn'

type ComposerProcessingStage = 'downloading' | 'starting' | 'processing'

type ComposerProps = {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onAttach?: (files: File[]) => void
  onSystemPrompt?: () => void
  attachmentCount?: number
  onStop?: () => void
  onRetry?: () => void
  canRetry?: boolean
  isStreaming?: boolean
  sendMode?: 'send' | 'queue'
  errorMessage?: string
  placeholder?: string
  disabled?: boolean
  isPreparingAttachments?: boolean
  preparingStage?: ComposerProcessingStage
  preparingAttachmentCount?: number
  textareaRef?: Ref<HTMLTextAreaElement>
  systemPromptButtonRef?: Ref<HTMLButtonElement>
  showSystemPromptButton?: boolean
}

function getPreparingLabel(stage: ComposerProcessingStage | undefined): string {
  if (stage === 'starting') return 'Starting local analyzer…'
  if (stage === 'processing') return 'Processing attachments…'
  return 'Downloading browser model…'
}

export function Composer({
  value,
  onChange,
  onSend,
  onAttach,
  onSystemPrompt,
  attachmentCount = 0,
  onStop,
  onRetry,
  canRetry = false,
  isStreaming = false,
  sendMode,
  errorMessage,
  placeholder = 'Ask me anything…',
  disabled = false,
  isPreparingAttachments = false,
  preparingStage,
  preparingAttachmentCount = 0,
  textareaRef,
  systemPromptButtonRef,
  showSystemPromptButton = false
}: ComposerProps) {
  const attachmentInputRef = useRef<HTMLInputElement | null>(null)
  const handleSend = useCallback(() => {
    if (!disabled && (value.trim() || attachmentCount > 0)) onSend()
  }, [attachmentCount, disabled, value, onSend])

  const sendDisabled = disabled || (!value.trim() && attachmentCount === 0)
  const retryDisabled = disabled || !canRetry
  const stopDisabled = disabled || !isStreaming || !onStop
  const submitQueuesPrompt = sendMode === 'queue' || isStreaming
  const preparingLabel = getPreparingLabel(preparingStage)

  return (
    <div
      className={cn(
        'composer-shell panel-shell overflow-hidden rounded-[var(--radius-lg)] border bg-panel transition-[border-color,box-shadow]',
        isPreparingAttachments
          ? 'border-[color:color-mix(in_oklab,var(--color-accent)_58%,var(--color-border))] shadow-[0_0_0_1px_color-mix(in_oklab,var(--color-accent)_18%,transparent),0_18px_55px_color-mix(in_oklab,var(--color-accent)_10%,transparent)]'
          : 'border-border'
      )}
      data-panel-interactive="true"
    >
      <label className="sr-only" htmlFor="prompt-composer">
        Prompt
      </label>
      <textarea
        ref={textareaRef}
        id="prompt-composer"
        className="block w-full resize-none bg-transparent px-4 py-3.5 text-[length:var(--density-type-body-lg)] leading-[1.5] outline-none focus-visible:outline-none placeholder:text-fg-faint"
        disabled={disabled}
        placeholder={isPreparingAttachments ? preparingLabel : placeholder}
        value={value}
        style={{ minHeight: 88, fontFamily: 'var(--font-sans)' }}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Escape' && isStreaming && onStop) {
            event.preventDefault()
            onStop()
            return
          }

          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            handleSend()
          }
        }}
      />
      <div className="flex items-center justify-between gap-3 border-t border-border-soft bg-panel px-3.5 py-2">
        <div className="flex shrink-0 items-center gap-1.5">
          <Tooltip content="Attach files">
            <button
              type="button"
              className="ui-control inline-flex size-7 items-center justify-center rounded-[var(--radius)] border"
              aria-label="Attach"
              disabled={disabled}
              onClick={() => attachmentInputRef.current?.click()}
            >
              <Paperclip className="size-[13px]" />
            </button>
          </Tooltip>
          <input
            ref={attachmentInputRef}
            hidden
            multiple
            type="file"
            disabled={disabled}
            onChange={(event) => {
              const files = Array.from(event.target.files ?? [])
              event.target.value = ''
              if (files.length > 0) onAttach?.(files)
            }}
          />
          {showSystemPromptButton ? (
            <Tooltip content="Set system prompt">
              <button
                ref={systemPromptButtonRef}
                type="button"
                className="ui-control inline-flex size-7 items-center justify-center rounded-[var(--radius)] border"
                aria-label="System prompt"
                disabled={disabled}
                onClick={onSystemPrompt}
              >
                <Code2 className="size-[13px]" />
              </button>
            </Tooltip>
          ) : null}
          <Tooltip content="Retry last response">
            <button
              type="button"
              className="ui-control inline-flex size-7 items-center justify-center rounded-[var(--radius)] border"
              aria-label="Retry last"
              disabled={retryDisabled}
              onClick={onRetry}
            >
              <RotateCcw className="size-[13px]" />
            </button>
          </Tooltip>
        </div>
        <div className="flex min-w-0 flex-1 items-center justify-end gap-3 text-[length:var(--density-type-caption)]">
          <div className="min-w-0 text-right">
            {errorMessage ? (
              <div className="space-y-2 text-left">
                <div
                  role="alert"
                  className="flex items-start gap-2.5 rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-bad)_38%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-bad)_10%,transparent)] px-3 py-2.5"
                >
                  <MessageSquareX aria-hidden={true} className="mt-0.5 size-5 shrink-0 text-bad" strokeWidth={1.5} />
                  <div className="min-w-0 space-y-1">
                    <div className="font-medium text-bad">Message failed to send</div>
                    <div className="break-words text-fg-muted">{errorMessage}</div>
                    <div className="text-fg-faint">Try selecting another model, then send the prompt again.</div>
                  </div>
                </div>
                {attachmentCount > 0 ? (
                  <span className="block text-fg-faint">
                    {attachmentCount} attachment{attachmentCount === 1 ? '' : 's'} ready
                  </span>
                ) : null}
              </div>
            ) : isPreparingAttachments ? (
              <span className="inline-flex items-center gap-2 text-accent">
                <span className="size-1.5 animate-pulse rounded-full bg-accent" aria-hidden={true} />
                {preparingLabel}{' '}
                {preparingAttachmentCount > 0 ? (
                  <span className="text-fg-faint">
                    {preparingAttachmentCount} file{preparingAttachmentCount === 1 ? '' : 's'}
                  </span>
                ) : null}
              </span>
            ) : isStreaming ? (
              <span className="text-fg-faint">Generating response… queue another prompt or stop.</span>
            ) : submitQueuesPrompt ? (
              <span className="text-fg-faint">Chat providers are busy… queue this prompt for this conversation.</span>
            ) : attachmentCount > 0 ? (
              <span className="text-fg-faint">
                {attachmentCount} attachment{attachmentCount === 1 ? '' : 's'} ready
              </span>
            ) : (
              <div className="hidden text-fg-faint md:block">
                <kbd
                  className="rounded border border-border bg-panel-strong px-[5px] py-px font-mono text-[length:var(--density-type-label)] text-fg-dim"
                  style={{ borderBottomWidth: 2 }}
                >
                  ↵
                </kbd>{' '}
                to send ·{' '}
                <kbd
                  className="rounded border border-border bg-panel-strong px-[5px] py-px font-mono text-[length:var(--density-type-label)] text-fg-dim"
                  style={{ borderBottomWidth: 2 }}
                >
                  ⇧ ↵
                </kbd>{' '}
                for newline
              </div>
            )}
          </div>
          {isStreaming ? (
            <button
              className="ui-control-destructive inline-flex items-center gap-1.5 rounded-[var(--radius)] px-3 py-1.5 text-[length:var(--density-type-control-lg)] font-medium"
              disabled={stopDisabled}
              type="button"
              onClick={onStop}
            >
              <Square className="size-[13px]" />
              Stop
            </button>
          ) : null}
          <button
            className="ui-control-primary inline-flex items-center gap-1.5 rounded-[var(--radius)] px-3 py-1.5 text-[length:var(--density-type-control-lg)] font-medium"
            disabled={sendDisabled}
            type="button"
            onClick={handleSend}
          >
            {submitQueuesPrompt ? <ListEnd className="size-[13px]" /> : <Send className="size-[13px]" />}
            {submitQueuesPrompt ? 'Queue' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}
