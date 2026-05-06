import { useCallback } from 'react'
import { Paperclip, Code2, RotateCcw, Send } from 'lucide-react'

type ComposerProps = {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  placeholder?: string
  disabled?: boolean
}

export function Composer({
  value,
  onChange,
  onSend,
  placeholder = 'Ask me anything\u2026',
  disabled = false
}: ComposerProps) {
  const handleSend = useCallback(() => {
    if (!disabled && value.trim()) onSend()
  }, [disabled, value, onSend])

  return (
    <div
      className="composer-shell panel-shell overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel transition-[border-color,box-shadow]"
      data-panel-interactive="true"
    >
      <label className="sr-only" htmlFor="prompt-composer">
        Prompt
      </label>
      <textarea
        id="prompt-composer"
        className="block w-full resize-none bg-transparent px-4 py-3.5 text-[length:var(--density-type-body-lg)] leading-[1.5] outline-none focus-visible:outline-none placeholder:text-fg-faint"
        disabled={disabled}
        placeholder={placeholder}
        value={value}
        style={{ minHeight: 88, fontFamily: 'var(--font-sans)' }}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            handleSend()
          }
        }}
      />
      <div className="flex items-center justify-between border-t border-border-soft bg-panel px-3.5 py-2">
        <div className="hidden text-[length:var(--density-type-caption)] text-fg-faint md:block">
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
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            className="ui-control inline-flex size-7 items-center justify-center rounded-[var(--radius)] border"
            aria-label="Attach"
          >
            <Paperclip className="size-[13px]" />
          </button>
          <button
            type="button"
            className="ui-control inline-flex size-7 items-center justify-center rounded-[var(--radius)] border"
            aria-label="System prompt"
          >
            <Code2 className="size-[13px]" />
          </button>
          <button
            type="button"
            className="ui-control inline-flex size-7 items-center justify-center rounded-[var(--radius)] border"
            aria-label="Retry last"
          >
            <RotateCcw className="size-[13px]" />
          </button>
          <button
            className="ui-control-primary inline-flex items-center gap-1.5 rounded-[var(--radius)] px-3 py-1.5 text-[length:var(--density-type-control-lg)] font-medium"
            disabled={disabled || !value.trim()}
            type="button"
            onClick={handleSend}
          >
            <Send className="size-[13px]" /> Send
          </button>
        </div>
      </div>
    </div>
  )
}
