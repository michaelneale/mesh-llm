import type { ReactNode } from 'react'
import { Copy } from 'lucide-react'
import { cn } from '@/lib/cn'
import { copyStateLabel } from '@/lib/copyStateLabel'
import { useClipboardCopy } from '@/lib/useClipboardCopy'

type CopyInstructionRowProps = {
  label: string
  value: string
  copyValue?: string
  prefix?: string
  hint?: ReactNode
  noWrapValue?: boolean
  disabled?: boolean
}

export function CopyInstructionRow({
  label,
  value,
  copyValue = value,
  prefix,
  hint,
  noWrapValue = false,
  disabled = false
}: CopyInstructionRowProps) {
  const { copyState, copyText } = useClipboardCopy()

  return (
    <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="type-label text-fg-faint">{label}</div>
          <div className="mt-1 flex items-start gap-2">
            {prefix ? (
              <span className="shrink-0 font-mono text-[length:var(--density-type-caption-lg)] text-accent">
                {prefix}
              </span>
            ) : null}
            <span
              className={cn(
                'min-w-0 font-mono text-[length:var(--density-type-caption-lg)] text-foreground',
                disabled ? 'text-fg-faint' : '',
                noWrapValue ? 'block max-w-full overflow-x-auto whitespace-nowrap' : 'break-words'
              )}
            >
              {value}
            </span>
          </div>
          {hint ? <div className="mt-1 text-[length:var(--density-type-caption)] text-fg-faint">{hint}</div> : null}
        </div>
        <button
          aria-label={`Copy ${label}`}
          className={cn(
            'ui-control inline-flex shrink-0 items-center gap-1.5 rounded-[var(--radius)] border px-2.5 py-1 text-[length:var(--density-type-caption)] font-medium',
            disabled ? 'cursor-not-allowed opacity-60' : ''
          )}
          disabled={disabled}
          onClick={() => {
            if (disabled) return
            void copyText(copyValue)
          }}
          type="button"
        >
          <Copy className="size-[11px]" aria-hidden="true" />
          {disabled ? 'Unavailable' : copyStateLabel(copyState)}
        </button>
      </div>
    </div>
  )
}
