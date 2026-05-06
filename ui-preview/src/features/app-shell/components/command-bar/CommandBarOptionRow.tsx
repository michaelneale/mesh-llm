import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'
import { Badge } from './command-bar-primitives'
import { getCommandBarOptionId } from './command-bar-helpers'
import type { CommandBarBehavior, CommandBarNormalizedResult } from './command-bar-types'

interface CommandBarOptionRowProps<T> {
  behavior: CommandBarBehavior
  isActive: boolean
  listboxId: string
  onClick: () => void
  onPointerMove: () => void
  optionClassName?: string
  optionRef?: (node: HTMLDivElement | null) => void
  renderItem: (result: CommandBarNormalizedResult<T>, isActive: boolean) => ReactNode
  result: CommandBarNormalizedResult<T>
}

export function CommandBarOptionRow<T>({
  behavior,
  isActive,
  listboxId,
  onClick,
  onPointerMove,
  optionClassName,
  optionRef,
  renderItem,
  result
}: CommandBarOptionRowProps<T>) {
  return (
    <div
      ref={optionRef}
      id={getCommandBarOptionId(listboxId, result.compositeKey)}
      role="option"
      tabIndex={-1}
      aria-selected={isActive}
      onClick={onClick}
      onKeyDown={(event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return
        event.preventDefault()
        onClick()
      }}
      onPointerMove={onPointerMove}
      className={cn(
        'mx-1 flex min-h-11 items-center gap-3 rounded-[var(--radius)] px-3 py-2 transition-colors',
        'cursor-pointer',
        isActive ? 'bg-[color:color-mix(in_oklch,var(--color-accent)_8%,var(--color-panel))]' : 'bg-transparent',
        optionClassName
      )}
    >
      <div className="min-w-0 flex-1">{renderItem(result, isActive)}</div>
      {behavior === 'combined' ? (
        <Badge className="shrink-0 rounded-[var(--radius)] border-border bg-panel px-2 py-0.5 uppercase tracking-[0.08em] text-muted-foreground">
          {result.modeLabel}
        </Badge>
      ) : null}
    </div>
  )
}
