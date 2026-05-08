import type { CSSProperties, ReactNode } from 'react'
import { cn } from '@/lib/cn'

export type StatusBadgeTone = 'muted' | 'accent' | 'good' | 'warn' | 'bad'

type StatusBadgeSize = 'label' | 'caption'

type StatusBadgeProps = {
  children: ReactNode
  className?: string
  dot?: boolean
  size?: StatusBadgeSize
  tone?: StatusBadgeTone
}

const toneColor: Record<StatusBadgeTone, string> = {
  muted: 'var(--color-fg-faint)',
  accent: 'var(--color-accent)',
  good: 'var(--color-good)',
  warn: 'var(--color-warn)',
  bad: 'var(--color-bad)'
}

const sizeClass: Record<StatusBadgeSize, string> = {
  label: 'text-[length:var(--density-type-label)]',
  caption: 'text-[length:var(--density-type-caption)]'
}

function statusBadgeStyle(tone: StatusBadgeTone = 'muted'): CSSProperties {
  const color = toneColor[tone]

  return {
    background:
      tone === 'muted'
        ? 'color-mix(in oklab, var(--color-fg-faint) 12%, var(--color-background))'
        : `color-mix(in oklab, ${color} 18%, var(--color-background))`,
    border:
      tone === 'muted'
        ? '1px solid color-mix(in oklab, var(--color-border) 80%, var(--color-background))'
        : `1px solid color-mix(in oklab, ${color} 30%, var(--color-background))`,
    color: tone === 'muted' ? 'var(--color-fg-dim)' : color
  }
}

export function StatusBadge({ children, className, dot = false, size = 'label', tone = 'muted' }: StatusBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-[5px] rounded-full px-2 py-px font-medium',
        sizeClass[size],
        className
      )}
      style={statusBadgeStyle(tone)}
    >
      {dot ? <span className="size-[5px] rounded-full bg-current" /> : null}
      {children}
    </span>
  )
}
