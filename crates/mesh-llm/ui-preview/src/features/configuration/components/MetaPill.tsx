import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

type MetaPillTone = 'dim' | 'faint'
type MetaPillSize = 'label' | 'annotation'

type MetaPillProps = {
  children: ReactNode
  className?: string
  size?: MetaPillSize
  tone?: MetaPillTone
}

const metaPillSizeClass: Record<MetaPillSize, string> = {
  label: 'text-[length:var(--density-type-label)]',
  annotation: 'text-[length:var(--density-type-annotation)]'
}

const metaPillToneClass: Record<MetaPillTone, string> = {
  dim: 'text-fg-dim',
  faint: 'text-fg-faint'
}

export function MetaPill({ children, className, size = 'label', tone = 'dim' }: MetaPillProps) {
  return (
    <span
      className={cn(
        'inline-flex min-w-0 items-center rounded-full border border-border-soft bg-background px-2 py-0.5 font-mono font-medium',
        metaPillSizeClass[size],
        metaPillToneClass[tone],
        className
      )}
    >
      {children}
    </span>
  )
}
