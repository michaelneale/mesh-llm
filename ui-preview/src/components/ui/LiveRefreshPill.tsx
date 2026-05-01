import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

type LiveRefreshPillProps = {
  children: ReactNode
  className?: string
}

export function LiveRefreshPill({ children, className }: LiveRefreshPillProps) {
  return (
    <div className={cn('inline-flex items-center gap-2 rounded-full border border-border-soft bg-panel px-2.5 py-1 text-[length:var(--density-type-caption)] font-medium text-fg-faint', className)}>
      <span aria-hidden="true" className="size-1.5 rounded-full bg-accent" />
      {children}
    </div>
  )
}
