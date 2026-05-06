import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

type KVProps = { label: string; children: ReactNode; mono?: boolean; icon?: ReactNode }

export function KV({ label, children, mono = true, icon }: KVProps) {
  return (
    <div className="min-w-0 flex-1 rounded-[var(--radius)] border border-border-soft bg-background px-[12px] py-[10px]">
      <div className="mb-[3px] flex items-center gap-[5px] whitespace-nowrap text-[length:var(--density-type-annotation)] font-medium uppercase leading-[15px] tracking-[0.6px] text-fg-faint">
        {icon}
        {label}
      </div>
      <div
        className={cn(
          'truncate text-[length:var(--density-type-body)] leading-[18px] text-foreground',
          mono && 'font-mono'
        )}
      >
        {children}
      </div>
    </div>
  )
}
