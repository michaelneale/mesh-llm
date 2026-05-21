import type { ReactNode } from 'react'

import { Badge } from '@/components/ui/badge'
import { TooltipContent, TooltipRoot, TooltipTrigger } from '@/components/ui/tooltip'

import { cn } from '@/lib/utils'

export type StatusPillTone = 'warm' | 'cold' | 'good' | 'info' | 'warn' | 'bad' | 'neutral'

export function StatusPill({
  label,
  tone,
  dot = false,
  icon,
  className,
  tooltip
}: {
  label: string
  tone: StatusPillTone
  dot?: boolean
  icon?: ReactNode
  className?: string
  tooltip?: string
}) {
  const badge = (
    <Badge
      className={cn(
        'h-5 shrink-0 rounded-full px-2 text-[10px] font-medium',
        statusPillToneClass(tone),
        dot || icon ? 'gap-1' : '',
        className
      )}
    >
      {dot ? <span className="h-1.5 w-1.5 rounded-full bg-current" /> : null}
      {icon}
      {label}
    </Badge>
  )
  if (!tooltip) return badge
  return (
    <TooltipRoot>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex rounded-full bg-transparent p-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {badge}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" align="center" sideOffset={8}>
        {tooltip}
      </TooltipContent>
    </TooltipRoot>
  )
}

function statusPillToneClass(tone: StatusPillTone) {
  if (tone === 'good') {
    return 'border-[color:color-mix(in_oklab,var(--color-good)_34%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-good)_7%,transparent)] text-[color:var(--color-good)]'
  }
  if (tone === 'warm' || tone === 'warn') {
    return 'border-[color:color-mix(in_oklab,var(--color-warn)_34%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-warn)_7%,transparent)] text-[color:var(--color-warn)]'
  }
  if (tone === 'cold' || tone === 'info') {
    return 'border-[color:color-mix(in_oklab,var(--color-accent)_34%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_7%,transparent)] text-[color:var(--color-accent)]'
  }
  if (tone === 'bad') {
    return 'border-[color:color-mix(in_oklab,var(--color-bad)_34%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-bad)_7%,transparent)] text-[color:var(--color-bad)]'
  }
  return 'border-border/70 bg-panel text-fg-dim'
}
