import { Badge } from '@/components/ui/badge'
import { Tooltip } from '@/components/ui/tooltip'
import { cn } from '@/lib/cn'

export const RESTART_REQUIRED_TOOLTIP = 'This setting requires a restart to take effect'

type RestartBadgeProps = {
  className?: string
}

export function RestartBadge({ className }: RestartBadgeProps) {
  return (
    <Tooltip content={RESTART_REQUIRED_TOOLTIP} side="bottom">
      <button
        type="button"
        className="inline-flex rounded-full bg-transparent p-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <Badge
          className={cn(
            'h-5 shrink-0 rounded-full border-border-soft bg-surface px-1.5 py-0 text-[length:var(--density-type-annotation)] font-medium text-fg-dim',
            className
          )}
        >
          Restart required
        </Badge>
      </button>
    </Tooltip>
  )
}
