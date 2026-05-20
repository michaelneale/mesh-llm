import { Cog, Moon, Signal, type LucideProps } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { StatusPill } from '@/components/ui/status-pill'
import type { ReservePolicyTileData } from '@/features/reserves/lib/reserve-types'
import { cn } from '@/lib/utils'

type ReservePolicyTileProps = {
  tile: ReservePolicyTileData
}

export function ReservePolicyTile({ tile }: ReservePolicyTileProps) {
  return (
    <Card className="rounded-[var(--radius)] border-border/70 bg-panel-strong shadow-none">
      <CardContent className="relative px-[14px] py-3">
        <div className="flex items-center gap-2 pr-24 text-[10.5px] font-medium uppercase leading-none tracking-[0.055em] text-fg-faint">
          <ReservePolicyTileIcon aria-hidden="true" className="size-[11px] shrink-0" title={tile.title} />
          <span>{tile.title}</span>
        </div>
        {tile.status ? (
          <StatusPill
            className="absolute right-[14px] top-3 h-[18px] px-1.5 text-[10px] font-medium"
            dot
            label={tile.status}
            tone={tile.status === 'Enabled' ? 'good' : 'neutral'}
          />
        ) : (
          <div className={cn('mt-2 text-[12px] font-medium leading-tight text-foreground', 'font-mono')}>
            {tile.value}
          </div>
        )}
        <p className={cn('text-[11.5px] leading-snug text-fg-faint', tile.status ? 'mt-3' : 'mt-1.5')}>
          {tile.explanation}
        </p>
      </CardContent>
    </Card>
  )
}

type ReservePolicyTileIconProps = LucideProps & {
  title: string
}

function ReservePolicyTileIcon({ title, ...props }: ReservePolicyTileIconProps) {
  if (title === 'Auto-wake') return <Signal {...props} />
  if (title === 'Sleep idle reserves') return <Moon {...props} />
  return <Cog {...props} />
}
