import type { ReactNode } from 'react'
import { badgeStyle, type DrawerBadgeTone } from '@/features/drawers/lib/badge-styles'

export type { DrawerBadgeTone }

export function DrawerBadge({ children, tone = 'muted', dot = false }: { children: ReactNode; tone?: DrawerBadgeTone; dot?: boolean }) {
  return (
    <span
      className="inline-flex items-center gap-[5px] rounded-full px-2 py-px text-[length:var(--density-type-label)] font-medium"
      style={badgeStyle(tone)}
    >
      {dot ? <span className="size-[5px] rounded-full bg-current" /> : null}
      {children}
    </span>
  )
}
