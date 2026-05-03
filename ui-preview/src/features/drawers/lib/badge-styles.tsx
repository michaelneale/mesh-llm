import type { LucideIcon } from 'lucide-react'

export type DrawerBadgeTone = 'muted' | 'accent' | 'good' | 'warn' | 'bad'

export function drawerIcon(Icon: LucideIcon) {
  return <Icon aria-hidden="true" className="size-3 shrink-0" strokeWidth={1.8} />
}

export function badgeStyle(tone: DrawerBadgeTone) {
  if (tone === 'accent') {
    return {
      background: 'color-mix(in oklab, var(--color-accent) 10%, var(--color-panel))',
      border: '1px solid color-mix(in oklab, var(--color-accent) 24%, var(--color-background))',
      color: 'var(--color-accent)',
    }
  }

  if (tone === 'good') {
    return {
      background: 'color-mix(in oklab, var(--color-good) 18%, var(--color-background))',
      border: '1px solid color-mix(in oklab, var(--color-good) 30%, var(--color-background))',
      color: 'var(--color-good)',
    }
  }

  if (tone === 'warn') {
    return {
      background: 'color-mix(in oklab, var(--color-warn) 18%, var(--color-background))',
      border: '1px solid color-mix(in oklab, var(--color-warn) 32%, var(--color-background))',
      color: 'var(--color-warn)',
    }
  }

  if (tone === 'bad') {
    return {
      background: 'color-mix(in oklab, var(--color-bad) 14%, var(--color-background))',
      border: '1px solid color-mix(in oklab, var(--color-bad) 24%, var(--color-background))',
      color: 'var(--color-bad)',
    }
  }

  return {
    background: 'var(--color-background)',
    border: '1px solid var(--color-border)',
    color: 'var(--color-fg-faint)',
  }
}
