import type { CSSProperties } from 'react'

type VramDropStyleOptions = {
  invalidDropActive: boolean
  validDropActive: boolean
  reservedGB: number
  dense: boolean
  reservedColumnGB: number
  usableColumnGB: number
}

type VramFreeLaneStyleOptions = {
  invalidDropActive: boolean
  validDropActive: boolean
  free: number
}

export function vramDropBarStyle({
  invalidDropActive,
  validDropActive,
  reservedGB,
  dense,
  reservedColumnGB,
  usableColumnGB
}: VramDropStyleOptions): CSSProperties {
  return {
    borderRadius: 7,
    height: 48,
    gap: 3,
    gridTemplateColumns:
      reservedGB > 0
        ? `minmax(${dense ? 20 : 28}px, ${reservedColumnGB}fr) minmax(0, ${usableColumnGB}fr)`
        : 'minmax(0, 1fr)',
    borderColor: invalidDropActive
      ? 'color-mix(in oklch, var(--color-bad), var(--color-border-soft) 42%)'
      : validDropActive
        ? 'color-mix(in oklch, var(--color-accent), var(--color-border-soft) 55%)'
        : undefined,
    boxShadow: invalidDropActive
      ? 'var(--shadow-config-drop-invalid)'
      : validDropActive
        ? 'var(--shadow-config-drop-valid)'
        : undefined
  }
}

export function vramFreeLaneStyle({
  invalidDropActive,
  validDropActive,
  free
}: VramFreeLaneStyleOptions): CSSProperties {
  return {
    borderColor: invalidDropActive
      ? 'color-mix(in oklch, var(--color-bad), transparent 45%)'
      : validDropActive
        ? 'color-mix(in oklch, var(--color-accent), transparent 48%)'
        : 'var(--color-border-soft)',
    color: invalidDropActive ? 'var(--color-bad)' : validDropActive ? 'var(--color-accent)' : 'var(--color-fg-faint)',
    background: invalidDropActive
      ? 'color-mix(in oklch, var(--color-bad), transparent 90%)'
      : validDropActive
        ? 'color-mix(in oklch, var(--color-accent), transparent 90%)'
        : 'transparent',
    opacity: free > 0 || validDropActive || invalidDropActive ? 1 : 0
  }
}
