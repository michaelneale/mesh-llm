import { animate, createScope, type Scope } from 'animejs'
import { useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'
import { formatReserveVram } from '@/features/reserves/lib/reserve-formatters'
import { getReserveStateMeta } from '@/features/reserves/lib/reserve-state'
import type { ReserveNode, ReserveNodeState } from '@/features/reserves/lib/reserve-types'

type ReserveNodeDotProps = {
  node: ReserveNode
}

export function ReserveNodeDot({ node }: ReserveNodeDotProps) {
  const dotRef = useRef<HTMLSpanElement | null>(null)
  const glowRef = useRef<HTMLSpanElement | null>(null)
  const scopeRef = useRef<Scope | null>(null)
  const meta = getReserveStateMeta(node.state)
  const detail = [node.id, node.hw, formatReserveVram(node.vram), meta.label, node.error, node.lastSeen]
    .filter(Boolean)
    .join(' · ')

  useEffect(() => {
    scopeRef.current?.revert()
    scopeRef.current = null

    const dot = dotRef.current
    const glow = glowRef.current
    if (!meta.pulse || !dot || !glow) return undefined

    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches

    if (prefersReducedMotion) return undefined

    scopeRef.current = createScope({ root: dotRef }).add(() => {
      animate(dot, {
        scale: [1, 1.025, 1],
        duration: 1600,
        loop: true,
        ease: 'inOutSine'
      })

      animate(glow, {
        opacity: [0.28, 0.58, 0.28],
        scale: [0.86, 1.16, 0.86],
        duration: 1600,
        loop: true,
        ease: 'inOutSine'
      })
    })

    return () => {
      scopeRef.current?.revert()
      scopeRef.current = null
    }
  }, [meta.pulse])

  return (
    <button
      aria-label={detail}
      className="relative inline-flex size-[14px] cursor-pointer overflow-visible rounded-[3px] border-0 bg-transparent p-0 transition-transform duration-150 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      ref={dotRef as React.RefObject<HTMLButtonElement>}
      title={detail}
      type="button"
    >
      {meta.pulse ? (
        <span
          aria-hidden="true"
          className={cn(
            'pointer-events-none absolute inset-[-5px] rounded-full opacity-30 blur-[1.5px]',
            reserveNodeDotGlowClass(node.state)
          )}
          ref={glowRef}
        />
      ) : null}
      <span
        aria-hidden="true"
        className={cn('relative z-[1] size-full rounded-[3px] border', reserveNodeDotClass(node.state))}
      />
    </button>
  )
}

function reserveNodeDotGlowClass(state: ReserveNodeState) {
  switch (state) {
    case 'waking':
      return 'bg-[radial-gradient(circle,color-mix(in_oklab,var(--color-warn)_38%,transparent)_0%,color-mix(in_oklab,var(--color-warn)_18%,transparent)_44%,transparent_74%)]'
    case 'joining':
      return 'bg-[radial-gradient(circle,color-mix(in_oklab,var(--color-accent)_42%,transparent)_0%,color-mix(in_oklab,var(--color-accent)_20%,transparent)_44%,transparent_74%)]'
    case 'standby':
    case 'online':
    case 'failed':
    case 'unreachable':
      return null
  }
}

function reserveNodeDotClass(state: ReserveNodeState) {
  switch (state) {
    case 'standby':
      return 'border-[color:color-mix(in_oklab,var(--color-fg-faint)_30%,transparent)] bg-[color:color-mix(in_oklab,var(--color-fg-faint)_14%,var(--color-panel-strong))]'
    case 'waking':
      return 'border-[color:color-mix(in_oklab,var(--color-warn)_70%,transparent)] bg-[color:color-mix(in_oklab,var(--color-warn)_60%,var(--color-panel-strong))]'
    case 'joining':
      return 'border-[color:color-mix(in_oklab,var(--color-accent)_70%,transparent)] bg-[color:color-mix(in_oklab,var(--color-accent)_75%,var(--color-panel-strong))]'
    case 'online':
      return 'border-[color:color-mix(in_oklab,var(--color-good)_70%,transparent)] bg-[color:color-mix(in_oklab,var(--color-good)_55%,var(--color-panel-strong))]'
    case 'failed':
      return 'border-[color:color-mix(in_oklab,var(--color-bad)_70%,transparent)] bg-[color:color-mix(in_oklab,var(--color-bad)_65%,var(--color-panel-strong))]'
    case 'unreachable':
      return 'border-[color:color-mix(in_oklab,var(--color-bad)_70%,transparent)] bg-[repeating-linear-gradient(135deg,color-mix(in_oklab,var(--color-bad)_55%,var(--color-panel-strong))_0_2px,color-mix(in_oklab,var(--color-bad)_18%,var(--color-panel-strong))_2px_4px)]'
  }
}
