import { useEffect, useRef, type RefObject } from 'react'
import { animate, createScope, stagger, type Scope } from 'animejs'

const LOADING_GHOST_SHIMMER_SELECTOR = '[data-loading-ghost-shimmer]'

export function useLoadingGhostShimmer<TElement extends HTMLElement>(rootRef: RefObject<TElement | null>) {
  const scopeRef = useRef<Scope | null>(null)

  useEffect(() => {
    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReducedMotion) return undefined

    scopeRef.current = createScope({ root: rootRef }).add(() => {
      animate(LOADING_GHOST_SHIMMER_SELECTOR, {
        translateX: ['-130%', '260%'],
        opacity: [0, 0.42, 0],
        duration: 2400,
        delay: stagger(70),
        loop: true,
        loopDelay: 280,
        ease: 'inOutSine'
      })
    })

    return () => {
      scopeRef.current?.revert()
      scopeRef.current = null
    }
  }, [rootRef])
}
