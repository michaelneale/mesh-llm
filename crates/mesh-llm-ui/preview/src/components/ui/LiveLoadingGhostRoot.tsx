import { useRef, type ReactNode } from 'react'
import { useLoadingGhostShimmer } from '@/components/ui/useLoadingGhostShimmer'

type LiveLoadingGhostRootProps = {
  children: ReactNode
}

export function LiveLoadingGhostRoot({ children }: LiveLoadingGhostRootProps) {
  const rootRef = useRef<HTMLDivElement | null>(null)
  useLoadingGhostShimmer(rootRef)

  return <div ref={rootRef}>{children}</div>
}
