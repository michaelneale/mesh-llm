import { useCallback, useEffect, useRef, useState } from 'react'

export type ClipboardCopyState = 'idle' | 'copied' | 'failed'

const DEFAULT_RESET_DELAY_MS = 1800

type UseClipboardCopyOptions = {
  resetDelayMs?: number
}

export function useClipboardCopy({ resetDelayMs = DEFAULT_RESET_DELAY_MS }: UseClipboardCopyOptions = {}) {
  const [copyState, setCopyState] = useState<ClipboardCopyState>('idle')
  const resetTimerRef = useRef<number | null>(null)

  const clearResetTimer = useCallback(() => {
    if (resetTimerRef.current === null || typeof window === 'undefined') return
    window.clearTimeout(resetTimerRef.current)
    resetTimerRef.current = null
  }, [])

  const queueReset = useCallback(() => {
    clearResetTimer()
    if (typeof window === 'undefined') {
      setCopyState('idle')
      return
    }

    resetTimerRef.current = window.setTimeout(() => {
      setCopyState('idle')
      resetTimerRef.current = null
    }, resetDelayMs)
  }, [clearResetTimer, resetDelayMs])

  useEffect(() => clearResetTimer, [clearResetTimer])

  const copyText = useCallback(
    async (text: string) => {
      if (typeof navigator === 'undefined' || typeof navigator.clipboard?.writeText !== 'function') {
        setCopyState('failed')
        queueReset()
        return false
      }

      try {
        await navigator.clipboard.writeText(text)
        setCopyState('copied')
        queueReset()
        return true
      } catch {
        setCopyState('failed')
        queueReset()
        return false
      }
    },
    [queueReset]
  )

  return { copyState, copyText }
}
