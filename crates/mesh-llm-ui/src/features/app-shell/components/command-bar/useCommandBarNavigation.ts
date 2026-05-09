import {
  type Dispatch,
  type KeyboardEvent as ReactKeyboardEvent,
  type SetStateAction,
  useCallback,
  useEffect,
  useRef
} from 'react'
import type {
  CommandBarBehavior,
  CommandBarMode,
  CommandBarNormalizedResult
} from '@/features/app-shell/components/command-bar/command-bar-types'

function isCommandBarNavigationTarget(target: EventTarget | null, listboxId: string) {
  if (!(target instanceof HTMLElement)) return false
  if (target instanceof HTMLInputElement && target.getAttribute('aria-controls') === listboxId) return true

  const listbox = target.closest('[role="listbox"]')
  return listbox instanceof HTMLElement && listbox.id === listboxId
}

interface UseCommandBarNavigationOptions<T> {
  activeIndex: number
  behavior: CommandBarBehavior
  isOpen: boolean
  listboxId: string
  modes: readonly CommandBarMode<T>[]
  onModeSwitch: (modeId: string) => void
  onSelectResult: (result: CommandBarNormalizedResult<T>) => void | Promise<void>
  results: readonly CommandBarNormalizedResult<T>[]
  setActiveIndex: Dispatch<SetStateAction<number>>
}

interface UseCommandBarNavigationResult {
  handleKeyDown: (event: ReactKeyboardEvent<HTMLDivElement>) => void
  registerOptionElement: (compositeKey: string, node: HTMLDivElement | null) => void
}

export function useCommandBarNavigation<T>({
  activeIndex,
  behavior,
  isOpen,
  listboxId,
  modes,
  onModeSwitch,
  onSelectResult,
  results,
  setActiveIndex
}: UseCommandBarNavigationOptions<T>): UseCommandBarNavigationResult {
  const activeOptionElementsRef = useRef(new Map<string, HTMLDivElement>())
  const shouldScrollActiveOptionIntoViewRef = useRef(false)

  const registerOptionElement = useCallback((compositeKey: string, node: HTMLDivElement | null) => {
    if (node) {
      activeOptionElementsRef.current.set(compositeKey, node)
      return
    }

    activeOptionElementsRef.current.delete(compositeKey)
  }, [])

  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (event.defaultPrevented) return

      if (event.key === 'ArrowDown') {
        if (!isCommandBarNavigationTarget(event.target, listboxId)) return
        if (results.length === 0) return
        event.preventDefault()
        shouldScrollActiveOptionIntoViewRef.current = true
        setActiveIndex((currentIndex) => {
          if (currentIndex < 0) return 0
          return Math.min(currentIndex + 1, results.length - 1)
        })
        return
      }

      if (event.key === 'ArrowUp') {
        if (!isCommandBarNavigationTarget(event.target, listboxId)) return
        if (results.length === 0) return
        event.preventDefault()
        shouldScrollActiveOptionIntoViewRef.current = true
        setActiveIndex((currentIndex) => {
          if (currentIndex < 0) return 0
          return Math.max(currentIndex - 1, 0)
        })
        return
      }

      if (event.key === 'Enter') {
        if (!isCommandBarNavigationTarget(event.target, listboxId)) return
        const activeResult = results[activeIndex]
        if (!activeResult) return
        event.preventDefault()
        void onSelectResult(activeResult)
        return
      }

      if (behavior !== 'distinct') return
      if (event.altKey || event.shiftKey) return
      if (!/^\d$/.test(event.key) || event.key === '0') return
      if (!event.ctrlKey) return

      const nextMode = modes[Number(event.key) - 1]
      if (!nextMode) return

      event.preventDefault()
      onModeSwitch(nextMode.id)
    },
    [activeIndex, behavior, listboxId, modes, onModeSwitch, onSelectResult, results, setActiveIndex]
  )

  useEffect(() => {
    if (!isOpen) {
      shouldScrollActiveOptionIntoViewRef.current = false
      return
    }

    if (!shouldScrollActiveOptionIntoViewRef.current) return

    const activeResult = results[activeIndex]
    if (!activeResult) {
      shouldScrollActiveOptionIntoViewRef.current = false
      return
    }

    activeOptionElementsRef.current.get(activeResult.compositeKey)?.scrollIntoView?.({ block: 'nearest' })
    shouldScrollActiveOptionIntoViewRef.current = false
  }, [activeIndex, isOpen, results])

  return { handleKeyDown, registerOptionElement }
}
