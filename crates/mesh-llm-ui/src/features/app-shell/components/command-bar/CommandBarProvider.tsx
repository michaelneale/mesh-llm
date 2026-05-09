import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  CommandBarContext,
  type CommandBarContextValue
} from '@/features/app-shell/components/command-bar/command-bar-context'

export function CommandBarProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [activeModeId, setActiveModeId] = useState<string | null>(null)
  const [activeIndex, setActiveIndex] = useState(0)
  const [selectionError, setSelectionError] = useState<string | null>(null)
  const [returnFocusElement, setReturnFocusElement] = useState<HTMLElement | null>(null)
  const isOpenRef = useRef(isOpen)
  const returnFocusElementRef = useRef(returnFocusElement)

  useEffect(() => {
    isOpenRef.current = isOpen
  }, [isOpen])

  useEffect(() => {
    returnFocusElementRef.current = returnFocusElement
  }, [returnFocusElement])

  const openCommandBar = useCallback((modeId?: string) => {
    if (!isOpenRef.current) {
      const activeElement = document.activeElement instanceof HTMLElement ? document.activeElement : null
      returnFocusElementRef.current = activeElement
      setReturnFocusElement(activeElement)
    }

    if (modeId !== undefined) setActiveModeId(modeId)
    setActiveIndex(0)
    setSelectionError(null)
    setIsOpen(true)
  }, [])

  const closeCommandBar = useCallback(() => {
    if (!isOpenRef.current) return

    const focusTarget = returnFocusElementRef.current

    setIsOpen(false)
    setQuery('')
    setActiveModeId(null)
    setActiveIndex(0)
    setSelectionError(null)
    setReturnFocusElement(null)
    returnFocusElementRef.current = null

    if (focusTarget?.isConnected) focusTarget.focus()
  }, [])

  const toggleCommandBar = useCallback(
    (modeId?: string) => {
      if (isOpenRef.current) {
        closeCommandBar()
        return
      }

      openCommandBar(modeId)
    },
    [closeCommandBar, openCommandBar]
  )

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!isOpenRef.current || event.key !== 'Escape') return
      event.preventDefault()
      closeCommandBar()
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [closeCommandBar])

  const value = useMemo<CommandBarContextValue>(
    () => ({
      isOpen,
      query,
      setQuery,
      activeModeId,
      setActiveModeId,
      activeIndex,
      setActiveIndex,
      selectionError,
      setSelectionError,
      returnFocusElement,
      openCommandBar,
      closeCommandBar,
      toggleCommandBar
    }),
    [
      activeIndex,
      activeModeId,
      closeCommandBar,
      isOpen,
      openCommandBar,
      query,
      returnFocusElement,
      selectionError,
      toggleCommandBar
    ]
  )

  return <CommandBarContext.Provider value={value}>{children}</CommandBarContext.Provider>
}
