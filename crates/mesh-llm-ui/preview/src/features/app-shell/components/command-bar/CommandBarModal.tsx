import { Search, X } from 'lucide-react'
import {
  type ElementType,
  type HTMLAttributes,
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef
} from 'react'
import { cn } from '@/lib/cn'
import { CommandBarResultsList } from './CommandBarResultsList'
import { Dialog, DialogContent, DialogDescription, DialogTitle, Input } from './command-bar-primitives'
import {
  filterCommandBarResults,
  getCommandBarErrorMessage,
  getCommandBarOptionId,
  resolveCommandBarActiveIndex
} from './command-bar-helpers'
import type {
  CommandBarBehavior,
  CommandBarMode,
  CommandBarNormalizedResult,
  CommandBarResultContainerProps
} from './command-bar-types'
import { useCommandBarAsyncResults } from './useCommandBarAsyncResults'
import { useCommandBar } from './useCommandBar'
import { useCommandBarNavigation } from './useCommandBarNavigation'

export interface CommandBarModalProps<T> {
  modes: readonly CommandBarMode<T>[]
  behavior: CommandBarBehavior
  defaultModeId?: string | null
  fallbackIcon?: ElementType
  title?: string
  description?: string
  placeholder?: string
  emptyMessage?: string
  interstitial?: ReactNode
  onClose?: () => void
  overlayClassName?: string
  contentClassName?: string
}

function DefaultResultContainer<T>({ children }: CommandBarResultContainerProps<T>) {
  return <div className="divide-y divide-border-soft">{children}</div>
}

export function CommandBarModal<T>({
  modes,
  behavior,
  defaultModeId,
  fallbackIcon = Search,
  title = 'Command bar',
  description = 'Search available results.',
  placeholder,
  emptyMessage = 'No matching results.',
  interstitial,
  onClose,
  overlayClassName,
  contentClassName
}: CommandBarModalProps<T>) {
  const {
    activeIndex,
    activeModeId,
    closeCommandBar,
    isOpen,
    query,
    returnFocusElement,
    selectionError,
    setActiveIndex,
    setActiveModeId,
    setQuery,
    setSelectionError
  } = useCommandBar()
  const inputRef = useRef<HTMLInputElement>(null)
  const listboxId = useId()
  const modeById = useMemo(() => new Map(modes.map((mode) => [mode.id, mode] as const)), [modes])
  const shortcutPrefix = 'Ctrl+'
  const resolvedDistinctModeId = useMemo(() => {
    if (behavior !== 'distinct') return null
    if (activeModeId && modeById.has(activeModeId)) return activeModeId
    if (defaultModeId && modeById.has(defaultModeId)) return defaultModeId
    return modes[0]?.id ?? null
  }, [activeModeId, behavior, defaultModeId, modeById, modes])
  const { asyncErrorMessage, isLoading, resolvedModes } = useCommandBarAsyncResults({
    behavior,
    isOpen,
    modeById,
    modes,
    query,
    resolvedDistinctModeId
  })
  const activeMode = resolvedDistinctModeId ? (modeById.get(resolvedDistinctModeId) ?? null) : null
  const LeadingIcon = activeMode?.leadingIcon ?? fallbackIcon
  const results = useMemo(
    () =>
      filterCommandBarResults({
        modes: resolvedModes,
        behavior,
        query,
        activeModeId: resolvedDistinctModeId,
        defaultModeId
      }),
    [behavior, defaultModeId, query, resolvedDistinctModeId, resolvedModes]
  )
  const activeOptionId =
    activeIndex >= 0 ? getCommandBarOptionId(listboxId, results[activeIndex]?.compositeKey ?? '') : undefined
  const ResultContainer =
    behavior === 'distinct' ? (activeMode?.ResultContainer ?? DefaultResultContainer) : DefaultResultContainer
  const showModeChips = behavior === 'distinct' && modes.length > 1
  const showModeShortcutHint = behavior === 'distinct' && modes.length > 1
  const modeShortcutHint = `${shortcutPrefix}1-${modes.length}`
  const inputPlaceholder = placeholder ?? (activeMode ? `Search ${activeMode.label.toLowerCase()}` : 'Search')
  const listProps = useMemo<HTMLAttributes<HTMLDivElement>>(
    () => ({
      id: listboxId,
      role: 'listbox',
      tabIndex: 0,
      'aria-label': 'Command bar results',
      'aria-activedescendant': activeOptionId,
      className: 'overflow-hidden rounded-[var(--radius-lg)] border border-border bg-background'
    }),
    [activeOptionId, listboxId]
  )
  const requestClose = useCallback(() => {
    closeCommandBar()
    onClose?.()
  }, [closeCommandBar, onClose])

  const handleModeSwitch = useCallback(
    (modeId: string) => {
      setSelectionError(null)
      setActiveModeId(modeId)
    },
    [setActiveModeId, setSelectionError]
  )

  const handleSelectResult = useCallback(
    async (result: CommandBarNormalizedResult<T>) => {
      const mode = modeById.get(result.modeId)
      if (!mode) return

      setSelectionError(null)

      try {
        const shouldClose = await Promise.resolve(mode.onSelect(result.item))
        if (shouldClose !== false) requestClose()
      } catch (error) {
        setSelectionError(getCommandBarErrorMessage(error))
      }
    },
    [modeById, requestClose, setSelectionError]
  )
  const { handleKeyDown, registerOptionElement } = useCommandBarNavigation({
    activeIndex,
    behavior,
    isOpen,
    listboxId,
    modes,
    onModeSwitch: handleModeSwitch,
    onSelectResult: handleSelectResult,
    results,
    setActiveIndex
  })

  useEffect(() => {
    if (!isOpen || behavior !== 'distinct' || !resolvedDistinctModeId || activeModeId === resolvedDistinctModeId) return
    setActiveModeId(resolvedDistinctModeId)
  }, [activeModeId, behavior, isOpen, resolvedDistinctModeId, setActiveModeId])

  useEffect(() => {
    if (!isOpen) return
    setActiveIndex(resolveCommandBarActiveIndex(results))
  }, [isOpen, results, setActiveIndex])

  const showLoadingState = isLoading && results.length === 0
  const showErrorState = !showLoadingState && Boolean(asyncErrorMessage) && results.length === 0

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) requestClose()
      }}
    >
      <DialogContent
        className={cn('gap-0 overflow-hidden p-0', contentClassName)}
        overlayClassName={overlayClassName}
        onCloseAutoFocus={(event) => event.preventDefault()}
        onEscapeKeyDown={(event) => {
          event.preventDefault()
          const focusTarget = returnFocusElement
          requestClose()
          if (focusTarget?.isConnected) {
            queueMicrotask(() => {
              if (focusTarget.isConnected) focusTarget.focus()
            })
          }
        }}
        onKeyDown={handleKeyDown}
        onOpenAutoFocus={(event) => {
          event.preventDefault()
          inputRef.current?.focus()
        }}
      >
        <DialogTitle className="sr-only">{title}</DialogTitle>
        <DialogDescription className="sr-only">{description}</DialogDescription>

        <div data-testid="command-bar-header" className="border-b border-border-soft bg-panel px-3.5 py-3">
          <div className="mb-2 flex items-center gap-2">
            <div className="min-w-0 flex-1">
              <div className="text-[length:var(--density-type-body)] font-semibold text-foreground">{title}</div>
            </div>
            {showModeShortcutHint ? (
              <div className="font-mono text-[length:var(--density-type-label)] text-muted-foreground">
                {modeShortcutHint}
              </div>
            ) : null}
            <button
              type="button"
              aria-label="Close"
              className="ui-control inline-flex size-7 shrink-0 items-center justify-center rounded-[var(--radius)] border"
              onClick={requestClose}
            >
              <X aria-hidden="true" className="size-3.5" strokeWidth={1.8} />
            </button>
          </div>

          <div className="relative min-w-0 flex-1">
            <span className="pointer-events-none absolute left-3 top-1/2 z-10 -translate-y-1/2 text-muted-foreground">
              <LeadingIcon className="size-4" aria-hidden="true" />
            </span>
            <Input
              ref={inputRef}
              value={query}
              onChange={(event) => {
                setSelectionError(null)
                setQuery(event.target.value)
              }}
              aria-label="Command bar search"
              aria-controls={listboxId}
              className="h-10 rounded-[var(--radius)] border-border bg-background pl-9 pr-10"
              placeholder={inputPlaceholder}
            />
            {query.length > 0 ? (
              <button
                type="button"
                aria-label="Clear search"
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => {
                  setSelectionError(null)
                  setQuery('')
                  inputRef.current?.focus()
                }}
                className="absolute right-2.5 top-1/2 z-10 inline-flex size-6 -translate-y-1/2 items-center justify-center rounded-[var(--radius)] text-muted-foreground hover:bg-panel-strong hover:text-foreground"
              >
                <X className="size-3.5" aria-hidden="true" />
              </button>
            ) : null}
          </div>

          {showModeChips ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {modes.map((mode, index) => {
                const isSelected = mode.id === resolvedDistinctModeId

                return (
                  <button
                    key={mode.id}
                    type="button"
                    aria-pressed={isSelected}
                    onClick={() => handleModeSwitch(mode.id)}
                    className="ui-control inline-flex h-8 items-center gap-2 rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-caption)] font-medium"
                  >
                    <span>{mode.label}</span>
                    <span className="font-mono text-[length:var(--density-type-annotation)] uppercase tracking-[0.08em] text-muted-foreground">
                      Ctrl+{index + 1}
                    </span>
                  </button>
                )
              })}
            </div>
          ) : null}
        </div>

        {interstitial ? (
          <div
            data-testid="command-bar-interstitial"
            className="border-b border-border-soft bg-background px-3.5 py-2.5"
          >
            {interstitial}
          </div>
        ) : null}

        <CommandBarResultsList
          activeIndex={activeIndex}
          activeModeId={behavior === 'distinct' ? resolvedDistinctModeId : activeModeId}
          asyncErrorMessage={asyncErrorMessage}
          behavior={behavior}
          emptyMessage={emptyMessage}
          listProps={listProps}
          listboxId={listboxId}
          modeById={modeById}
          onRegisterOptionElement={registerOptionElement}
          onSelectResult={(result) => {
            void handleSelectResult(result)
          }}
          onSetActiveIndex={setActiveIndex}
          query={query}
          results={results}
          ResultContainer={ResultContainer}
          selectionError={selectionError}
          showErrorState={showErrorState}
          showLoadingState={showLoadingState}
        />
      </DialogContent>
    </Dialog>
  )
}
