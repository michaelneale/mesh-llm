import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { Bug, ChevronRight, ChevronUp, Eye, Network, ScanLine } from 'lucide-react'
import type { MeshVizDotColorScheme } from '@/features/network/lib/mesh-viz-dot-color-schemes'
import { cn } from '@/lib/cn'

export type MeshVizGridMode = 'line' | 'dot'

type MeshVizDebugControlsProps = {
  showPanBounds: boolean
  gridMode: MeshVizGridMode
  dotColorSchemeIndex: number
  dotColorSchemes: readonly MeshVizDotColorScheme[]
  debugNodeCount: number
  isFullscreen: boolean
  onDotColorSchemeChange: (index: number) => void
  onDotColorSchemeNext: () => void
  onGridModeChange: (mode: MeshVizGridMode) => void
  onShowPanBoundsChange: (show: boolean) => void
  onPlayRandomTraffic: () => void
  onPlaySelfTraffic: () => void
  onAddDebugNode: (shortcut: 1 | 2 | 3) => void
  onRemoveDebugNode: (shortcut: 1 | 2 | 3) => void
}

const HOTKEYS = {
  randomTraffic: 'Z',
  selfTraffic: 'X',
  addClientNode: 'Ctrl+1',
  addWorkerNode: 'Ctrl+2',
  addHostNode: 'Ctrl+3',
  removeClientNode: 'Shift+1',
  removeWorkerNode: 'Shift+2',
  removeHostNode: 'Shift+3',
  boundaries: 'Ctrl+B',
  gridStyle: 'Ctrl+G',
  dotColorScheme: 'Ctrl+C',
} as const

const ARIA_HOTKEYS = {
  randomTraffic: 'Z',
  selfTraffic: 'X',
  addClientNode: 'Control+1',
  addWorkerNode: 'Control+2',
  addHostNode: 'Control+3',
  removeClientNode: 'Shift+1',
  removeWorkerNode: 'Shift+2',
  removeHostNode: 'Shift+3',
  boundaries: 'Control+B',
  gridStyle: 'Control+G',
  dotColorScheme: 'Control+C',
} as const

const triggerClassName =
  'surface-panel-translucent shadow-surface-low pointer-events-auto ui-control inline-flex items-center rounded-[var(--radius)] border font-mono font-medium uppercase tracking-[0.08em]'
const menuClassName =
  'shadow-surface-popover pointer-events-auto absolute bottom-full mb-1.5 w-[244px] overflow-visible rounded-[var(--radius)] border border-border bg-panel text-left'
const sectionClassName = 'border-b border-border-soft last:border-b-0'
const sectionHeaderClassName =
  'flex items-center gap-1.5 bg-panel-strong/60 px-2.5 py-1.5 font-mono text-[length:var(--density-type-label)] font-medium uppercase tracking-[0.08em] text-fg-faint'
const itemClassName =
  'flex w-full items-center justify-between gap-2 px-2.5 py-2 text-left text-[length:var(--density-type-caption)] hover:bg-panel-strong focus-visible:bg-panel-strong focus-visible:outline-none'
const nestedMenuClassName =
  'shadow-surface-popover absolute left-full top-0 z-30 m-0 ml-1.5 min-w-0 w-[214px] overflow-hidden rounded-[var(--radius)] border border-border bg-panel py-1 text-left'
const kbdClassName =
  'rounded-[3px] border border-border bg-background px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.06em] text-fg-faint'

function ShortcutHint({ children }: { children: ReactNode }) {
  return (
    <kbd className={kbdClassName}>
      {children}
    </kbd>
  )
}

function DebugAction({
  ariaKeyShortcuts,
  children,
  onClick,
  disabled,
  pressed,
  role,
  shortcut,
}: {
  ariaKeyShortcuts: string
  children: ReactNode
  disabled?: boolean
  onClick: () => void
  pressed?: boolean
  role?: 'menuitem'
  shortcut: string
}) {
  return (
    <button
      aria-keyshortcuts={ariaKeyShortcuts}
      aria-pressed={pressed}
      className={cn(itemClassName, disabled && 'cursor-not-allowed opacity-45 hover:bg-transparent')}
      disabled={disabled}
      onClick={onClick}
      role={role}
      type="button"
    >
      <span>{children}</span>
      <ShortcutHint>{shortcut}</ShortcutHint>
    </button>
  )
}

function DotThemeSwatch({
  index,
  isSelected,
  onSelect,
  scheme,
}: {
  index: number
  isSelected: boolean
  onSelect: () => void
  scheme: MeshVizDotColorScheme
}) {
  return (
    <button
      aria-label={`Dot theme ${index + 1}: ${scheme.label}`}
      aria-pressed={isSelected}
      className={cn(
        'group inline-flex min-w-0 items-center justify-center gap-1 rounded-[4px] border p-1 outline-none transition-[background-color,border-color,box-shadow] active:translate-y-px',
        'focus-visible:border-accent focus-visible:shadow-[var(--shadow-focus-accent)]',
        isSelected
          ? 'border-transparent bg-panel-strong/45'
          : 'border-transparent hover:bg-panel-strong/55',
      )}
      data-testid={`mesh-viz-dot-theme-${index + 1}-swatch`}
      onClick={onSelect}
      type="button"
    >
      <span
        className={cn(
          'font-mono text-[10px] uppercase tracking-[0.08em] transition-opacity',
          isSelected ? 'text-foreground opacity-100' : 'text-fg-faint opacity-55 group-hover:opacity-85',
        )}
        data-testid={`mesh-viz-dot-theme-${index + 1}-index`}
      >
        {index + 1}
      </span>
      <span
        aria-hidden="true"
        className="grid grid-cols-4 overflow-hidden rounded-[3px] border border-border bg-background"
      >
        {scheme.nodeColors.map((color, colorIndex) => (
          <span
            aria-hidden="true"
            className={cn(
              'size-3.5 transition-opacity',
              colorIndex > 0 && 'border-l border-border',
              isSelected ? 'opacity-100' : 'opacity-45 group-hover:opacity-75',
            )}
            data-color-value={color}
            data-testid={`mesh-viz-dot-theme-${index + 1}-color-${colorIndex + 1}`}
            key={color}
            style={{ backgroundColor: color }}
          />
        ))}
      </span>
    </button>
  )
}

export function MeshVizDebugControls({
  showPanBounds,
  gridMode,
  dotColorSchemeIndex,
  dotColorSchemes,
  debugNodeCount,
  isFullscreen,
  onDotColorSchemeChange,
  onDotColorSchemeNext,
  onGridModeChange,
  onShowPanBoundsChange,
  onPlayRandomTraffic,
  onPlaySelfTraffic,
  onAddDebugNode,
  onRemoveDebugNode,
}: MeshVizDebugControlsProps) {
  const debugMenuId = useId()
  const addNodesSubmenuId = useId()
  const removeNodesSubmenuId = useId()
  const controlsRef = useRef<HTMLFieldSetElement>(null)
  const [isOpen, setIsOpen] = useState(false)
  const [isAddNodesSubmenuOpen, setIsAddNodesSubmenuOpen] = useState(false)
  const [isRemoveNodesSubmenuOpen, setIsRemoveNodesSubmenuOpen] = useState(false)
  const triggerIconClassName = isFullscreen ? 'size-6' : 'size-3'

  useEffect(() => {
    if (!isOpen) {
      return undefined
    }

    const closeMenu = (event: PointerEvent) => {
      if (event.target instanceof Node && controlsRef.current?.contains(event.target)) {
        return
      }

      setIsOpen(false)
      setIsAddNodesSubmenuOpen(false)
      setIsRemoveNodesSubmenuOpen(false)
    }

    window.addEventListener('pointerdown', closeMenu)

    return () => window.removeEventListener('pointerdown', closeMenu)
  }, [isOpen])

  const runMenuAction = (action: () => void) => {
    action()
    setIsOpen(false)
    setIsAddNodesSubmenuOpen(false)
    setIsRemoveNodesSubmenuOpen(false)
  }

  const shouldShowRemoveNodesSubmenu = isRemoveNodesSubmenuOpen && debugNodeCount > 0

  const toggleMenu = () => {
    setIsOpen((current) => {
      if (current) {
        setIsAddNodesSubmenuOpen(false)
        setIsRemoveNodesSubmenuOpen(false)
      }

      return !current
    })
  }

  return (
    <fieldset
      ref={controlsRef}
      className="pointer-events-none absolute bottom-3 left-3 right-14 z-20 flex flex-row flex-wrap items-center gap-1.5"
    >
      <legend className="sr-only">Mesh debug controls</legend>
      <div className="relative">
        <button
          aria-controls={isOpen ? debugMenuId : undefined}
          aria-expanded={isOpen}
          className={cn(
            triggerClassName,
            isFullscreen
              ? 'gap-3 px-5 py-2 text-[length:var(--density-type-caption)]'
              : 'gap-1.5 px-2.5 py-1 text-[length:var(--density-type-annotation)]',
          )}
          onClick={toggleMenu}
          onPointerDown={(event) => event.stopPropagation()}
          type="button"
        >
          <Bug aria-hidden="true" className={triggerIconClassName} strokeWidth={1.9} />
          Debug
          <ChevronUp aria-hidden="true" className={cn(triggerIconClassName, 'transition-transform', isOpen && 'rotate-180')} strokeWidth={1.9} />
        </button>

        {isOpen ? (
          <div
            className={menuClassName}
            id={debugMenuId}
            onPointerDown={(event) => event.stopPropagation()}
          >
            <section aria-label="Traffic debug actions" className={sectionClassName}>
              <div className={sectionHeaderClassName}>
                <Network aria-hidden="true" className="size-3 text-accent" strokeWidth={1.8} />
                Traffic
              </div>
              <DebugAction
                ariaKeyShortcuts={ARIA_HOTKEYS.randomTraffic}
                onClick={() => runMenuAction(onPlayRandomTraffic)}
                shortcut={HOTKEYS.randomTraffic}
              >
                Random traffic
              </DebugAction>
              <DebugAction
                ariaKeyShortcuts={ARIA_HOTKEYS.selfTraffic}
                onClick={() => runMenuAction(onPlaySelfTraffic)}
                shortcut={HOTKEYS.selfTraffic}
              >
                Self traffic
              </DebugAction>
            </section>

            <section aria-label="Debug node actions" className={sectionClassName}>
              <div className={sectionHeaderClassName}>
                <Bug aria-hidden="true" className="size-3 text-accent" strokeWidth={1.8} />
                Debug nodes
              </div>

              <div className="relative">
                <button
                  aria-controls={addNodesSubmenuId}
                  aria-expanded={isAddNodesSubmenuOpen}
                  aria-haspopup="menu"
                  className={itemClassName}
                  onClick={() => {
                    setIsAddNodesSubmenuOpen(true)
                    setIsRemoveNodesSubmenuOpen(false)
                  }}
                  onPointerEnter={() => {
                    setIsAddNodesSubmenuOpen(true)
                    setIsRemoveNodesSubmenuOpen(false)
                  }}
                  type="button"
                >
                  <span>Add nodes</span>
                  <ChevronRight aria-hidden="true" className="size-3" strokeWidth={1.9} />
                </button>

                {isAddNodesSubmenuOpen ? (
                  <div
                    aria-label="Add debug nodes"
                    className={nestedMenuClassName}
                    id={addNodesSubmenuId}
                    onPointerDown={(event) => event.stopPropagation()}
                    role="menu"
                  >
                    <DebugAction
                      ariaKeyShortcuts={ARIA_HOTKEYS.addClientNode}
                      onClick={() => runMenuAction(() => onAddDebugNode(1))}
                      role="menuitem"
                      shortcut={HOTKEYS.addClientNode}
                    >
                      Debug client
                    </DebugAction>
                    <DebugAction
                      ariaKeyShortcuts={ARIA_HOTKEYS.addWorkerNode}
                      onClick={() => runMenuAction(() => onAddDebugNode(2))}
                      role="menuitem"
                      shortcut={HOTKEYS.addWorkerNode}
                    >
                      Debug worker
                    </DebugAction>
                    <DebugAction
                      ariaKeyShortcuts={ARIA_HOTKEYS.addHostNode}
                      onClick={() => runMenuAction(() => onAddDebugNode(3))}
                      role="menuitem"
                      shortcut={HOTKEYS.addHostNode}
                    >
                      Debug host
                    </DebugAction>
                  </div>
                ) : null}
              </div>

              <div className="relative">
                <button
                  aria-controls={removeNodesSubmenuId}
                  aria-expanded={shouldShowRemoveNodesSubmenu}
                  aria-haspopup="menu"
                  className={cn(itemClassName, debugNodeCount === 0 && 'cursor-not-allowed opacity-45 hover:bg-transparent')}
                  disabled={debugNodeCount === 0}
                  onClick={() => {
                    setIsAddNodesSubmenuOpen(false)
                    setIsRemoveNodesSubmenuOpen(true)
                  }}
                  onPointerEnter={() => {
                    if (debugNodeCount === 0) {
                      return
                    }

                    setIsAddNodesSubmenuOpen(false)
                    setIsRemoveNodesSubmenuOpen(true)
                  }}
                  type="button"
                >
                  <span>Remove nodes</span>
                  <ChevronRight aria-hidden="true" className="size-3" strokeWidth={1.9} />
                </button>

                {shouldShowRemoveNodesSubmenu ? (
                  <div
                    aria-label="Remove debug nodes"
                    className={nestedMenuClassName}
                    id={removeNodesSubmenuId}
                    onPointerDown={(event) => event.stopPropagation()}
                    role="menu"
                  >
                    <DebugAction
                      ariaKeyShortcuts={ARIA_HOTKEYS.removeClientNode}
                      onClick={() => runMenuAction(() => onRemoveDebugNode(1))}
                      role="menuitem"
                      shortcut={HOTKEYS.removeClientNode}
                    >
                      Debug client
                    </DebugAction>
                    <DebugAction
                      ariaKeyShortcuts={ARIA_HOTKEYS.removeWorkerNode}
                      onClick={() => runMenuAction(() => onRemoveDebugNode(2))}
                      role="menuitem"
                      shortcut={HOTKEYS.removeWorkerNode}
                    >
                      Debug worker
                    </DebugAction>
                    <DebugAction
                      ariaKeyShortcuts={ARIA_HOTKEYS.removeHostNode}
                      onClick={() => runMenuAction(() => onRemoveDebugNode(3))}
                      role="menuitem"
                      shortcut={HOTKEYS.removeHostNode}
                    >
                      Debug host
                    </DebugAction>
                  </div>
                ) : null}
              </div>
            </section>

            <section aria-label="Visual debug actions" className={sectionClassName}>
              <div className={sectionHeaderClassName}>
                <Eye aria-hidden="true" className="size-3 text-accent" strokeWidth={1.8} />
                Visuals
              </div>
              <DebugAction
                ariaKeyShortcuts={ARIA_HOTKEYS.boundaries}
                onClick={() => runMenuAction(() => onShowPanBoundsChange(!showPanBounds))}
                pressed={showPanBounds}
                shortcut={HOTKEYS.boundaries}
              >
                Debug boundaries
              </DebugAction>
              <DebugAction
                ariaKeyShortcuts={ARIA_HOTKEYS.gridStyle}
                onClick={() => runMenuAction(() => onGridModeChange(gridMode === 'line' ? 'dot' : 'line'))}
                pressed={gridMode === 'dot'}
                shortcut={HOTKEYS.gridStyle}
              >
                Toggle Grid Style ({gridMode === 'line' ? 'Lines' : 'DOTS'})
              </DebugAction>
              <fieldset
                aria-keyshortcuts={ARIA_HOTKEYS.dotColorScheme}
                aria-label="Dot theme options"
                className="px-2.5 py-2"
              >
                <legend className="sr-only">Dot theme options</legend>
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <span
                    className="text-[length:var(--density-type-caption)] text-foreground"
                    data-testid="mesh-viz-dot-theme-label"
                  >
                    Dot Theme
                  </span>
                  <button
                    aria-label="Cycle dot theme"
                    aria-keyshortcuts={ARIA_HOTKEYS.dotColorScheme}
                    className="outline-none focus-visible:rounded-[3px] focus-visible:shadow-[var(--shadow-focus-accent)]"
                    onClick={() => runMenuAction(onDotColorSchemeNext)}
                    type="button"
                  >
                    <ShortcutHint>{HOTKEYS.dotColorScheme}</ShortcutHint>
                  </button>
                </div>
                <div className="grid grid-cols-3 gap-1.5">
                  {dotColorSchemes.map((scheme, index) => (
                    <DotThemeSwatch
                      index={index}
                      isSelected={index === dotColorSchemeIndex}
                      key={scheme.label}
                      onSelect={() => runMenuAction(() => onDotColorSchemeChange(index))}
                      scheme={scheme}
                    />
                  ))}
                </div>
              </fieldset>
            </section>
          </div>
        ) : null}
      </div>

      {debugNodeCount > 0 ? (
        <span className="pointer-events-auto inline-flex items-center gap-1.5 rounded-full border border-border bg-panel/95 px-2 py-px font-mono text-[length:var(--density-type-label)] uppercase tracking-[0.08em] text-fg-faint">
          <ScanLine aria-hidden="true" className="size-3 text-accent" strokeWidth={1.8} />
          {debugNodeCount} debug
        </span>
      ) : null}
    </fieldset>
  )
}
