import { Fragment, type CSSProperties, useCallback, useRef, useState } from 'react'
import { Cloud, GripVertical, Server } from 'lucide-react'
import { SegmentedControl } from '@/components/ui/SegmentedControl'
import { MESH_PROVIDERS } from '@/features/reserves/lib/mesh-providers'
import type { ReserveWakePolicySettings } from '@/features/reserves/lib/reserve-types'
import { cn } from '@/lib/utils'

type ReservePolicySettingsFormProps = {
  settings: ReserveWakePolicySettings
  onSettingsChange: (settings: ReserveWakePolicySettings) => void
  providers?: string[]
}

type ProviderOrderListProps = {
  providers?: string[]
  onReorder: (reorderedProviders: string[]) => void
}

function providerDropLaneStyle(): CSSProperties {
  return {
    borderColor: 'color-mix(in oklch, var(--color-accent), transparent 48%)',
    color: 'var(--color-accent)',
    background: 'color-mix(in oklch, var(--color-accent), transparent 90%)'
  }
}

export function ProviderOrderList({ providers, onReorder }: ProviderOrderListProps) {
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [dropTarget, setDropTarget] = useState<number | null>(null)
  const dragCounter = useRef(0)

  const hasProviders = providers != null && providers.length > 0

  function handleDragStart(index: number, event: React.DragEvent) {
    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('text/plain', String(index))
    setDragIndex(index)
  }

  function handleDragOver(index: number, event: React.DragEvent<HTMLElement>) {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'

    const bounds = event.currentTarget.getBoundingClientRect()
    const isAfterMiddle = event.clientY > bounds.top + bounds.height / 2
    setDropTarget(index + (isAfterMiddle ? 1 : 0))
  }

  function handleLaneDragOver(index: number, event: React.DragEvent) {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
    setDropTarget(index)
  }

  function handleDragEnter(event: React.DragEvent) {
    event.preventDefault()
    dragCounter.current += 1
  }

  function handleDragLeave() {
    dragCounter.current -= 1
    if (dragCounter.current <= 0) {
      dragCounter.current = 0
      setDropTarget(null)
    }
  }

  function handleDrop(targetIndex: number, event: React.DragEvent) {
    event.preventDefault()
    dragCounter.current = 0

    if (dragIndex == null || !providers) {
      resetDragState()
      return
    }

    const reordered = [...providers]
    const [moved] = reordered.splice(dragIndex, 1)
    const insertionIndex = dragIndex < targetIndex ? targetIndex - 1 : targetIndex
    reordered.splice(insertionIndex, 0, moved)
    onReorder(reordered)
    resetDragState()
  }

  function handleDragEnd() {
    resetDragState()
  }

  function resetDragState() {
    setDragIndex(null)
    setDropTarget(null)
    dragCounter.current = 0
  }

  if (!hasProviders) {
    return <p className="text-[length:var(--density-type-caption)] text-fg-faint">No provider categories configured.</p>
  }

  const renderDropLane = (index: number) => {
    if (dropTarget !== index || dragIndex === index || dragIndex === index - 1) {
      return null
    }

    return (
      <li
        aria-hidden="true"
        className="grid h-9 min-w-0 place-items-center rounded-[5px] border border-dashed text-[length:var(--density-type-label)] font-mono uppercase tracking-[0.18em]"
        key={`drop-lane-${index}`}
        onDragOver={(event) => handleLaneDragOver(index, event)}
        onDrop={(event) => handleDrop(index, event)}
        style={providerDropLaneStyle()}
      >
        Drop here
      </li>
    )
  }

  return (
    <ul aria-label="Provider priority order" className="w-full space-y-2">
      {providers.map((providerName, index) => {
        const meshProvider = MESH_PROVIDERS.find((p) => p.name === providerName)
        const kindLabel = meshProvider?.kind ?? 'Custom provider'
        const detailLabel = meshProvider ? `${meshProvider.summary} · Priority ${index + 1}` : `Priority ${index + 1}`
        const ProviderIcon = meshProvider?.icon === 'server' ? Server : Cloud

        return (
          <Fragment key={providerName}>
            {renderDropLane(index)}
            <li
              className={cn(
                'flex items-center gap-3 rounded-[var(--radius)] border border-border-soft bg-panel-strong px-3.5 py-3 shadow-[inset_0_1px_0_color-mix(in_oklab,var(--color-fg)_5%,transparent)] transition-[opacity,border-color,background] duration-150 hover:border-[color:color-mix(in_oklab,var(--color-accent)_30%,var(--color-border-soft))] hover:bg-[color:color-mix(in_oklab,var(--color-panel-strong)_86%,var(--color-accent)_14%)]',
                dragIndex === index && 'opacity-40'
              )}
              draggable
              onDragEnd={handleDragEnd}
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDragOver={(event) => handleDragOver(index, event)}
              onDragStart={(event) => handleDragStart(index, event)}
              onDrop={(event) => handleDrop(dropTarget ?? index, event)}
            >
              <span
                className="flex size-8 shrink-0 items-center justify-center rounded-[var(--radius-sm)] border border-[color:color-mix(in_oklab,var(--color-accent)_34%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_10%,transparent)] text-accent"
                aria-hidden="true"
              >
                <ProviderIcon className="size-4" />
              </span>
              <span className="min-w-0 flex-1 space-y-1">
                <span className="flex min-w-0 flex-wrap items-center gap-2">
                  <span className="truncate text-[length:var(--density-type-control-lg)] font-semibold text-foreground">
                    {providerName}
                  </span>
                  <span className="shrink-0 rounded-full border border-border-soft bg-background px-2 py-0.5 text-[10px] uppercase tracking-[0.08em] text-fg-dim">
                    {kindLabel}
                  </span>
                </span>
                <span className="block truncate text-[length:var(--density-type-caption)] text-fg-faint">
                  {detailLabel}
                </span>
              </span>
              <span className="cursor-grab text-fg-faint active:cursor-grabbing" aria-hidden="true">
                <GripVertical className="size-3.5" />
              </span>
            </li>
          </Fragment>
        )
      })}
      {renderDropLane(providers.length)}
    </ul>
  )
}

export function ReservePolicySettingsForm({ settings, onSettingsChange, providers }: ReservePolicySettingsFormProps) {
  const update = useCallback(
    (patch: Partial<ReserveWakePolicySettings>) => {
      onSettingsChange({ ...settings, ...patch })
    },
    [settings, onSettingsChange]
  )

  return (
    <div className="space-y-5">
      <div className="grid gap-x-6 gap-y-4 sm:grid-cols-2">
        <div className="space-y-2">
          <p className="type-label text-fg-faint">Wake mode</p>
          <SegmentedControl
            ariaLabel="Wake mode"
            options={[
              { value: 'true', label: 'Enabled' },
              { value: 'false', label: 'Paused' }
            ]}
            value={String(settings.autoWakeEnabled)}
            onValueChange={(value) => update({ autoWakeEnabled: value === 'true' })}
          />
        </div>

        <div className="space-y-2">
          <p className="type-label text-fg-faint">Utilization threshold</p>
          <SegmentedControl
            ariaLabel="Utilization threshold"
            options={[65, 75, 85].map((threshold) => ({
              value: String(threshold),
              label: `${threshold}%`
            }))}
            value={String(settings.thresholdPercent)}
            onValueChange={(value) => update({ thresholdPercent: Number(value) })}
          />
        </div>

        <div className="space-y-2">
          <p className="type-label text-fg-faint">Sustained for</p>
          <SegmentedControl
            ariaLabel="Sustained for"
            options={[30, 60, 120].map((seconds) => ({
              value: String(seconds),
              label: seconds < 60 ? `${seconds}s` : `${seconds / 60} min`
            }))}
            value={String(settings.sustainedSeconds)}
            onValueChange={(value) => update({ sustainedSeconds: Number(value) })}
          />
        </div>

        <div className="space-y-2">
          <p className="type-label text-fg-faint">Sleep idle reserves</p>
          <SegmentedControl
            ariaLabel="Sleep idle reserves"
            options={[5, 8, 12].map((minutes) => ({
              value: String(minutes),
              label: `${minutes} min`
            }))}
            value={String(settings.idleMinutes)}
            onValueChange={(value) => update({ idleMinutes: Number(value) })}
          />
        </div>
      </div>

      <ProviderOrderList providers={providers} onReorder={(reordered) => update({ providerOrder: reordered })} />
    </div>
  )
}
