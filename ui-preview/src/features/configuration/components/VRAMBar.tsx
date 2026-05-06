import type { Dispatch, DragEvent, PointerEvent, SetStateAction } from 'react'
import {
  canFitModelInContainer,
  containerAssigns,
  containerUsedGB,
  contextGB,
  findModel,
  kvGB,
  modelFamilyColorKey
} from '@/features/configuration/lib/config-math'
import { createAssignmentId } from '@/features/configuration/lib/assignment-ids'
import { reservedVramSelectionId } from '@/features/configuration/lib/selection'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'

type VRAMBarLabel = { prefix: string; main: string; sub?: string }

type VRAMBarProps = {
  node: ConfigNode
  label: VRAMBarLabel
  totalGB: number
  reservedGB?: number
  containerIdx: number
  assigns: ConfigAssign[]
  models?: ConfigModel[]
  selectedId?: string | null
  selectedContainer?: boolean
  onPick: (id: string | null) => void
  onSelectContainer: () => void
  setAssigns: Dispatch<SetStateAction<ConfigAssign[]>>
  dragOver: string | null
  setDragOver: (key: string | null) => void
  dense?: boolean
  interactiveTabIndex?: number
  readOnly?: boolean
}

const SOURCE_CONTAINER_MIME_PREFIX = 'application/x-mesh-source-container-'
const MODEL_MIME_PREFIX = 'application/x-mesh-model-'
const ASSIGN_MIME_PREFIX = 'application/x-mesh-assign-'

function formatGB(value: number) {
  return Number.isInteger(value) ? value.toString() : value.toFixed(1)
}

export function VRAMBar({
  node,
  label,
  totalGB,
  reservedGB = 0,
  containerIdx,
  assigns,
  models,
  selectedId,
  selectedContainer = false,
  onPick,
  onSelectContainer,
  setAssigns,
  dragOver,
  setDragOver,
  dense = false,
  interactiveTabIndex,
  readOnly = false
}: VRAMBarProps) {
  const key = `${node.id}-${containerIdx}`
  const sourceContainerType = `${SOURCE_CONTAINER_MIME_PREFIX}${key}`
  const used = containerUsedGB(assigns, node.id, containerIdx, models)
  const free = Math.max(0, totalGB - used - reservedGB)
  const current = containerAssigns(assigns, node.id, containerIdx)
  const safeTotal = totalGB || 1
  const usableGB = Math.max(0, totalGB - reservedGB)
  const safeUsable = usableGB || 1
  const usableColumnGB = Math.max(usableGB, 0.0001)
  const reservedColumnGB = Math.max(reservedGB, 0.0001)
  const reservedPct = reservedGB > 0 ? Math.min(100, (reservedGB / safeTotal) * 100) : 0
  const barHeight = 48
  const invalidDragKey = `${key}:no-fit`
  const validDropActive = !readOnly && dragOver === key
  const invalidDropActive = !readOnly && dragOver === invalidDragKey
  const reservedSelectionKey = reservedVramSelectionId(node.id, containerIdx)
  const reservedSelected = selectedId === reservedSelectionKey

  const isWithinBar = (event: DragEvent<HTMLElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    return (
      event.clientX >= rect.left &&
      event.clientX <= rect.right &&
      event.clientY >= rect.top &&
      event.clientY <= rect.bottom
    )
  }

  const isReservedLaneEvent = (event: DragEvent<HTMLElement>) => {
    const target = event.target
    return target instanceof Element && Boolean(target.closest('[data-vram-reserved-lane="true"]'))
  }

  const getTypedDataId = (types: string[], prefix: string) =>
    types.find((type) => type.startsWith(prefix))?.slice(prefix.length) ?? ''

  const getDropIntent = (event: DragEvent<HTMLElement>) => {
    if (readOnly) return { supported: false, fits: false, effect: 'none' as const }

    const types = Array.from(event.dataTransfer.types)
    const modelId = event.dataTransfer.getData('text/model') || getTypedDataId(types, MODEL_MIME_PREFIX)
    if (modelId) {
      const model = findModel(modelId, models)
      return {
        supported: true,
        fits: Boolean(model && canFitModelInContainer(model, node, assigns, containerIdx, 4096, undefined, models)),
        effect: 'copy' as const
      }
    }

    if (types.includes('text/model')) return { supported: true, fits: true, effect: 'copy' as const }

    if (!types.includes('text/assign-id') && !types.some((type) => type.startsWith(ASSIGN_MIME_PREFIX))) {
      return { supported: false, fits: false, effect: 'none' as const }
    }

    const sourceType = types.find((type) => type.startsWith(SOURCE_CONTAINER_MIME_PREFIX))
    if (sourceType === sourceContainerType) return { supported: false, fits: false, effect: 'none' as const }

    const assignId = event.dataTransfer.getData('text/assign-id') || getTypedDataId(types, ASSIGN_MIME_PREFIX)
    if (!assignId) return { supported: true, fits: true, effect: 'move' as const }

    const sourceNodeId = event.dataTransfer.getData('text/source-node')
    const sourceContainerIdx = Number(event.dataTransfer.getData('text/source-container'))
    if (
      sourceNodeId &&
      Number.isFinite(sourceContainerIdx) &&
      sourceNodeId === node.id &&
      sourceContainerIdx === containerIdx
    ) {
      return { supported: false, fits: false, effect: 'none' as const }
    }

    const existing = assigns.find((assign) => assign.id === assignId)
    if (!existing || (existing.nodeId === node.id && existing.containerIdx === containerIdx)) {
      return { supported: false, fits: false, effect: 'none' as const }
    }

    const model = findModel(existing.modelId, models)
    return {
      supported: true,
      fits: Boolean(
        model && canFitModelInContainer(model, node, assigns, containerIdx, existing.ctx, existing.id, models)
      ),
      effect: 'move' as const
    }
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragOver(null)
    if (readOnly) return
    if (isReservedLaneEvent(event)) return

    const modelId = event.dataTransfer.getData('text/model')
    const assignId = event.dataTransfer.getData('text/assign-id')
    if (modelId) {
      const model = findModel(modelId, models)
      if (!model || !canFitModelInContainer(model, node, assigns, containerIdx, 4096, undefined, models)) return

      const id = createAssignmentId(assigns)
      setAssigns((items) => [...items, { id, modelId, nodeId: node.id, containerIdx, ctx: 4096 }])
      onPick(id)
      return
    }
    if (assignId) {
      setAssigns((items) => {
        const existing = items.find((assign) => assign.id === assignId)
        if (!existing || (existing.nodeId === node.id && existing.containerIdx === containerIdx)) return items
        const model = findModel(existing.modelId, models)
        if (!model || !canFitModelInContainer(model, node, items, containerIdx, existing.ctx, existing.id, models))
          return items
        return items.map((assign) => (assign.id === assignId ? { ...assign, nodeId: node.id, containerIdx } : assign))
      })
    }
  }

  const onContainerPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (readOnly) return

    onSelectContainer()
    if (event.target instanceof Element && event.target.closest('button')) return
    onPick(null)
  }

  return (
    <div
      className={`rounded-[8px] border px-3.5 py-2.5 transition-[background-color,border-color] ${selectedContainer ? 'border-[color:color-mix(in_oklab,var(--color-accent)_36%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_4%,var(--color-panel))]' : 'border-border-soft bg-background'}`}
      data-config-selection-area="true"
      data-config-container-selected={selectedContainer ? 'true' : undefined}
      onPointerDown={onContainerPointerDown}
    >
      <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1.5">
        <div className="min-w-0">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <span className="font-mono text-[length:var(--density-type-annotation)] uppercase tracking-[0.18em] text-fg-faint">
              {label.prefix}
            </span>
            <span
              className={
                dense
                  ? 'text-[length:var(--density-type-caption-lg)] font-semibold leading-none'
                  : 'text-[length:var(--density-type-control-lg)] font-semibold leading-none'
              }
            >
              {label.main}
            </span>
            {label.sub ? (
              <span className="text-[length:var(--density-type-caption)] text-fg-faint">{label.sub}</span>
            ) : null}
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-x-1 font-mono text-[length:var(--density-type-label)] text-fg-faint">
          <span>{formatGB(totalGB)} GB</span>
          <span aria-hidden>·</span>
          <span>used</span>
          <span className="text-foreground">{formatGB(used)}</span>
          <span aria-hidden>·</span>
          <span>free</span>
          <span className="text-foreground">{formatGB(free)}</span>
          <span aria-hidden>·</span>
          <span>reserved</span>
          <span className="text-foreground">{formatGB(reservedGB)}</span>
        </div>
      </div>
      <section
        aria-label={`${label.main} capacity ${formatGB(used)} of ${formatGB(totalGB)} GB, free ${formatGB(free)} GB, reserved ${formatGB(reservedGB)} GB`}
        className="mt-2 grid overflow-hidden border border-border-soft bg-panel p-0.5"
        onDragEnter={(event) => {
          if (readOnly) return

          if (isReservedLaneEvent(event)) {
            setDragOver(invalidDragKey)
            return
          }
          const intent = getDropIntent(event)
          if (!intent.supported) {
            setDragOver(null)
            return
          }
          setDragOver(intent.fits ? key : invalidDragKey)
        }}
        onDragOver={(event) => {
          if (readOnly) {
            event.dataTransfer.dropEffect = 'none'
            return
          }

          if (isReservedLaneEvent(event)) {
            event.dataTransfer.dropEffect = 'none'
            setDragOver(invalidDragKey)
            return
          }
          const intent = getDropIntent(event)
          if (!intent.supported) {
            event.dataTransfer.dropEffect = 'none'
            setDragOver(null)
            return
          }
          event.preventDefault()
          if (!intent.fits) {
            event.dataTransfer.dropEffect = 'none'
            setDragOver(invalidDragKey)
            return
          }
          event.dataTransfer.dropEffect = intent.effect
          setDragOver(key)
        }}
        onDragLeave={(event) => {
          if (readOnly) return

          if (isWithinBar(event)) return
          setDragOver(null)
        }}
        onDrop={onDrop}
        style={{
          borderRadius: 7,
          height: barHeight,
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
        }}
      >
        {reservedGB > 0 ? (
          <button
            aria-label={`System reserved space, ${formatGB(reservedGB)} GB reserved on ${label.main}`}
            aria-pressed={reservedSelected}
            className="grid shrink-0 cursor-pointer place-items-center rounded-[5px] border border-border-soft bg-background px-1.5 font-mono text-[length:var(--density-type-annotation)] uppercase tracking-[0.16em] text-fg-faint outline-none transition-[border-color,box-shadow,color] focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
            data-config-selection-area="true"
            data-vram-reserved-lane="true"
            disabled={readOnly}
            onClick={() => onPick(reservedSelectionKey)}
            style={{
              backgroundImage:
                'repeating-linear-gradient(45deg, color-mix(in oklch, var(--color-foreground), transparent 88%) 0 4px, transparent 4px 8px)',
              borderColor: reservedSelected
                ? 'color-mix(in oklch, var(--color-accent), var(--color-border-soft) 42%)'
                : undefined,
              boxShadow: reservedSelected ? 'var(--shadow-config-reserved-selected)' : undefined,
              color: reservedSelected ? 'var(--color-foreground)' : undefined
            }}
            tabIndex={interactiveTabIndex}
            title={`Reserved ${formatGB(reservedGB)} GB`}
            type="button"
          >
            {reservedPct >= (dense ? 13 : 9) ? 'RSV' : ''}
          </button>
        ) : null}
        <div className="flex min-w-0 overflow-hidden rounded-[5px]" style={{ gap: 3 }}>
          {current.map((assign) => {
            const model = findModel(assign.modelId, models)
            const weightsGB = model?.sizeGB ?? 1
            const exactCacheGB = model ? contextGB(model, assign.ctx) : 0
            const displayCacheGB = model ? kvGB(model, assign.ctx) : 0
            const minWeightsPct = dense ? 10 : 8
            const weightsPct = (weightsGB / safeUsable) * 100
            const exactCachePct = (exactCacheGB / safeUsable) * 100
            const visualWidthPct = Math.max(minWeightsPct, weightsPct) + exactCachePct
            const cachePct = visualWidthPct > 0 ? (exactCachePct / visualWidthPct) * 100 : 0
            const selected = assign.id === selectedId
            const modelName = model?.name ?? assign.modelId
            const detailLabel = `${formatGB(weightsGB)} GB · ${assign.ctx.toLocaleString()} ctx (${formatGB(displayCacheGB)} GB)`

            return (
              <button
                key={assign.id}
                aria-label={`${modelName}, ${formatGB(weightsGB)} GB weights, ${formatGB(displayCacheGB)} GB context cache${readOnly ? ', read-only' : ', drag to move'}`}
                aria-pressed={selected}
                className={`relative h-full min-w-0 shrink-0 overflow-hidden rounded-[5px] px-2 text-left outline-none transition-[box-shadow] focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent ${readOnly ? 'cursor-not-allowed opacity-85' : ''}`}
                data-config-model-chip="true"
                data-model-family-color={modelFamilyColorKey(model)}
                data-model-selection-area="true"
                disabled={readOnly}
                draggable={!readOnly}
                onClick={() => onPick(assign.id)}
                onDragStart={(event) => {
                  if (readOnly) {
                    event.preventDefault()
                    return
                  }

                  event.dataTransfer.setData('text/assign-id', assign.id)
                  event.dataTransfer.setData(`${ASSIGN_MIME_PREFIX}${assign.id}`, assign.id)
                  event.dataTransfer.setData('text/source-node', node.id)
                  event.dataTransfer.setData('text/source-container', containerIdx.toString())
                  event.dataTransfer.setData(sourceContainerType, key)
                  event.dataTransfer.effectAllowed = 'move'
                }}
                style={{
                  width: `${visualWidthPct}%`,
                  boxShadow: selected ? 'var(--shadow-config-chip-selected)' : 'var(--shadow-config-chip-resting)'
                }}
                tabIndex={interactiveTabIndex}
                title={`${modelName} · ${detailLabel}${readOnly ? ' · read-only' : ' · drag to move'}`}
                type="button"
              >
                {exactCacheGB > 0 ? (
                  <span
                    aria-hidden
                    className="pointer-events-none absolute inset-y-0 right-0"
                    style={{
                      width: `${cachePct}%`,
                      borderLeft: '1px solid color-mix(in oklch, currentColor, transparent 65%)',
                      backgroundImage:
                        'repeating-linear-gradient(45deg, color-mix(in oklch, currentColor, transparent 78%) 0 4px, transparent 4px 8px)'
                    }}
                  />
                ) : null}
                <span className="relative z-10 flex h-full flex-col justify-center py-1">
                  <span
                    className={`truncate font-mono font-semibold uppercase ${dense ? 'text-[length:var(--density-type-annotation)] tracking-[0.06em]' : 'text-[length:var(--density-type-label)] tracking-[0.08em]'}`}
                  >
                    {modelName}
                  </span>
                  <span className="mt-0.5 truncate font-mono text-[length:var(--density-type-annotation)] leading-none">
                    {detailLabel}
                  </span>
                </span>
              </button>
            )
          })}
          <span
            className="grid min-w-0 flex-1 place-items-center rounded-[5px] border border-dashed text-[length:var(--density-type-label)] font-mono uppercase tracking-[0.18em]"
            style={{
              borderColor: invalidDropActive
                ? 'color-mix(in oklch, var(--color-bad), transparent 45%)'
                : validDropActive
                  ? 'color-mix(in oklch, var(--color-accent), transparent 48%)'
                  : 'var(--color-border-soft)',
              color: invalidDropActive
                ? 'var(--color-bad)'
                : validDropActive
                  ? 'var(--color-accent)'
                  : 'var(--color-fg-faint)',
              background: invalidDropActive
                ? 'color-mix(in oklch, var(--color-bad), transparent 90%)'
                : validDropActive
                  ? 'color-mix(in oklch, var(--color-accent), transparent 90%)'
                  : 'transparent',
              opacity: free > 0 || validDropActive || invalidDropActive ? 1 : 0
            }}
          >
            {invalidDropActive ? 'No fit' : validDropActive ? 'Drop to assign' : free > 0 ? 'Free' : ''}
          </span>
        </div>
      </section>
    </div>
  )
}
