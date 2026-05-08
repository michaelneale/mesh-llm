import * as Collapsible from '@radix-ui/react-collapsible'
import { Fragment, useEffect, useState, type Dispatch, type SetStateAction } from 'react'
import { MetaPill } from '@/features/configuration/components/MetaPill'
import { SelectedModelConfig, SelectedReservedConfig } from '@/features/configuration/components/NodeSectionParts'
import { PlacementToggle } from '@/features/configuration/components/PlacementToggle'
import { VRAMBar } from '@/features/configuration/components/VRAMBar'
import {
  containerUsedGB,
  findModel,
  hasConfigurablePlacement,
  isUnifiedMemoryNode,
  nodeReservedGB,
  nodeTotalGB,
  nodeUsableGB
} from '@/features/configuration/lib/config-math'
import { formatGB, nodeGpuCountLabel, nodeUsedGB } from '@/features/configuration/lib/config-display'
import { nodeKeyboardAttributes } from '@/features/configuration/lib/node-keyboard'
import type { ConfigAssign, ConfigModel, ConfigNode, Placement } from '@/features/app-tabs/types'

type NodeSectionProps = {
  node: ConfigNode
  assigns: ConfigAssign[]
  models?: ConfigModel[]
  setAssigns: Dispatch<SetStateAction<ConfigAssign[]>>
  selectedId?: string | null
  selectedContainerIdx?: number | null
  selectedNode: boolean
  onPick: (id: string | null) => void
  onSelectContainer: (containerIdx: number) => void
  onFocusNode: () => void
  collapsed: boolean
  setCollapsed: (collapsed: boolean) => void
  onOpenCatalog: (node: ConfigNode) => void
  onPlacementChange: (nodeId: string, placement: Placement) => void
  readOnly?: boolean
}

export function NodeSection({
  node,
  assigns,
  models,
  setAssigns,
  selectedId,
  selectedContainerIdx,
  selectedNode,
  onPick,
  onSelectContainer,
  onFocusNode,
  collapsed,
  setCollapsed,
  onOpenCatalog,
  onPlacementChange,
  readOnly = false
}: NodeSectionProps) {
  const [dragKey, setDragKey] = useState<string | null>(null)
  const open = !collapsed
  const totalNodeGB = nodeTotalGB(node)
  const reservedNodeGB = nodeReservedGB(node)
  const usableNodeGB = nodeUsableGB(node)
  const usedNodeGB = nodeUsedGB(node, assigns, models)
  const assignedCount = assigns.filter((assign) => assign.nodeId === node.id).length
  const selectedAssign = assigns.find((assign) => assign.id === selectedId && assign.nodeId === node.id)
  const selectedAssignContainerIdx = node.placement === 'pooled' ? 0 : (selectedAssign?.containerIdx ?? 0)
  const highlightedContainerIdx = selectedContainerIdx ?? (selectedAssign ? selectedAssignContainerIdx : null)
  const selectedGpu = node.gpus.find((gpu) => gpu.idx === selectedAssignContainerIdx)
  const selectedModel = selectedAssign ? findModel(selectedAssign.modelId, models) : undefined
  const selectedFootprint = selectedModel && selectedAssign ? selectedModel.sizeGB : 0
  const unifiedMemory = isUnifiedMemoryNode(node)
  const configurablePlacement = hasConfigurablePlacement(node)
  const singleGpu = !unifiedMemory && node.gpus.length === 1
  const readOnlyReason = 'Remote node context is read-only. This page only writes the local node config.'
  const placementDisabledReason = readOnly
    ? readOnlyReason
    : unifiedMemory
      ? 'Unified memory SoC nodes use a fixed pooled placement.'
      : singleGpu
        ? 'Single-GPU nodes use a fixed placement.'
        : undefined
  const selectedReservedGB =
    node.placement === 'pooled'
      ? node.gpus.reduce((sum, gpu) => sum + (gpu.reservedGB ?? 0), 0)
      : (selectedGpu?.reservedGB ?? 0)
  const selectedTotalGB = node.placement === 'pooled' ? totalNodeGB : (selectedGpu?.totalGB ?? 0)
  const containerFreeGB = selectedAssign
    ? selectedTotalGB -
      selectedReservedGB -
      containerUsedGB(
        assigns.filter((assign) => assign.id !== selectedId),
        node.id,
        selectedAssignContainerIdx,
        models
      ) -
      selectedFootprint
    : 0
  const { keyShortcuts: nodeKeyShortcuts, shortcutHelp: nodeShortcutHelp } = nodeKeyboardAttributes({
    collapsed,
    placement: node.placement,
    gpuCount: node.gpus.length,
    readOnly,
    configurablePlacement
  })

  useEffect(() => {
    const clearDragTarget = () => setDragKey(null)

    window.addEventListener('pointerup', clearDragTarget, { capture: true })
    window.addEventListener('mouseup', clearDragTarget, { capture: true })
    window.addEventListener('dragend', clearDragTarget)
    window.addEventListener('drop', clearDragTarget)
    return () => {
      window.removeEventListener('pointerup', clearDragTarget, { capture: true })
      window.removeEventListener('mouseup', clearDragTarget, { capture: true })
      window.removeEventListener('dragend', clearDragTarget)
      window.removeEventListener('drop', clearDragTarget)
    }
  }, [])

  const handlePlacementChange = (next: Placement) => {
    if (readOnly || !configurablePlacement) return

    onPlacementChange(node.id, next)
  }

  return (
    <Collapsible.Root asChild open={open} onOpenChange={(nextOpen) => setCollapsed(!nextOpen)}>
      <section
        id={`node-${node.id}`}
        aria-label={`${node.hostname} configuration node`}
        className={`panel-shell select-none overflow-hidden rounded-[var(--radius-lg)] border bg-panel transition-[background-color,border-color] ${selectedNode ? 'border-[color:color-mix(in_oklab,var(--color-accent)_36%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_4%,var(--color-panel))]' : 'border-border'}`}
        data-config-node-selected={selectedNode ? 'true' : undefined}
      >
        <header
          className={`panel-divider flex flex-wrap items-start justify-between gap-3 px-3.5 py-2.5 ${collapsed ? '' : 'border-b border-border-soft'}`}
        >
          <div className="flex min-w-0 flex-1 items-start gap-2">
            <Collapsible.Trigger
              aria-keyshortcuts={nodeKeyShortcuts}
              aria-label={`${collapsed ? 'Expand' : 'Collapse'} ${node.hostname}. ${nodeShortcutHelp}`}
              className={`ui-control grid size-6 shrink-0 place-items-center rounded-[var(--radius)] border border-border-soft bg-background text-[length:var(--density-type-caption-lg)] outline-none transition-[box-shadow] ${selectedNode ? 'focus-visible:outline-0 focus-visible:outline-transparent' : 'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent'}`}
              data-config-node-focus-target="true"
              data-config-node-id={node.id}
              data-config-selection-area="true"
              onFocus={onFocusNode}
              style={selectedNode ? { outline: '0 solid transparent' } : undefined}
              type="button"
            >
              {collapsed ? '▸' : '▾'}
            </Collapsible.Trigger>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                <h2 className="text-[length:var(--density-type-title)] font-bold tracking-[-0.02em]">
                  {node.hostname}
                </h2>
                <MetaPill className="uppercase tracking-[0.14em]">{assignedCount} assigned</MetaPill>
                <MetaPill>
                  <span className="text-foreground">{formatGB(usedNodeGB)}</span>
                  <span className="text-fg-faint"> / </span>
                  <span className="text-foreground">{formatGB(usableNodeGB)}</span>
                  <span> GB usable</span>
                  {reservedNodeGB > 0 ? (
                    <>
                      <span className="text-fg-faint"> · </span>
                      <span className="text-foreground">{formatGB(reservedNodeGB)}</span>
                      <span> reserved</span>
                    </>
                  ) : null}
                </MetaPill>
              </div>
              <span className="mt-1.5 flex flex-wrap items-center gap-1.5">
                <MetaPill tone="faint">{node.region}</MetaPill>
                <MetaPill tone="faint">{node.cpu}</MetaPill>
                <MetaPill tone="faint">{formatGB(totalNodeGB)} GB VRAM</MetaPill>
              </span>
            </div>
          </div>
          <div className="flex shrink-0 items-start gap-1.5 self-start">
            <PlacementToggle
              disabled={readOnly || !configurablePlacement}
              disabledReason={placementDisabledReason}
              groupId={node.id}
              itemTabIndex={-1}
              placement={node.placement}
              onChange={handlePlacementChange}
            />
            <button
              aria-label={`Add model to ${node.hostname}`}
              className={`ui-control-primary inline-flex h-[30px] items-center rounded-[var(--radius)] px-3 text-[length:var(--density-type-control)] font-semibold leading-none ${readOnly ? 'cursor-not-allowed opacity-55' : ''}`}
              data-config-selection-area="true"
              disabled={readOnly}
              onClick={() => onOpenCatalog(node)}
              tabIndex={-1}
              title={readOnly ? readOnlyReason : undefined}
              type="button"
            >
              Add model
            </button>
          </div>
        </header>
        <Collapsible.Content className="space-y-2.5 px-3.5 pt-2.5 pb-3">
          {node.placement === 'pooled' ? (
            <>
              <VRAMBar
                node={node}
                label={{
                  prefix: 'POOL',
                  main: `${node.hostname} · unified memory`,
                  sub: nodeGpuCountLabel(node)
                }}
                totalGB={totalNodeGB}
                reservedGB={node.gpus.reduce((sum, gpu) => sum + (gpu.reservedGB ?? 0), 0)}
                containerIdx={0}
                assigns={assigns}
                models={models}
                selectedId={selectedId}
                selectedContainer={highlightedContainerIdx === 0}
                onPick={onPick}
                onSelectContainer={() => onSelectContainer(0)}
                setAssigns={setAssigns}
                dragOver={dragKey}
                interactiveTabIndex={-1}
                readOnly={readOnly}
                setDragOver={setDragKey}
              />
              <SelectedReservedConfig
                node={node}
                selectedId={selectedId}
                containerIdx={0}
                locationLabel={`${node.hostname} pool`}
                reservedGB={node.gpus.reduce((sum, gpu) => sum + (gpu.reservedGB ?? 0), 0)}
              />
              <SelectedModelConfig
                node={node}
                containerIdx={0}
                containerFreeGB={containerFreeGB}
                models={models}
                onPick={onPick}
                readOnly={readOnly}
                selectedAssign={selectedAssign}
                selectedAssignContainerIdx={selectedAssignContainerIdx}
                setAssigns={setAssigns}
              />
            </>
          ) : (
            node.gpus.map((gpu) => (
              <Fragment key={gpu.idx}>
                <VRAMBar
                  node={node}
                  label={{ prefix: `GPU ${gpu.idx}`, main: gpu.name, sub: `${formatGB(gpu.totalGB)} GB` }}
                  totalGB={gpu.totalGB}
                  reservedGB={gpu.reservedGB}
                  containerIdx={gpu.idx}
                  assigns={assigns}
                  models={models}
                  selectedId={selectedId}
                  selectedContainer={highlightedContainerIdx === gpu.idx}
                  onPick={onPick}
                  onSelectContainer={() => onSelectContainer(gpu.idx)}
                  setAssigns={setAssigns}
                  dragOver={dragKey}
                  interactiveTabIndex={-1}
                  readOnly={readOnly}
                  setDragOver={setDragKey}
                  dense={node.gpus.length > 3}
                />
                <SelectedReservedConfig
                  node={node}
                  selectedId={selectedId}
                  containerIdx={gpu.idx}
                  locationLabel={`GPU ${gpu.idx} · ${gpu.name}`}
                  reservedGB={gpu.reservedGB ?? 0}
                />
                <SelectedModelConfig
                  node={node}
                  containerIdx={gpu.idx}
                  containerFreeGB={containerFreeGB}
                  models={models}
                  onPick={onPick}
                  readOnly={readOnly}
                  selectedAssign={selectedAssign}
                  selectedAssignContainerIdx={selectedAssignContainerIdx}
                  setAssigns={setAssigns}
                />
              </Fragment>
            ))
          )}
        </Collapsible.Content>
      </section>
    </Collapsible.Root>
  )
}
