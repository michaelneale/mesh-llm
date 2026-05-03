import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import { getPreferredConfigurationSelection, type ConfigurationState } from '@/features/configuration/hooks/useConfigurationHistory'
import { createAssignmentId } from '@/features/configuration/lib/assignment-ids'
import { containerAssigns, findPreferredModelFitContainerIdx } from '@/features/configuration/lib/config-math'
import type { CatalogTarget, SelectedContainerTarget } from '@/features/configuration/pages/ConfigurationPage.helpers'
import { getNodeTargetContainerIdx, nodeAssignmentIds } from '@/features/configuration/pages/ConfigurationPage.helpers'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'

type UseConfigurationPageSelectionParams = {
  nodes: ConfigNode[]
  assigns: ConfigAssign[]
  models?: ConfigModel[]
  initialConfiguration: ConfigurationState
  preferredAssignId?: string
  setAssigns: Dispatch<SetStateAction<ConfigAssign[]>>
}

export function useConfigurationPageSelection({ nodes, assigns, models, initialConfiguration, preferredAssignId, setAssigns }: UseConfigurationPageSelectionParams) {
  const [selectedId, setSelectedId] = useState<string | null>(() => getPreferredConfigurationSelection(initialConfiguration, preferredAssignId).assignId)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(() => getPreferredConfigurationSelection(initialConfiguration, preferredAssignId).nodeId)
  const [selectedContainerTarget, setSelectedContainerTarget] = useState<SelectedContainerTarget | null>(null)
  const [catalogFor, setCatalogFor] = useState<CatalogTarget | null>(null)
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const nodeRefs = useRef<Record<string, HTMLElement | null>>({})

  const selectedAssignId = useMemo(() => assigns.find((assign) => assign.id === selectedId)?.id ?? null, [assigns, selectedId])
  const selectedAssign = useMemo(() => assigns.find((assign) => assign.id === selectedId) ?? null, [assigns, selectedId])
  const selectedCatalogNode = useMemo(() => nodes.find((node) => node.id === catalogFor?.nodeId) ?? nodes[0], [catalogFor, nodes])

  const setNodeRef = useCallback((nodeId: string, element: HTMLElement | null) => {
    nodeRefs.current[nodeId] = element
  }, [])

  const focusNodeTarget = useCallback((nodeId: string) => {
    const nodeElement = nodeRefs.current[nodeId]
    nodeElement?.querySelector<HTMLElement>('[data-config-node-focus-target="true"]')?.focus()
    if (typeof nodeElement?.scrollIntoView === 'function') nodeElement.scrollIntoView({ block: 'nearest' })
  }, [])

  const closeCatalog = useCallback(() => {
    setCatalogFor(null)
    setCatalogError(null)
  }, [])

  const restorePreferredSelection = useCallback((configuration: ConfigurationState, preferredId = preferredAssignId) => {
    const selection = getPreferredConfigurationSelection(configuration, preferredId)
    const assign = configuration.assigns.find((item) => item.id === selection.assignId)
    const node = assign ? configuration.nodes.find((item) => item.id === assign.nodeId) : null

    setSelectedId(selection.assignId)
    setSelectedNodeId(selection.nodeId)
    setSelectedContainerTarget(assign && node ? { nodeId: node.id, containerIdx: getNodeTargetContainerIdx(node, assign.containerIdx) } : null)
    setCatalogError(null)
  }, [preferredAssignId])

  const selectContainerTarget = useCallback((nodeId: string, containerIdx: number) => {
    setSelectedNodeId(nodeId)
    setSelectedContainerTarget({ nodeId, containerIdx })
  }, [])

  const openCatalogForNode = useCallback((node: ConfigNode) => {
    const selectedAssignmentContainerIdx = selectedAssign?.nodeId === node.id
      ? getNodeTargetContainerIdx(node, selectedAssign.containerIdx)
      : null
    const selectedTargetContainerIdx = selectedContainerTarget?.nodeId === node.id
      ? getNodeTargetContainerIdx(node, selectedContainerTarget.containerIdx)
      : null

    setCatalogError(null)
    setCatalogFor({
      nodeId: node.id,
      preferredContainerIdx: selectedTargetContainerIdx ?? selectedAssignmentContainerIdx,
    })
  }, [selectedAssign, selectedContainerTarget])

  const selectNode = useCallback((node: ConfigNode) => {
    const assignIds = nodeAssignmentIds(node, assigns)
    const nextAssign = assigns.find((assign) => assign.id === assignIds[0])

    setSelectedNodeId(node.id)
    setSelectedId(assignIds[0] ?? null)
    setSelectedContainerTarget(nextAssign
      ? { nodeId: node.id, containerIdx: getNodeTargetContainerIdx(node, nextAssign.containerIdx) }
      : { nodeId: node.id, containerIdx: getNodeTargetContainerIdx(node, node.gpus[0]?.idx ?? 0) })
    focusNodeTarget(node.id)
  }, [assigns, focusNodeTarget])

  const selectNodeByOffset = useCallback((direction: -1 | 1) => {
    if (nodes.length <= 1) return

    const currentNodeId = selectedNodeId ?? selectedAssign?.nodeId ?? nodes[0]?.id
    const currentIndex = Math.max(0, nodes.findIndex((node) => node.id === currentNodeId))
    const nextIndex = (currentIndex + direction + nodes.length) % nodes.length
    const nextNode = nodes[nextIndex]

    if (nextNode) selectNode(nextNode)
  }, [nodes, selectNode, selectedAssign, selectedNodeId])

  const selectGpuSlotByOffset = useCallback((direction: -1 | 1) => {
    const node = nodes.find((item) => item.id === (selectedNodeId ?? selectedAssign?.nodeId))
    if (!node) return

    const containerIdxs = node.placement === 'pooled' ? [0] : node.gpus.map((gpu) => gpu.idx)
    if (containerIdxs.length === 0) return

    const currentContainerIdx = selectedContainerTarget?.nodeId === node.id
      ? getNodeTargetContainerIdx(node, selectedContainerTarget.containerIdx)
      : selectedAssign?.nodeId === node.id
        ? getNodeTargetContainerIdx(node, selectedAssign.containerIdx)
        : null
    const currentIndex = currentContainerIdx === null ? -1 : containerIdxs.indexOf(currentContainerIdx)
    const nextIndex = currentIndex === -1
      ? direction > 0 ? 0 : containerIdxs.length - 1
      : (currentIndex + direction + containerIdxs.length) % containerIdxs.length
    const nextContainerIdx = containerIdxs[nextIndex]
    const nextContainerAssigns = typeof nextContainerIdx === 'number' ? containerAssigns(assigns, node.id, nextContainerIdx) : []
    const selectedAssignContainerIdx = selectedAssign?.nodeId === node.id ? getNodeTargetContainerIdx(node, selectedAssign.containerIdx) : null
    const preservedAssign = selectedAssignContainerIdx === nextContainerIdx ? selectedAssign : null
    const nextAssign = preservedAssign ?? (direction > 0 ? nextContainerAssigns[0] : nextContainerAssigns.at(-1)) ?? null

    setSelectedNodeId(node.id)
    setSelectedId(nextAssign?.id ?? null)
    if (typeof nextContainerIdx === 'number') setSelectedContainerTarget({ nodeId: node.id, containerIdx: getNodeTargetContainerIdx(node, nextContainerIdx) })
    focusNodeTarget(node.id)
  }, [assigns, focusNodeTarget, nodes, selectedAssign, selectedContainerTarget, selectedNodeId])

  const selectModelInCurrentGpu = useCallback((direction: -1 | 1, jumpToBoundary = false) => {
    const node = nodes.find((item) => item.id === (selectedContainerTarget?.nodeId ?? selectedAssign?.nodeId ?? selectedNodeId))
    if (!node) return

    const fallbackContainerIdx = node.placement === 'pooled' ? 0 : node.gpus[0]?.idx
    const currentContainerIdx = selectedContainerTarget?.nodeId === node.id
      ? getNodeTargetContainerIdx(node, selectedContainerTarget.containerIdx)
      : selectedAssign?.nodeId === node.id
        ? getNodeTargetContainerIdx(node, selectedAssign.containerIdx)
        : fallbackContainerIdx
    if (typeof currentContainerIdx !== 'number') return

    const currentAssigns = containerAssigns(assigns, node.id, currentContainerIdx)
    if (currentAssigns.length === 0) return

    const selectedAssignContainerIdx = selectedAssign?.nodeId === node.id ? getNodeTargetContainerIdx(node, selectedAssign.containerIdx) : null
    const currentIndex = selectedAssignContainerIdx === currentContainerIdx
      ? currentAssigns.findIndex((assign) => assign.id === selectedAssign?.id)
      : -1
    const nextIndex = jumpToBoundary
      ? direction > 0 ? currentAssigns.length - 1 : 0
      : currentIndex === -1
        ? direction > 0 ? 0 : currentAssigns.length - 1
        : (currentIndex + direction + currentAssigns.length) % currentAssigns.length
    const nextAssign = currentAssigns[nextIndex]
    if (!nextAssign) return

    setSelectedNodeId(node.id)
    setSelectedContainerTarget({ nodeId: node.id, containerIdx: getNodeTargetContainerIdx(node, currentContainerIdx) })
    setSelectedId(nextAssign.id)
    focusNodeTarget(node.id)
  }, [assigns, focusNodeTarget, nodes, selectedAssign, selectedContainerTarget, selectedNodeId])

  const moveSelectedAssignByGpuOffset = useCallback((direction: -1 | 1) => {
    if (!selectedAssign) return

    const node = nodes.find((item) => item.id === selectedAssign.nodeId)
    if (!node || node.placement === 'pooled' || node.gpus.length < 2) return

    const currentIndex = Math.max(0, node.gpus.findIndex((gpu) => gpu.idx === selectedAssign.containerIdx))
    const nextGpu = node.gpus[(currentIndex + direction + node.gpus.length) % node.gpus.length]
    if (!nextGpu || nextGpu.idx === selectedAssign.containerIdx) return

    setSelectedNodeId(node.id)
    setSelectedContainerTarget({ nodeId: node.id, containerIdx: nextGpu.idx })
    setAssigns((items) => items.map((assign) => (assign.id === selectedAssign.id ? { ...assign, containerIdx: nextGpu.idx } : assign)))
    focusNodeTarget(node.id)
  }, [focusNodeTarget, nodes, selectedAssign, setAssigns])

  const removeAssignById = useCallback((assignId: string) => {
    setAssigns((items) => items.filter((assign) => assign.id !== assignId))
    setSelectedId((current) => (current === assignId ? null : current))
  }, [setAssigns])

  const pickNodeAssignment = useCallback((node: ConfigNode, assignId: string | null) => {
    const assign = assignId ? assigns.find((item) => item.id === assignId) : null

    setSelectedNodeId(node.id)
    if (assign) setSelectedContainerTarget({ nodeId: node.id, containerIdx: getNodeTargetContainerIdx(node, assign.containerIdx) })
    setSelectedId(assignId)
  }, [assigns])

  const selectCatalogModel = useCallback((model: ConfigModel) => {
    if (!selectedCatalogNode) return false

    const containerIdx = findPreferredModelFitContainerIdx(model, selectedCatalogNode, assigns, catalogFor?.preferredContainerIdx ?? null, 4096, models)
    if (containerIdx === null) {
      setCatalogError(`${model.name} does not fit on any GPU in ${selectedCatalogNode.hostname}. Pick a smaller model, reduce context on existing models, or free GPU memory.`)
      return false
    }

    const id = createAssignmentId(assigns)

    setAssigns((items) => [...items, { id, modelId: model.id, nodeId: selectedCatalogNode.id, containerIdx, ctx: 4096 }])
    setSelectedNodeId(selectedCatalogNode.id)
    setSelectedContainerTarget({ nodeId: selectedCatalogNode.id, containerIdx })
    setSelectedId(id)
    closeCatalog()
    return true
  }, [assigns, catalogFor?.preferredContainerIdx, closeCatalog, models, selectedCatalogNode, setAssigns])

  useEffect(() => {
    if (!selectedId) return undefined

    const onPointerDown = (event: PointerEvent) => {
      const target = event.target
      if (!(target instanceof Element)) return
      if (target.closest('[data-model-selection-area="true"], [data-config-selection-area="true"]')) return
      setSelectedId(null)
    }

    window.addEventListener('pointerdown', onPointerDown, { capture: true })
    return () => window.removeEventListener('pointerdown', onPointerDown, { capture: true })
  }, [selectedId])

  return {
    selectedId,
    selectedNodeId,
    selectedContainerTarget,
    selectedAssignId,
    selectedAssign,
    catalogFor,
    catalogError,
    selectedCatalogNode,
    setCatalogError,
    setSelectedId,
    setSelectedNodeId,
    setSelectedContainerTarget,
    setNodeRef,
    focusNodeTarget,
    closeCatalog,
    restorePreferredSelection,
    selectContainerTarget,
    openCatalogForNode,
    selectNode,
    selectNodeByOffset,
    selectGpuSlotByOffset,
    selectModelInCurrentGpu,
    moveSelectedAssignByGpuOffset,
    removeAssignById,
    pickNodeAssignment,
    selectCatalogModel,
  }
}
