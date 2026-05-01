import { useEffect, useMemo, useRef, useState } from 'react'
import { createAssignmentId } from '@/features/configuration/lib/assignment-ids'
import {
  containerReservedGB,
  containerTotalGB,
  containerUsedGB,
  findModel,
  findPreferredModelFitContainerIdx,
  nodeReservedGB,
  nodeTotalGB,
} from '@/features/configuration/lib/config-math'
import { CHAT_HARNESS, CONFIGURATION_HARNESS, DASHBOARD_HARNESS } from '@/features/app-tabs/data'
import type {
  ConfigAssign,
  ConfigNode,
  ModelSelectOption,
  ModelSummary,
  Placement,
  StatusBadgeTone,
  ThreadMessage,
  TransparencyMessage,
} from '@/features/app-tabs/types'
import { useClipboardCopy } from '@/lib/useClipboardCopy'

export const TAG_OPTIONS = ['Text', 'Vision', 'Tools', 'Warm'] as const

const PRIMARY_CONFIG_NODE_ID = CONFIGURATION_HARNESS.nodes[0]?.id ?? 'node-a'
const INITIAL_THREAD_ID = CHAT_HARNESS.conversations[0]?.id ?? ''
const INITIAL_CONFIG_MODEL_ID = CONFIGURATION_HARNESS.catalog.find((model) => model.id === 'phi4')?.id ?? CONFIGURATION_HARNESS.catalog[0]?.id ?? ''
const INITIAL_FOCUSED_ASSIGN_ID = CONFIGURATION_HARNESS.preferredAssignId
  ?? CONFIGURATION_HARNESS.assigns.find((assign) => assign.nodeId === PRIMARY_CONFIG_NODE_ID)?.id
  ?? ''

function cloneDashboardModels(models: ModelSummary[]) {
  return models.map((model) => ({ ...model, tags: [...model.tags] }))
}

function cloneConfigNodes(nodes: ConfigNode[]) {
  return nodes.map((node) => ({ ...node, gpus: node.gpus.map((gpu) => ({ ...gpu })) }))
}

function cloneConfigAssigns(assigns: ConfigAssign[]) {
  return assigns.map((assign) => ({ ...assign }))
}

function modelStatusBadge(label: string, tone?: StatusBadgeTone) {
  if (tone === 'bad') return { label, tone: 'bad' as const }
  if (tone === 'warn') return { label, tone: 'warn' as const }
  if (tone === 'accent') return { label, tone: 'accent' as const }
  if (tone === 'muted') return { label, tone: 'muted' as const }
  return { label, tone: 'good' as const }
}

function selectStatusForModel(status: 'ready' | 'warming' | 'offline' | 'warm'): ModelSelectOption['status'] {
  if (status === 'offline') return modelStatusBadge('Offline', 'bad')
  if (status === 'warming') return modelStatusBadge('Warming', 'warn')
  if (status === 'ready') return modelStatusBadge('Ready', 'good')
  return modelStatusBadge('Warm', 'good')
}

function updateTransparencyText(message: TransparencyMessage | undefined, text: string) {
  if (!message) return undefined
  return { ...message, text }
}

type ChatPreviewDraft = {
  conversationLabel: string
  userBody: string
  assistantBody: string
}

function buildChatPreviewDrafts(): Record<string, ChatPreviewDraft> {
  return Object.fromEntries(CHAT_HARNESS.conversations.map((conversation) => {
    const messages = CHAT_HARNESS.threads[conversation.id] ?? []
    const userMessage = messages.find((message) => message.messageRole === 'user')
    const assistantMessage = messages.find((message) => message.messageRole === 'assistant')

    return [conversation.id, {
      conversationLabel: conversation.title,
      userBody: userMessage?.body ?? '',
      assistantBody: assistantMessage?.body ?? '',
    }]
  }))
}

function nextGpuIdx(node: ConfigNode) {
  return node.gpus.reduce((maxIdx, gpu) => Math.max(maxIdx, gpu.idx), -1) + 1
}

function resolveSeparateContainerIdx(node: ConfigNode, desiredIdx?: number) {
  if (typeof desiredIdx === 'number' && node.gpus.some((gpu) => gpu.idx === desiredIdx)) return desiredIdx
  return node.gpus[0]?.idx ?? 0
}

export function useDeveloperPlaygroundState() {
  const { copyState: connectCopyState, copyText: copyConnectText } = useClipboardCopy()
  const [selectedDashboardModelName, setSelectedDashboardModelName] = useState(DASHBOARD_HARNESS.models[0]?.name ?? '')
  const [selectedPeerId, setSelectedPeerId] = useState(DASHBOARD_HARNESS.peers[0]?.id)
  const [dashboardModels, setDashboardModels] = useState<ModelSummary[]>(() => cloneDashboardModels(DASHBOARD_HARNESS.models))
  const [connectDescription, setConnectDescription] = useState(DASHBOARD_HARNESS.connect.description)
  const [connectRunCommand, setConnectRunCommand] = useState(DASHBOARD_HARNESS.connect.runCommand)
  const [activeConversationId, setActiveConversationId] = useState(INITIAL_THREAD_ID)
  const [sidebarTab, setSidebarTab] = useState<'conversations' | 'transparency'>('conversations')
  const [inspectedMessage, setInspectedMessage] = useState<TransparencyMessage | undefined>()
  const [prompt, setPrompt] = useState('')
  const [selectedChatModel, setSelectedChatModel] = useState(CHAT_HARNESS.models[0]?.name ?? '')
  const [shellSelectedModel, setShellSelectedModel] = useState(CHAT_HARNESS.models[0]?.name ?? '')
  const [chatHeaderTitle, setChatHeaderTitle] = useState(CHAT_HARNESS.title)
  const [chatPreviewDrafts, setChatPreviewDrafts] = useState<Record<string, ChatPreviewDraft>>(() => buildChatPreviewDrafts())
  const [configNodes, setConfigNodes] = useState<ConfigNode[]>(() => cloneConfigNodes(CONFIGURATION_HARNESS.nodes))
  const [configAssigns, setConfigAssigns] = useState<ConfigAssign[]>(() => cloneConfigAssigns(CONFIGURATION_HARNESS.assigns))
  const [selectedConfigModelId, setSelectedConfigModelId] = useState(INITIAL_CONFIG_MODEL_ID)
  const [focusedAssignId, setFocusedAssignId] = useState(INITIAL_FOCUSED_ASSIGN_ID)
  const [selectedConfigSelectionId, setSelectedConfigSelectionId] = useState<string | null>(INITIAL_FOCUSED_ASSIGN_ID || null)
  const [selectedContainerIdx, setSelectedContainerIdx] = useState(CONFIGURATION_HARNESS.assigns.find((assign) => assign.id === INITIAL_FOCUSED_ASSIGN_ID)?.containerIdx ?? 0)
  const [dragOver, setDragOver] = useState<string | null>(null)
  const separateContainersRef = useRef<Record<string, number>>(
    Object.fromEntries(
      cloneConfigAssigns(CONFIGURATION_HARNESS.assigns)
        .filter((assign) => assign.nodeId === PRIMARY_CONFIG_NODE_ID)
        .map((assign) => [assign.id, assign.containerIdx]),
    ),
  )

  useEffect(() => {
    const clearDragTarget = () => setDragOver(null)

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

  const chatOptions = useMemo<ModelSelectOption[]>(() => {
    return CHAT_HARNESS.models.slice(0, 5).map((model) => ({
      value: model.name,
      label: model.name,
      meta: `${model.family} · ${model.context}`,
      status: selectStatusForModel(model.status),
    }))
  }, [])

  const configCardOptions = useMemo<ModelSelectOption[]>(() => {
    return CONFIGURATION_HARNESS.catalog.map((model) => ({
      value: model.id,
      label: model.name,
      meta: `${model.family} · ${model.ctxMaxK}k ctx`,
    }))
  }, [])

  const activeConversation = CHAT_HARNESS.conversations.find((conversation) => conversation.id === activeConversationId) ?? CHAT_HARNESS.conversations[0]
  const activeMessages = useMemo(() => CHAT_HARNESS.threads[activeConversation?.id ?? ''] ?? [], [activeConversation?.id])
  const activeUserMessage = activeMessages.find((message) => message.messageRole === 'user')
  const activeAssistantMessage = activeMessages.find((message) => message.messageRole === 'assistant')
  const selectedDashboardModel = dashboardModels.find((model) => model.name === selectedDashboardModelName) ?? dashboardModels[0]
  const primaryNode = configNodes.find((node) => node.id === PRIMARY_CONFIG_NODE_ID) ?? configNodes[0]
  const primaryNodeAssigns = useMemo(() => configAssigns.filter((assign) => assign.nodeId === primaryNode.id), [configAssigns, primaryNode.id])
  const focusedAssign = primaryNodeAssigns.find((assign) => assign.id === focusedAssignId) ?? primaryNodeAssigns[0]
  const focusedConfigModel = focusedAssign ? findModel(focusedAssign.modelId, CONFIGURATION_HARNESS.catalog) : undefined
  const activeDraft = chatPreviewDrafts[activeConversation?.id ?? ''] ?? {
    conversationLabel: activeConversation?.title ?? '',
    userBody: activeUserMessage?.body ?? '',
    assistantBody: activeAssistantMessage?.body ?? '',
  }
  const previewConversations = useMemo(() => {
    return CHAT_HARNESS.conversations.map((conversation) => (
      conversation.id === activeConversation?.id ? { ...conversation, title: activeDraft.conversationLabel } : conversation
    ))
  }, [activeConversation?.id, activeDraft.conversationLabel])
  const effectiveSelectedContainerIdx = primaryNode.placement === 'pooled'
    ? 0
    : primaryNode.gpus.some((gpu) => gpu.idx === selectedContainerIdx)
      ? selectedContainerIdx
      : primaryNode.gpus[0]?.idx ?? 0

  useEffect(() => {
    if (primaryNode.placement !== 'separate') return
    for (const assign of primaryNodeAssigns) {
      separateContainersRef.current[assign.id] = assign.containerIdx
    }
  }, [primaryNode.placement, primaryNodeAssigns])

  const previewMessages = useMemo<ThreadMessage[]>(() => {
    return activeMessages.map((message) => {
      if (message.id === activeUserMessage?.id) {
        return {
          ...message,
          body: activeDraft.userBody,
          inspectMessage: updateTransparencyText(message.inspectMessage, activeDraft.userBody),
        }
      }

      if (message.id === activeAssistantMessage?.id) {
        return {
          ...message,
          body: activeDraft.assistantBody,
          inspectMessage: updateTransparencyText(message.inspectMessage, activeDraft.assistantBody),
        }
      }

      return message
    })
  }, [activeAssistantMessage?.id, activeDraft.assistantBody, activeDraft.userBody, activeMessages, activeUserMessage?.id])

  const focusedContainerFreeGB = useMemo(() => {
    if (!focusedAssign) return 0
    const containerIdx = primaryNode.placement === 'pooled' ? 0 : focusedAssign.containerIdx
    const model = findModel(focusedAssign.modelId, CONFIGURATION_HARNESS.catalog)
    return containerTotalGB(primaryNode, containerIdx)
      - containerReservedGB(primaryNode, containerIdx)
      - containerUsedGB(configAssigns.filter((assign) => assign.id !== focusedAssign.id), primaryNode.id, containerIdx, CONFIGURATION_HARNESS.catalog)
      - (model?.sizeGB ?? 0)
  }, [configAssigns, focusedAssign, primaryNode])

  const vramBars = useMemo(() => {
    if (primaryNode.placement === 'pooled') {
      return [{
        containerIdx: 0,
        totalGB: nodeTotalGB(primaryNode),
        reservedGB: nodeReservedGB(primaryNode),
        label: {
          prefix: 'POOL',
          main: `${primaryNode.hostname} pool`,
          sub: `${primaryNode.gpus.length} GPUs combined`,
        },
      }]
    }

    return primaryNode.gpus.map((gpu) => ({
      containerIdx: gpu.idx,
      totalGB: gpu.totalGB,
      reservedGB: gpu.reservedGB,
      label: {
        prefix: `GPU ${gpu.idx}`,
        main: gpu.name,
        sub: `${gpu.totalGB} GB`,
      },
    }))
  }, [primaryNode])

  function toggleSelectedModelTag(tag: (typeof TAG_OPTIONS)[number]) {
    if (!selectedDashboardModel) return

    setDashboardModels((models) => models.map((model) => {
      if (model.name !== selectedDashboardModel.name) return model
      const hasTag = model.tags.includes(tag)
      return {
        ...model,
        tags: hasTag ? model.tags.filter((item) => item !== tag) : [tag, ...model.tags.filter((item) => item !== tag)],
      }
    }))
  }

  function updateActiveChatDraft(field: keyof ChatPreviewDraft, value: string) {
    const conversationId = activeConversation?.id
    if (!conversationId) return

    setChatPreviewDrafts((drafts) => {
      const baseDraft = drafts[conversationId] ?? {
        conversationLabel: activeConversation.title,
        userBody: activeUserMessage?.body ?? '',
        assistantBody: activeAssistantMessage?.body ?? '',
      }

      return {
        ...drafts,
        [conversationId]: {
          ...baseDraft,
          [field]: value,
        },
      }
    })
  }

  function handlePlacementChange(nextPlacement: Placement) {
    if (!primaryNode || nextPlacement === primaryNode.placement) return

    if (nextPlacement === 'pooled') {
      for (const assign of primaryNodeAssigns) {
        separateContainersRef.current[assign.id] = assign.containerIdx
      }

      setConfigNodes((nodes) => nodes.map((node) => (node.id === primaryNode.id ? { ...node, placement: 'pooled' } : node)))
      setConfigAssigns((assigns) => assigns.map((assign) => (
        assign.nodeId === primaryNode.id ? { ...assign, containerIdx: 0 } : assign
      )))
      setSelectedContainerIdx(0)
      return
    }

    const fallbackContainerIdx = primaryNode.gpus[0]?.idx ?? 0
    const validGpuIds = new Set(primaryNode.gpus.map((gpu) => gpu.idx))

    setConfigNodes((nodes) => nodes.map((node) => (node.id === primaryNode.id ? { ...node, placement: 'separate' } : node)))
    setConfigAssigns((assigns) => assigns.map((assign) => {
      if (assign.nodeId !== primaryNode.id) return assign
      const restoredContainerIdx = separateContainersRef.current[assign.id]
      return {
        ...assign,
        containerIdx: validGpuIds.has(restoredContainerIdx) ? restoredContainerIdx : fallbackContainerIdx,
      }
    }))
    setSelectedContainerIdx(resolveSeparateContainerIdx(primaryNode, separateContainersRef.current[focusedAssignId]))
  }

  function addGpu() {
    if (!primaryNode) return
    const gpuIdx = nextGpuIdx(primaryNode)

    setConfigNodes((nodes) => nodes.map((node) => {
      if (node.id !== primaryNode.id) return node
      return {
        ...node,
        gpus: [...node.gpus, { idx: gpuIdx, name: `Playground GPU ${gpuIdx}`, totalGB: 24, reservedGB: 0.7 }],
      }
    }))

    if (primaryNode.placement === 'separate') setSelectedContainerIdx(gpuIdx)
  }

  function removeGpu(gpuIdx: number) {
    if (!primaryNode || primaryNode.gpus.length <= 1) return

    const remainingGpus = primaryNode.gpus.filter((gpu) => gpu.idx !== gpuIdx)
    const fallbackContainerIdx = remainingGpus[0]?.idx ?? 0

    Object.entries(separateContainersRef.current).forEach(([assignId, containerIdx]) => {
      if (containerIdx === gpuIdx) separateContainersRef.current[assignId] = fallbackContainerIdx
    })

    setConfigNodes((nodes) => nodes.map((node) => (
      node.id === primaryNode.id ? { ...node, gpus: remainingGpus } : node
    )))
    setConfigAssigns((assigns) => assigns.map((assign) => {
      if (assign.nodeId !== primaryNode.id || assign.containerIdx !== gpuIdx) return assign
      return { ...assign, containerIdx: primaryNode.placement === 'pooled' ? 0 : fallbackContainerIdx }
    }))

    if (effectiveSelectedContainerIdx === gpuIdx) setSelectedContainerIdx(primaryNode.placement === 'pooled' ? 0 : fallbackContainerIdx)
  }

  function selectFocusedAssign(nextAssignId: string) {
    const nextAssign = primaryNodeAssigns.find((assign) => assign.id === nextAssignId)
    setFocusedAssignId(nextAssignId)
    setSelectedConfigSelectionId(nextAssignId)
    if (nextAssign) setSelectedContainerIdx(primaryNode.placement === 'pooled' ? 0 : nextAssign.containerIdx)
  }

  function moveFocusedAssign(nextContainerIdx: number) {
    if (!focusedAssign) return
    const resolvedContainerIdx = primaryNode.placement === 'pooled' ? 0 : resolveSeparateContainerIdx(primaryNode, nextContainerIdx)

    if (primaryNode.placement === 'separate') {
      separateContainersRef.current[focusedAssign.id] = resolvedContainerIdx
    }

    setConfigAssigns((assigns) => assigns.map((assign) => (
      assign.id === focusedAssign.id ? { ...assign, containerIdx: resolvedContainerIdx } : assign
    )))
    setSelectedContainerIdx(resolvedContainerIdx)
    setSelectedConfigSelectionId(focusedAssign.id)
  }

  function updateAssignCtx(assignId: string, ctx: number) {
    setConfigAssigns((assigns) => assigns.map((assign) => (assign.id === assignId ? { ...assign, ctx } : assign)))
  }

  function addConfigCard() {
    if (!primaryNode || !selectedConfigModelId) return
    const model = findModel(selectedConfigModelId, CONFIGURATION_HARNESS.catalog)
    if (!model) return

    const preferredContainerIdx = primaryNode.placement === 'pooled' ? 0 : effectiveSelectedContainerIdx
    const resolvedContainerIdx = findPreferredModelFitContainerIdx(
      model,
      primaryNode,
      configAssigns,
      preferredContainerIdx,
      4096,
      CONFIGURATION_HARNESS.catalog,
    ) ?? preferredContainerIdx

    const nextId = createAssignmentId(configAssigns)
    if (primaryNode.placement === 'separate') separateContainersRef.current[nextId] = resolvedContainerIdx
    setFocusedAssignId(nextId)
    setSelectedConfigSelectionId(nextId)
    setSelectedContainerIdx(resolvedContainerIdx)
    setConfigAssigns((assigns) => [...assigns, { id: nextId, modelId: model.id, nodeId: primaryNode.id, containerIdx: resolvedContainerIdx, ctx: 4096 }])
  }

  function removeConfigCard(assignId: string) {
    delete separateContainersRef.current[assignId]
    const nextFocusedAssign = primaryNodeAssigns.find((assign) => assign.id !== assignId)
    setConfigAssigns((assigns) => assigns.filter((assign) => assign.id !== assignId))

    if (focusedAssign?.id === assignId) {
      setFocusedAssignId(nextFocusedAssign?.id ?? '')
      if (nextFocusedAssign) setSelectedContainerIdx(primaryNode.placement === 'pooled' ? 0 : nextFocusedAssign.containerIdx)
    }

    if (selectedConfigSelectionId === assignId) {
      setSelectedConfigSelectionId(nextFocusedAssign?.id ?? null)
    }
  }

  return {
    activeAssistantMessage,
    activeConversation,
    activeDraft,
    activeUserMessage,
    addConfigCard,
    addGpu,
    chatHeaderTitle,
    chatOptions,
    configAssigns,
    configCardOptions,
    configNodes,
    connectCopyState,
    connectDescription,
    connectRunCommand,
    copyConnectText,
    dashboardModels,
    dragOver,
    effectiveSelectedContainerIdx,
    focusedAssign,
    focusedAssignId,
    focusedConfigModel,
    focusedContainerFreeGB,
    handlePlacementChange,
    inspectedMessage,
    moveFocusedAssign,
    previewConversations,
    previewMessages,
    primaryNode,
    primaryNodeAssigns,
    prompt,
    removeConfigCard,
    removeGpu,
    selectedChatModel,
    selectedConfigModelId,
    selectedConfigSelectionId,
    selectedDashboardModel,
    selectedDashboardModelName,
    selectedPeerId,
    selectFocusedAssign,
    setActiveConversationId,
    setChatHeaderTitle,
    setConfigAssigns,
    setConnectDescription,
    setConnectRunCommand,
    setDragOver,
    setInspectedMessage,
    setPrompt,
    setSelectedChatModel,
    setSelectedConfigModelId,
    setSelectedConfigSelectionId,
    setSelectedDashboardModelName,
    setSelectedPeerId,
    setShellSelectedModel,
    setSidebarTab,
    shellSelectedModel,
    sidebarTab,
    toggleSelectedModelTag,
    updateActiveChatDraft,
    updateAssignCtx,
    vramBars,
  }
}

export type DeveloperPlaygroundState = ReturnType<typeof useDeveloperPlaygroundState>
