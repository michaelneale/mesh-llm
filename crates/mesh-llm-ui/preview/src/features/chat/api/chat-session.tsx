import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import { CHAT_HARNESS } from '@/features/app-tabs/data'
import type { ChatHarnessData, ThreadMessage } from '@/features/app-tabs/types'
import { useDataMode } from '@/lib/data-mode'
import { useBooleanFeatureFlag } from '@/lib/feature-flags'
import { useMeshChat } from './use-chat'
import { usePersistentChatSystemPrompt } from './system-prompt'
import { useChatMessages } from './use-chat-messages'
import { useConversations } from './use-conversations'
import { ChatSessionContext, type ChatSessionContextValue, type ChatSessionProviderProps } from './chat-session-context'
import { createChatDraftConversationId } from './chat-session-ids'
import {
  extractThreadMessageMetadata,
  mergeThreadMessageMetadata,
  responseMetadataToThreadMessage,
  threadMessageMetadataEquals,
  type ThreadMessageMetadata
} from './response-metadata'

function inspectMessagesMatch(left: ThreadMessage['inspectMessage'], right: ThreadMessage['inspectMessage']): boolean {
  if (!left || !right) return left === right

  return JSON.stringify(left) === JSON.stringify(right)
}

function threadMessagesMatch(left: ThreadMessage[] | undefined, right: ThreadMessage[]): boolean {
  if (!left || left.length !== right.length) return false

  return left.every((message, index) => {
    const other = right[index]
    return (
      message?.id === other?.id &&
      message?.messageRole === other?.messageRole &&
      message?.body === other?.body &&
      message?.inspectLabel === other?.inspectLabel &&
      inspectMessagesMatch(message?.inspectMessage, other?.inspectMessage) &&
      threadMessageMetadataEquals(message, other)
    )
  })
}

type ChatLaneId = 'primary' | 'secondary'

type ChatLaneRoutingState = {
  activeLaneId: ChatLaneId
  laneConversationIds: Record<ChatLaneId, string>
}

type NormalizeChatLaneRoutingOptions = {
  activeConversationKey: string
  draftConversationId: string
  primaryIsStreaming: boolean
  secondaryIsStreaming: boolean
  selectedConversationId: string
}

type ChatLane = {
  chat: ReturnType<typeof useMeshChat>
  conversationId: string
  initialThread: ThreadMessage[]
  isSyncedToConversation: boolean
  isStreaming: boolean
  liveMessagesWithModels: ThreadMessage[]
}

const CHAT_LANE_IDS: readonly ChatLaneId[] = ['primary', 'secondary']
const STREAMING_THREAD_PERSIST_INTERVAL_MS = 2_000

function otherChatLaneId(laneId: ChatLaneId): ChatLaneId {
  return laneId === 'primary' ? 'secondary' : 'primary'
}

function findChatLaneId(
  laneConversationIds: Record<ChatLaneId, string>,
  conversationId: string
): ChatLaneId | undefined {
  return CHAT_LANE_IDS.find((laneId) => laneConversationIds[laneId] === conversationId)
}

function chatLaneIsStreaming(laneId: ChatLaneId, options: NormalizeChatLaneRoutingOptions): boolean {
  return laneId === 'primary' ? options.primaryIsStreaming : options.secondaryIsStreaming
}

function normalizeChatLaneRoutingState(
  current: ChatLaneRoutingState,
  options: NormalizeChatLaneRoutingOptions
): ChatLaneRoutingState {
  let activeLaneId = current.activeLaneId
  let laneConversationIds = current.laneConversationIds

  function setActiveLaneId(nextActiveLaneId: ChatLaneId) {
    activeLaneId = nextActiveLaneId
  }

  function setLaneConversationId(laneId: ChatLaneId, conversationId: string) {
    if (laneConversationIds[laneId] === conversationId) return
    laneConversationIds = { ...laneConversationIds, [laneId]: conversationId }
  }

  const selectedLaneId = findChatLaneId(laneConversationIds, options.selectedConversationId)
  if (selectedLaneId && selectedLaneId !== activeLaneId) {
    setActiveLaneId(selectedLaneId)
  }

  if (options.selectedConversationId && !selectedLaneId) {
    const inactiveLaneId = otherChatLaneId(activeLaneId)
    const activeLaneIsStreaming = chatLaneIsStreaming(activeLaneId, options)
    const inactiveLaneIsStreaming = chatLaneIsStreaming(inactiveLaneId, options)
    const nextLaneId = activeLaneIsStreaming ? inactiveLaneId : activeLaneId

    if (nextLaneId === activeLaneId || !inactiveLaneIsStreaming) {
      setActiveLaneId(nextLaneId)
      setLaneConversationId(nextLaneId, options.selectedConversationId)
    }
  }

  const idleLaneId = otherChatLaneId(activeLaneId)
  const currentSelectedLaneId = findChatLaneId(laneConversationIds, options.selectedConversationId)
  if (
    currentSelectedLaneId !== idleLaneId &&
    laneConversationIds[activeLaneId] !== options.draftConversationId &&
    !chatLaneIsStreaming(idleLaneId, options) &&
    laneConversationIds[idleLaneId] !== options.draftConversationId
  ) {
    setLaneConversationId(idleLaneId, options.draftConversationId)
  }

  const activeConversationOwner = findChatLaneId(laneConversationIds, options.activeConversationKey)
  if (activeConversationOwner) {
    setActiveLaneId(activeConversationOwner)
  }

  if (activeLaneId === current.activeLaneId && laneConversationIds === current.laneConversationIds) {
    return current
  }

  return { activeLaneId, laneConversationIds }
}

type UseChatLaneOptions = {
  conversationId: string
  initialThread: ThreadMessage[]
  liveMode: boolean
  messageModels: Record<string, string>
  responseMetadataByConversation: Record<string, Record<string, ThreadMessageMetadata>>
  setResponseMetadataByConversation: Dispatch<SetStateAction<Record<string, Record<string, ThreadMessageMetadata>>>>
  sessionModel: string
  systemPrompt: string
  updateThread: ReturnType<typeof useConversations>['updateThread']
}

function useChatLane({
  conversationId,
  initialThread,
  liveMode,
  messageModels,
  responseMetadataByConversation,
  setResponseMetadataByConversation,
  sessionModel,
  systemPrompt,
  updateThread
}: UseChatLaneOptions): ChatLane {
  const [previousPersistedConversationId, setPreviousPersistedConversationId] = useState(conversationId)
  const previousPersistedConversationRef = useRef(previousPersistedConversationId)
  const initialResponseMetadata = useMemo(() => {
    const next: Record<string, ThreadMessageMetadata> = {}
    for (const message of initialThread) {
      const metadata = extractThreadMessageMetadata(message)
      if (metadata) next[message.id] = metadata
    }
    return next
  }, [initialThread])
  const handleResponseMetadata = useCallback(
    (metadata: Parameters<typeof responseMetadataToThreadMessage>[0]) => {
      const threadMetadata = responseMetadataToThreadMessage(metadata)
      setResponseMetadataByConversation((current) => ({
        ...current,
        [conversationId]: { ...(current[conversationId] ?? {}), [metadata.messageId]: threadMetadata }
      }))
    },
    [conversationId, setResponseMetadataByConversation]
  )
  const chat = useMeshChat({
    conversationId,
    model: sessionModel,
    systemPrompt,
    initialMessages: initialThread,
    onResponseMetadata: handleResponseMetadata
  })
  const liveMessages = useChatMessages(chat.messages)
  const liveMessagesWithModels = useMemo<ThreadMessage[]>(() => {
    const liveResponseMetadata = responseMetadataByConversation[conversationId] ?? {}
    return liveMessages.map((message) => {
      const submittedModel = messageModels[message.id]
      const metadata = liveResponseMetadata[message.id] ?? initialResponseMetadata[message.id]
      return mergeThreadMessageMetadata(message, metadata, submittedModel)
    })
  }, [conversationId, initialResponseMetadata, liveMessages, messageModels, responseMetadataByConversation])
  const isStreaming = chat.status === 'submitted' || chat.status === 'streaming'

  useEffect(() => {
    previousPersistedConversationRef.current = previousPersistedConversationId
  }, [previousPersistedConversationId])

  useEffect(() => {
    if (!liveMode || !conversationId) return

    if (previousPersistedConversationRef.current !== conversationId) {
      window.queueMicrotask(() => setPreviousPersistedConversationId(conversationId))
      return
    }

    if (liveMessagesWithModels.length === 0 || threadMessagesMatch(initialThread, liveMessagesWithModels)) {
      return
    }

    if (!isStreaming) {
      updateThread(conversationId, liveMessagesWithModels)
      return
    }

    const timerId = setTimeout(() => {
      updateThread(conversationId, liveMessagesWithModels)
    }, STREAMING_THREAD_PERSIST_INTERVAL_MS)
    return () => clearTimeout(timerId)
  }, [conversationId, initialThread, isStreaming, liveMessagesWithModels, liveMode, updateThread])

  return {
    chat,
    conversationId,
    initialThread,
    isSyncedToConversation: previousPersistedConversationId === conversationId,
    isStreaming,
    liveMessagesWithModels
  }
}

export function ChatSessionProvider({ children, data = CHAT_HARNESS }: ChatSessionProviderProps) {
  const { mode } = useDataMode()
  const liveMode = mode === 'live'
  const conversationFallback = useMemo<ChatHarnessData>(() => {
    if (!liveMode) return data
    return { ...data, conversations: [], conversationGroups: [], threads: {} }
  }, [data, liveMode])
  const conversations = useConversations(conversationFallback, liveMode ? 'live' : 'harness')
  const activeConversation =
    conversations.conversations.find((conversation) => conversation.id === conversations.activeConversationId) ??
    conversations.conversations[0]
  const activeConversationKey = activeConversation?.id ?? ''
  const [draftConversationId, setDraftConversationId] = useState(() => createChatDraftConversationId())
  const [laneRoutingState, setLaneRoutingState] = useState<ChatLaneRoutingState>(() => ({
    activeLaneId: 'primary',
    laneConversationIds: {
      primary: activeConversationKey || draftConversationId,
      secondary: createChatDraftConversationId()
    }
  }))
  const { activeLaneId, laneConversationIds } = laneRoutingState
  const selectedConversationId = activeConversationKey || draftConversationId
  const selectedLaneId = findChatLaneId(laneConversationIds, selectedConversationId)
  const visibleLaneId = selectedLaneId ?? activeLaneId
  const [sessionModel, setSessionModel] = useState('auto')
  const [messageModels, setMessageModels] = useState<Record<string, string>>({})
  const systemPromptButtonEnabled = useBooleanFeatureFlag('chat/systemPromptButton')
  const { systemPrompt, setSystemPrompt } = usePersistentChatSystemPrompt()
  const effectiveSystemPrompt = systemPromptButtonEnabled ? systemPrompt : ''
  const [responseMetadataByConversation, setResponseMetadataByConversation] = useState<
    Record<string, Record<string, ThreadMessageMetadata>>
  >({})
  const primaryInitialThread = useMemo(
    () => conversations.threads[laneConversationIds.primary] ?? [],
    [conversations.threads, laneConversationIds.primary]
  )
  const secondaryInitialThread = useMemo(
    () => conversations.threads[laneConversationIds.secondary] ?? [],
    [conversations.threads, laneConversationIds.secondary]
  )
  const updateThread = conversations.updateThread
  const primaryLane = useChatLane({
    conversationId: laneConversationIds.primary,
    initialThread: primaryInitialThread,
    liveMode,
    messageModels,
    responseMetadataByConversation,
    setResponseMetadataByConversation,
    sessionModel,
    systemPrompt: effectiveSystemPrompt,
    updateThread
  })
  const secondaryLane = useChatLane({
    conversationId: laneConversationIds.secondary,
    initialThread: secondaryInitialThread,
    liveMode,
    messageModels,
    responseMetadataByConversation,
    setResponseMetadataByConversation,
    sessionModel,
    systemPrompt: effectiveSystemPrompt,
    updateThread
  })
  const activeLane = visibleLaneId === 'primary' ? primaryLane : secondaryLane
  const chat = activeLane.chat
  const chatConversationId = activeLane.conversationId
  const initialThread = activeLane.initialThread
  const isStreaming = activeLane.isStreaming
  const liveConversationChanged = !activeLane.isSyncedToConversation
  const liveMessagesWithModels = activeLane.liveMessagesWithModels
  const selectedConversationIsLiveSession = activeConversationKey === chatConversationId
  const activeMessages = useMemo(
    () =>
      liveMode
        ? !selectedConversationIsLiveSession || liveConversationChanged
          ? (conversations.threads[activeConversationKey] ?? [])
          : liveMessagesWithModels.length > 0
            ? liveMessagesWithModels
            : initialThread
        : (conversations.threads[activeConversationKey] ?? []),
    [
      activeConversationKey,
      conversations.threads,
      initialThread,
      liveConversationChanged,
      liveMessagesWithModels,
      liveMode,
      selectedConversationIsLiveSession
    ]
  )
  const messageCounts = useMemo(() => {
    const counts = Object.fromEntries(Object.entries(conversations.threads).map(([id, thread]) => [id, thread.length]))
    for (const lane of [primaryLane, secondaryLane]) {
      if (liveMode && lane.isStreaming && lane.conversationId) {
        counts[lane.conversationId] = lane.liveMessagesWithModels.length
      }
    }
    return counts
  }, [conversations.threads, liveMode, primaryLane, secondaryLane])
  const streamingConversationIds = useMemo(
    () =>
      liveMode
        ? [primaryLane, secondaryLane]
            .filter((lane) => lane.isStreaming && lane.conversationId)
            .map((lane) => lane.conversationId)
        : [],
    [liveMode, primaryLane, secondaryLane]
  )
  const nextLaneRoutingState = normalizeChatLaneRoutingState(laneRoutingState, {
    activeConversationKey,
    draftConversationId,
    primaryIsStreaming: primaryLane.isStreaming,
    secondaryIsStreaming: secondaryLane.isStreaming,
    selectedConversationId
  })
  if (nextLaneRoutingState !== laneRoutingState) {
    setLaneRoutingState(nextLaneRoutingState)
  }

  const value = useMemo<ChatSessionContextValue>(
    () => ({
      activeConversation,
      activeConversationKey,
      activeMessages,
      chat,
      chatConversationId,
      conversations,
      createConversation: conversations.createConversation,
      deleteConversation: conversations.deleteConversation,
      draftConversationId,
      initialThread,
      isStreaming,
      liveMessagesWithModels,
      liveMode,
      messageCounts,
      renameConversation: conversations.renameConversation,
      selectConversation: conversations.selectConversation,
      setDraftConversationId,
      setMessageModels,
      setSessionModel,
      setSystemPrompt,
      systemPrompt,
      streamingConversationIds,
      updateThread
    }),
    [
      activeConversation,
      activeConversationKey,
      activeMessages,
      chat,
      chatConversationId,
      conversations,
      draftConversationId,
      initialThread,
      isStreaming,
      liveMessagesWithModels,
      liveMode,
      messageCounts,
      setSystemPrompt,
      streamingConversationIds,
      systemPrompt,
      updateThread
    ]
  )

  return <ChatSessionContext.Provider value={value}>{children}</ChatSessionContext.Provider>
}
