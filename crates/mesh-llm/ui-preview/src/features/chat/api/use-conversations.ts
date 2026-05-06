import { useCallback, useEffect, useState } from 'react'
import type { Conversation, ConversationGroup, ChatHarnessData, ThreadMessage } from '@/features/app-tabs/types'
import {
  MAX_CHAT_CONVERSATIONS,
  loadChatState,
  saveChatState,
  trimThreadMessages,
  type ChatState,
  type ChatStorageScope
} from './chat-storage'
import { buildConversationGroups } from './conversation-groups'

type ResolvedChatState = Pick<ChatState, 'conversations' | 'conversationGroups' | 'threads' | 'activeConversationId'>

const DEFAULT_CONVERSATION_TITLE = 'New chat'

function cloneConversations(conversations: Conversation[]): Conversation[] {
  return conversations.map((conversation) => ({ ...conversation }))
}

function cloneConversationGroups(conversationGroups: ConversationGroup[]): ConversationGroup[] {
  return conversationGroups.map((group) => ({ ...group, conversationIds: [...group.conversationIds] }))
}

function trimConversations(conversations: Conversation[]): Conversation[] {
  return conversations.length > MAX_CHAT_CONVERSATIONS ? conversations.slice(0, MAX_CHAT_CONVERSATIONS) : conversations
}

function cloneThreadsForConversations(
  threads: Record<string, ThreadMessage[]>,
  conversations: Conversation[]
): Record<string, ThreadMessage[]> {
  const clonedThreads: Record<string, ThreadMessage[]> = {}

  for (const conversation of conversations) {
    const messages = threads[conversation.id]
    clonedThreads[conversation.id] = Array.isArray(messages) ? trimThreadMessages(messages) : []
  }

  return clonedThreads
}

function retainThreadsForConversations(
  threads: Record<string, ThreadMessage[]>,
  conversations: Conversation[],
  updatedThread?: { conversationId: string; messages: ThreadMessage[] }
): Record<string, ThreadMessage[]> {
  const retainedThreads: Record<string, ThreadMessage[]> = {}

  for (const conversation of conversations) {
    if (updatedThread && conversation.id === updatedThread.conversationId) {
      retainedThreads[conversation.id] = updatedThread.messages
      continue
    }

    retainedThreads[conversation.id] = threads[conversation.id] ?? []
  }

  return retainedThreads
}

function normalizeConversationGroups(
  conversations: Conversation[],
  conversationGroups: ConversationGroup[]
): ConversationGroup[] {
  return buildConversationGroups(conversations, conversationGroups)
}

function normalizeActiveConversationId(
  activeConversationId: string | undefined,
  conversations: Conversation[]
): string {
  if (!conversations.length) return ''
  if (activeConversationId && conversations.some((conversation) => conversation.id === activeConversationId)) {
    return activeConversationId
  }
  return conversations[0]?.id ?? ''
}

function createFallbackState(fallback: ChatHarnessData): ResolvedChatState {
  const conversations = trimConversations(cloneConversations(fallback.conversations))
  const conversationGroups = cloneConversationGroups(fallback.conversationGroups)
  const threads = cloneThreadsForConversations(fallback.threads, conversations)
  return {
    conversations,
    conversationGroups,
    threads,
    activeConversationId: fallback.conversations[0]?.id ?? ''
  }
}

function normalizeChatState(state: ChatState | undefined, fallback: ChatHarnessData): ResolvedChatState {
  const fallbackState = createFallbackState(fallback)
  if (!state) {
    return fallbackState
  }

  const conversations = Array.isArray(state.conversations)
    ? trimConversations(cloneConversations(state.conversations))
    : fallbackState.conversations
  const conversationGroups = Array.isArray(state.conversationGroups)
    ? cloneConversationGroups(state.conversationGroups)
    : fallbackState.conversationGroups
  const threads =
    state.threads && typeof state.threads === 'object' && !Array.isArray(state.threads)
      ? cloneThreadsForConversations(state.threads, conversations)
      : fallbackState.threads

  return {
    conversations,
    conversationGroups: normalizeConversationGroups(conversations, conversationGroups),
    threads,
    activeConversationId: normalizeActiveConversationId(state.activeConversationId, conversations)
  }
}

function createConversationId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `conversation-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function createConversationRecord(): Conversation {
  return {
    id: createConversationId(),
    title: DEFAULT_CONVERSATION_TITLE,
    subtitle: '',
    updatedAt: new Date().toISOString()
  }
}

function deriveConversationTitle(messages: ThreadMessage[]): string {
  const firstUserLine = messages
    .filter((message) => message.messageRole === 'user')
    .flatMap((message) => message.body.split(/\r?\n/))
    .map((line) => line.trim())
    .find((line) => line.length > 0)

  if (!firstUserLine) return DEFAULT_CONVERSATION_TITLE

  return (
    firstUserLine
      .replace(/[\r\n]+/g, ' ')
      .trim()
      .slice(0, 60)
      .trimEnd() || DEFAULT_CONVERSATION_TITLE
  )
}

function normalizeConversationTitle(title: string): string {
  const trimmedTitle = title.trim().replace(/\s+/g, ' ')
  return trimmedTitle.slice(0, 140).trimEnd() || DEFAULT_CONVERSATION_TITLE
}

function reorderConversations(conversations: Conversation[], updatedConversation: Conversation): Conversation[] {
  return [updatedConversation, ...conversations.filter((conversation) => conversation.id !== updatedConversation.id)]
}

export function useConversations(fallback: ChatHarnessData, scope: ChatStorageScope) {
  const [state, setState] = useState<ResolvedChatState>(() => createFallbackState(fallback))

  useEffect(() => {
    let cancelled = false

    loadChatState(scope).then((storedState) => {
      if (cancelled) return

      setState(normalizeChatState(storedState, fallback))
    })

    return () => {
      cancelled = true
    }
  }, [fallback, scope])

  const persistState = useCallback(
    async (nextState: ResolvedChatState) => {
      await saveChatState(scope, nextState)
    },
    [scope]
  )

  const selectConversation = useCallback(
    (conversationId: string) => {
      setState((currentState) => {
        const activeConversationId = normalizeActiveConversationId(conversationId, currentState.conversations)
        const nextState =
          activeConversationId === currentState.activeConversationId
            ? currentState
            : { ...currentState, activeConversationId }

        if (nextState !== currentState) {
          void persistState(nextState)
        }

        return nextState
      })
    },
    [persistState]
  )

  const createConversation = useCallback(
    (conversationId?: string) => {
      const conversation = createConversationRecord()
      const nextConversation = conversationId ? { ...conversation, id: conversationId } : conversation

      setState((currentState) => {
        const existingConversation = currentState.conversations.find((item) => item.id === nextConversation.id)
        if (existingConversation) {
          const nextState =
            currentState.activeConversationId === existingConversation.id
              ? currentState
              : { ...currentState, activeConversationId: existingConversation.id }

          if (nextState !== currentState) {
            void persistState(nextState)
          }

          return nextState
        }

        const conversations = trimConversations([nextConversation, ...currentState.conversations])
        const threads = retainThreadsForConversations(currentState.threads, conversations, {
          conversationId: nextConversation.id,
          messages: []
        })
        const nextState = {
          conversations,
          conversationGroups: normalizeConversationGroups(conversations, currentState.conversationGroups),
          threads,
          activeConversationId: nextConversation.id
        }
        void persistState(nextState)
        return nextState
      })

      return nextConversation.id
    },
    [persistState]
  )

  const updateThread = useCallback(
    (conversationId: string, messages: ThreadMessage[]) => {
      setState((currentState) => {
        const existingConversation = currentState.conversations.find(
          (conversation) => conversation.id === conversationId
        )
        if (!existingConversation) return currentState

        const updatedMessages = trimThreadMessages(messages)
        const updatedConversation = {
          ...existingConversation,
          title: deriveConversationTitle(messages),
          updatedAt: messages[messages.length - 1]?.timestamp ?? existingConversation.updatedAt
        }
        const conversations = trimConversations(reorderConversations(currentState.conversations, updatedConversation))
        const threads = retainThreadsForConversations(currentState.threads, conversations, {
          conversationId,
          messages: updatedMessages
        })
        const nextState = {
          conversations,
          conversationGroups: normalizeConversationGroups(conversations, currentState.conversationGroups),
          threads,
          activeConversationId: normalizeActiveConversationId(
            currentState.activeConversationId || conversationId,
            conversations
          )
        }

        void persistState(nextState)
        return nextState
      })
    },
    [persistState]
  )

  const renameConversation = useCallback(
    (conversationId: string, title: string) => {
      setState((currentState) => {
        const existingConversation = currentState.conversations.find(
          (conversation) => conversation.id === conversationId
        )
        if (!existingConversation) return currentState

        const updatedConversation = {
          ...existingConversation,
          title: normalizeConversationTitle(title),
          updatedAt: new Date().toISOString()
        }
        const conversations = trimConversations(reorderConversations(currentState.conversations, updatedConversation))
        const nextState = {
          ...currentState,
          conversations,
          conversationGroups: normalizeConversationGroups(conversations, currentState.conversationGroups),
          threads: retainThreadsForConversations(currentState.threads, conversations),
          activeConversationId: normalizeActiveConversationId(currentState.activeConversationId, conversations)
        }

        void persistState(nextState)
        return nextState
      })
    },
    [persistState]
  )

  const deleteConversation = useCallback(
    (conversationId: string) => {
      setState((currentState) => {
        const conversations = currentState.conversations.filter((conversation) => conversation.id !== conversationId)
        if (conversations.length === currentState.conversations.length) return currentState

        const threads = retainThreadsForConversations(currentState.threads, conversations)
        const activeConversationId =
          currentState.activeConversationId === conversationId
            ? (conversations[0]?.id ?? '')
            : normalizeActiveConversationId(currentState.activeConversationId, conversations)
        const nextState = {
          conversations,
          conversationGroups: normalizeConversationGroups(conversations, currentState.conversationGroups),
          threads,
          activeConversationId
        }

        void persistState(nextState)
        return nextState
      })
    },
    [persistState]
  )

  return {
    conversations: state.conversations,
    conversationGroups: state.conversationGroups,
    threads: state.threads,
    activeConversationId: state.activeConversationId,
    selectConversation,
    createConversation,
    updateThread,
    renameConversation,
    deleteConversation
  }
}
