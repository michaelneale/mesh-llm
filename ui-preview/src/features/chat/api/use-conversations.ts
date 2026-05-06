import { useState, useEffect } from 'react'
import type { Conversation, ConversationGroup, ChatHarnessData, ThreadMessage } from '@/features/app-tabs/types'
import { loadChatState, saveChatState } from './chat-storage'

type ResolvedChatState = Pick<ChatHarnessData, 'conversations' | 'conversationGroups' | 'threads'>

function normalizeChatState(state: ResolvedChatState | undefined, fallback: ChatHarnessData): ResolvedChatState {
  if (!state) {
    return fallback
  }

  return {
    conversations: Array.isArray(state.conversations) ? state.conversations : fallback.conversations,
    conversationGroups: Array.isArray(state.conversationGroups)
      ? state.conversationGroups
      : fallback.conversationGroups,
    threads:
      state.threads && typeof state.threads === 'object' && !Array.isArray(state.threads)
        ? state.threads
        : fallback.threads
  }
}

export function useConversations(fallback: ChatHarnessData) {
  const [conversations, setConversations] = useState<Conversation[]>(fallback.conversations)
  const [conversationGroups, setConversationGroups] = useState<ConversationGroup[]>(fallback.conversationGroups)
  const [threads, setThreads] = useState<Record<string, ThreadMessage[]>>(fallback.threads)

  useEffect(() => {
    let cancelled = false

    loadChatState().then((state) => {
      if (cancelled) return

      const nextState = normalizeChatState(state, fallback)
      setConversations(nextState.conversations)
      setConversationGroups(nextState.conversationGroups)
      setThreads(nextState.threads)
    })

    return () => {
      cancelled = true
    }
  }, [fallback])

  function addConversation(conv: Conversation): void {
    const next = [conv, ...conversations]
    setConversations(next)
    void saveChatState({ conversations: next, conversationGroups, threads })
  }

  function updateThread(id: string, messages: ThreadMessage[]): void {
    const next = { ...threads, [id]: messages }
    setThreads(next)
    void saveChatState({ conversations, conversationGroups, threads: next })
  }

  return { conversations, conversationGroups, threads, addConversation, updateThread }
}
