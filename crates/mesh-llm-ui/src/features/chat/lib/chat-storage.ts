import type { Conversation, ThreadMessage } from '@/features/app-tabs/types'
import type { ChatConversation, ChatMessage, ChatState } from '@/features/chat/lib/chat-types'
export type { ChatState as ChatStateExported } from '@/features/chat/lib/chat-types'
export {
  MAX_CHAT_CONVERSATIONS,
  loadChatState,
  saveChatState,
  trimThreadMessages,
  type ChatStorageScope
} from '@/features/chat/api/chat-storage'

/** Convert a full ChatConversation to a lightweight Conversation record. */
export function chatToConversation(chat: {
  id: string
  title: string
  subtitle: string
  updatedAt: string
}): Conversation {
  return {
    id: chat.id,
    title: chat.title,
    subtitle: chat.subtitle,
    updatedAt: chat.updatedAt
  }
}

export function createConversation(): ChatConversation {
  return {
    id: crypto.randomUUID(),
    title: 'New chat',
    subtitle: '',
    createdAt: Date.now(),
    updatedAt: String(Date.now()),
    messages: []
  }
}

export function createInitialChatState(): ChatState {
  return { conversations: [], conversationGroups: [], threads: {}, activeConversationId: '' }
}

export function findLastUserMessageIndex(messages: ThreadMessage[] | ChatMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if ('role' in msg && msg.role === 'user') return i
    if ('messageRole' in msg && msg.messageRole === 'user') return i
  }
  return -1
}

export async function loadPersistedChatState(
  _reader?: () => Promise<ChatState | null>
): Promise<ChatState | undefined> {
  const { loadChatState: load } = await import('@/features/chat/api/chat-storage')
  const state = await load('live')
  return state
}
