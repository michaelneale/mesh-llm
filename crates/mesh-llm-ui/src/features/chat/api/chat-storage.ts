import { openDB, type DBSchema } from 'idb'
import type { ConversationGroup, ThreadMessage } from '@/features/app-tabs/types'
import type { ChatConversation } from '@/features/chat/lib/chat-types'

export type ChatStorageScope = 'harness' | 'live'

export type ChatState = {
  conversations: ChatConversation[]
  conversationGroups: ConversationGroup[]
  threads: Record<string, ThreadMessage[]>
  activeConversationId: string
}

interface ChatDB extends DBSchema {
  state: {
    key: string
    value: ChatState
  }
}

const DB_NAME = 'mesh-llm-chat-db'
const DB_VERSION = 1
const STORE_NAME = 'state'
const STATE_KEYS: Record<ChatStorageScope, string> = {
  harness: 'chat-state:harness',
  live: 'chat-state:live'
}
export const MAX_CHAT_CONVERSATIONS = 80
export const MAX_CHAT_THREAD_MESSAGES = 120
export const MAX_CHAT_MESSAGE_BODY_CHARS = 20_000
const TRUNCATED_MESSAGE_SUFFIX = '\n\n[Message truncated in local history to keep the browser responsive.]'

const saveQueues: Record<ChatStorageScope, Promise<void>> = {
  harness: Promise.resolve(),
  live: Promise.resolve()
}

function trimConversations(conversations: ChatConversation[]): ChatConversation[] {
  return conversations.length > MAX_CHAT_CONVERSATIONS ? conversations.slice(0, MAX_CHAT_CONVERSATIONS) : conversations
}

function trimTextValue(value: string): string {
  if (value.length <= MAX_CHAT_MESSAGE_BODY_CHARS) return value

  const preservedLength = Math.max(0, MAX_CHAT_MESSAGE_BODY_CHARS - TRUNCATED_MESSAGE_SUFFIX.length)
  return `${value.slice(0, preservedLength).trimEnd()}${TRUNCATED_MESSAGE_SUFFIX}`
}

function trimThreadMessage(message: ThreadMessage): ThreadMessage {
  const trimmedMessage: ThreadMessage = {
    ...message,
    body: trimTextValue(message.body)
  }

  if (message.inspectMessage) {
    trimmedMessage.inspectMessage = { ...message.inspectMessage, text: trimTextValue(message.inspectMessage.text) }
  }

  return trimmedMessage
}

export function trimThreadMessages(messages: ThreadMessage[]): ThreadMessage[] {
  const retainedMessages =
    messages.length > MAX_CHAT_THREAD_MESSAGES ? messages.slice(-MAX_CHAT_THREAD_MESSAGES) : messages
  return retainedMessages.map(trimThreadMessage)
}

function pruneConversationGroups(
  conversationGroups: ConversationGroup[],
  retainedConversationIds: Set<string>
): ConversationGroup[] {
  return conversationGroups.map((group) => ({
    ...group,
    conversationIds: group.conversationIds.filter((conversationId) => retainedConversationIds.has(conversationId))
  }))
}

function pruneThreads(
  threads: Record<string, ThreadMessage[]>,
  retainedConversationIds: Set<string>
): Record<string, ThreadMessage[]> {
  const prunedThreads: Record<string, ThreadMessage[]> = {}

  for (const conversationId of retainedConversationIds) {
    const messages = threads[conversationId]
    if (messages) {
      prunedThreads[conversationId] = trimThreadMessages(messages)
    }
  }

  return prunedThreads
}

function normalizeActiveConversationId(activeConversationId: string, conversations: ChatConversation[]): string {
  if (conversations.some((conversation) => conversation.id === activeConversationId)) {
    return activeConversationId
  }
  return conversations[0]?.id ?? ''
}

async function openChatDB() {
  return openDB<ChatDB>(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME)
      }
    }
  })
}

export async function loadChatState(scope: ChatStorageScope): Promise<ChatState | undefined> {
  try {
    const db = await openChatDB()
    return await db.get(STORE_NAME, STATE_KEYS[scope])
  } catch {
    return undefined
  }
}

async function writeChatState(scope: ChatStorageScope, state: ChatState): Promise<void> {
  try {
    const db = await openChatDB()
    const conversations = trimConversations(state.conversations)
    const retainedConversationIds = new Set(conversations.map((conversation) => conversation.id))
    const trimmed: ChatState = {
      ...state,
      conversations,
      conversationGroups: pruneConversationGroups(state.conversationGroups, retainedConversationIds),
      threads: pruneThreads(state.threads, retainedConversationIds),
      activeConversationId: normalizeActiveConversationId(state.activeConversationId, conversations)
    }
    await db.put(STORE_NAME, trimmed, STATE_KEYS[scope])
  } catch (_) {
    void _
  }
}

export function saveChatState(scope: ChatStorageScope, state: ChatState): Promise<void> {
  const write = saveQueues[scope].then(
    () => writeChatState(scope, state),
    () => writeChatState(scope, state)
  )
  saveQueues[scope] = write.catch(() => undefined)
  return write
}

export async function clearChatState(scope: ChatStorageScope): Promise<void> {
  try {
    const db = await openChatDB()
    await db.delete(STORE_NAME, STATE_KEYS[scope])
  } catch (_) {
    void _
  }
}
