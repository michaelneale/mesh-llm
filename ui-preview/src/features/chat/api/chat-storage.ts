import { openDB, type DBSchema } from 'idb'
import type { Conversation, ConversationGroup, ThreadMessage } from '@/features/app-tabs/types'

export type ChatState = {
  conversations: Conversation[]
  conversationGroups: ConversationGroup[]
  threads: Record<string, ThreadMessage[]>
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
const STATE_KEY = 'chat-state'
const MAX_CONVERSATIONS = 80

async function openChatDB() {
  return openDB<ChatDB>(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME)
      }
    },
  })
}

export async function loadChatState(): Promise<ChatState | undefined> {
  try {
    const db = await openChatDB()
    return await db.get(STORE_NAME, STATE_KEY)
  } catch {
    return undefined
  }
}

export async function saveChatState(state: ChatState): Promise<void> {
  try {
    const db = await openChatDB()
    const trimmed: ChatState = {
      ...state,
      conversations:
        state.conversations.length > MAX_CONVERSATIONS
          ? state.conversations.slice(state.conversations.length - MAX_CONVERSATIONS)
          : state.conversations,
    }
    await db.put(STORE_NAME, trimmed, STATE_KEY)
  } catch (_) {
    void _
  }
}

export async function clearChatState(): Promise<void> {
  try {
    const db = await openChatDB()
    await db.delete(STORE_NAME, STATE_KEY)
  } catch (_) {
    void _
  }
}
