import { openDB, type DBSchema, type IDBPDatabase } from 'idb'
import type { ConversationGroup, ThreadMessage } from '@/features/app-tabs/types'
import type { ChatConversation } from '@/features/chat/lib/chat-types'

export type ChatStorageScope = 'harness' | 'live'

export type ChatState = {
  conversations: ChatConversation[]
  conversationGroups: ConversationGroup[]
  threads: Record<string, ThreadMessage[]>
  activeConversationId: string
}

type StoredChatStateEnvelope = {
  schemaVersion: 1
  savedAt: number
  recovery?: true
  state: ChatState
}

type StoredChatStateRecord = ChatState | StoredChatStateEnvelope

type ResolvedStoredChatState = {
  savedAt: number
  isRecovery: boolean
  state: ChatState
}

interface ChatDB extends DBSchema {
  state: {
    key: string
    value: StoredChatStateRecord
  }
}

const CHAT_STATE_SCHEMA_VERSION = 1
const DB_NAME = 'mesh-llm-chat-db'
const DB_VERSION = 2
const STORE_NAME = 'state'
const STATE_KEYS: Record<ChatStorageScope, string> = {
  harness: 'chat-state:harness',
  live: 'chat-state:live'
}
const LOCAL_STORAGE_KEYS: Record<ChatStorageScope, string> = {
  harness: 'mesh-llm.chat-state.harness',
  live: 'mesh-llm.chat-state.live'
}
/**
 * Key used by the v1 schema (old UI) to store chat state.
 * The v1 store used `keyPath: "id"` and stored a wrapper object
 * `{ id: "chat-state", state: <ChatState>, updatedAt: ... }`.
 */
const V1_STATE_KEY = 'chat-state'
export const MAX_CHAT_CONVERSATIONS = 80
export const MAX_CHAT_THREAD_MESSAGES = 120
export const MAX_CHAT_MESSAGE_BODY_CHARS = 20_000
const TRUNCATED_MESSAGE_SUFFIX = '\n\n[Message truncated in local history to keep the browser responsive.]'
const LOCAL_STORAGE_RECOVERY_CONVERSATION_LIMIT = 80
const LOCAL_STORAGE_RECOVERY_THREAD_MESSAGES = 40
const LOCAL_STORAGE_RECOVERY_MESSAGE_BODY_CHARS = 4_000

const saveQueues: Record<ChatStorageScope, Promise<void>> = {
  harness: Promise.resolve(),
  live: Promise.resolve()
}
let lastSavedAt = 0

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

function prepareChatStateForStorage(state: ChatState): ChatState {
  const conversations = trimConversations(state.conversations)
  const retainedConversationIds = new Set(conversations.map((conversation) => conversation.id))
  return {
    ...state,
    conversations,
    conversationGroups: pruneConversationGroups(state.conversationGroups, retainedConversationIds),
    threads: pruneThreads(state.threads, retainedConversationIds),
    activeConversationId: normalizeActiveConversationId(state.activeConversationId, conversations)
  }
}

function nextSavedAt(): number {
  const now = Date.now()
  lastSavedAt = Math.max(now, lastSavedAt + 1)
  return lastSavedAt
}

function createStoredChatStateEnvelope(
  state: ChatState,
  savedAt = nextSavedAt(),
  options: { recovery?: boolean } = {}
): StoredChatStateEnvelope {
  return {
    schemaVersion: CHAT_STATE_SCHEMA_VERSION,
    savedAt,
    ...(options.recovery ? { recovery: true } : {}),
    state
  }
}

function localChatStorage(): Storage | undefined {
  if (typeof window === 'undefined') return undefined

  try {
    return window.localStorage
  } catch (error) {
    void error
    return undefined
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isThreadRecord(value: unknown): value is Record<string, ThreadMessage[]> {
  if (!isRecord(value)) return false

  return Object.values(value).every(
    (messages) => Array.isArray(messages) && messages.every((message) => isThreadMessage(message))
  )
}

function isChatConversation(value: unknown): value is ChatConversation {
  if (!isRecord(value)) return false

  return (
    typeof value.id === 'string' &&
    typeof value.title === 'string' &&
    (value.subtitle === undefined || typeof value.subtitle === 'string') &&
    (value.createdAt === undefined || typeof value.createdAt === 'number') &&
    typeof value.updatedAt === 'string' &&
    (value.messages === undefined || Array.isArray(value.messages))
  )
}

function isConversationGroup(value: unknown): value is ConversationGroup {
  if (!isRecord(value)) return false

  return (
    typeof value.title === 'string' &&
    Array.isArray(value.conversationIds) &&
    value.conversationIds.every((conversationId) => typeof conversationId === 'string')
  )
}

function isThreadMessage(value: unknown): value is ThreadMessage {
  if (!isRecord(value)) return false

  return (
    typeof value.id === 'string' &&
    (value.messageRole === 'user' || value.messageRole === 'assistant') &&
    typeof value.timestamp === 'string' &&
    typeof value.body === 'string'
  )
}

function isChatState(value: unknown): value is ChatState {
  if (!isRecord(value)) return false

  return (
    Array.isArray(value.conversations) &&
    value.conversations.every((conversation) => isChatConversation(conversation)) &&
    Array.isArray(value.conversationGroups) &&
    value.conversationGroups.every((group) => isConversationGroup(group)) &&
    isThreadRecord(value.threads) &&
    typeof value.activeConversationId === 'string'
  )
}

function isStoredChatStateEnvelope(value: unknown): value is StoredChatStateEnvelope {
  if (!isRecord(value)) return false

  return (
    value.schemaVersion === CHAT_STATE_SCHEMA_VERSION &&
    typeof value.savedAt === 'number' &&
    Number.isFinite(value.savedAt) &&
    (value.recovery === undefined || value.recovery === true) &&
    isChatState(value.state)
  )
}

function resolveStoredChatState(record: unknown): ResolvedStoredChatState | undefined {
  if (isStoredChatStateEnvelope(record)) {
    return {
      savedAt: record.savedAt,
      isRecovery: record.recovery === true,
      state: prepareChatStateForStorage(record.state)
    }
  }

  if (isChatState(record)) {
    return {
      savedAt: 0,
      isRecovery: false,
      state: prepareChatStateForStorage(record)
    }
  }

  return undefined
}

function newestStoredChatState(
  indexedState: ResolvedStoredChatState | undefined,
  localState: ResolvedStoredChatState | undefined
): ResolvedStoredChatState | undefined {
  if (!indexedState) return localState
  if (!localState) return indexedState
  if (localState.savedAt !== indexedState.savedAt) {
    return localState.savedAt > indexedState.savedAt ? localState : indexedState
  }
  if (localState.isRecovery !== indexedState.isRecovery) {
    return localState.isRecovery ? indexedState : localState
  }
  return localState
}

function trimTextForLocalRecovery(value: string): string {
  if (value.length <= LOCAL_STORAGE_RECOVERY_MESSAGE_BODY_CHARS) return value

  return `${value.slice(0, LOCAL_STORAGE_RECOVERY_MESSAGE_BODY_CHARS).trimEnd()}${TRUNCATED_MESSAGE_SUFFIX}`
}

function trimThreadMessageForLocalRecovery(message: ThreadMessage): ThreadMessage {
  const trimmedMessage: ThreadMessage = {
    ...message,
    body: trimTextForLocalRecovery(message.body)
  }

  if (message.inspectMessage) {
    trimmedMessage.inspectMessage = {
      ...message.inspectMessage,
      text: trimTextForLocalRecovery(message.inspectMessage.text)
    }
  }

  return trimmedMessage
}

function createLocalRecoveryState(state: ChatState): ChatState {
  const conversations = trimConversations(state.conversations).slice(0, LOCAL_STORAGE_RECOVERY_CONVERSATION_LIMIT)
  const retainedConversationIds = new Set(conversations.map((conversation) => conversation.id))
  const activeConversationId = normalizeActiveConversationId(state.activeConversationId, conversations)
  const threads: Record<string, ThreadMessage[]> = {}
  const activeThread = state.threads[activeConversationId]

  if (activeThread) {
    threads[activeConversationId] = activeThread
      .slice(-LOCAL_STORAGE_RECOVERY_THREAD_MESSAGES)
      .map(trimThreadMessageForLocalRecovery)
  }

  return {
    ...state,
    conversations,
    conversationGroups: pruneConversationGroups(state.conversationGroups, retainedConversationIds),
    threads,
    activeConversationId
  }
}

function loadLocalChatState(scope: ChatStorageScope): ResolvedStoredChatState | undefined {
  const storage = localChatStorage()
  if (!storage) return undefined

  try {
    const rawState = storage.getItem(LOCAL_STORAGE_KEYS[scope])
    if (!rawState) return undefined

    const parsedState: unknown = JSON.parse(rawState)
    const state = resolveStoredChatState(parsedState)
    if (state) return state

    storage.removeItem(LOCAL_STORAGE_KEYS[scope])
    return undefined
  } catch (error) {
    void error
    storage.removeItem(LOCAL_STORAGE_KEYS[scope])
    return undefined
  }
}

function saveLocalChatState(scope: ChatStorageScope, envelope: StoredChatStateEnvelope): void {
  const storage = localChatStorage()
  if (!storage) return

  try {
    storage.setItem(LOCAL_STORAGE_KEYS[scope], JSON.stringify(envelope))
  } catch (error) {
    void error

    try {
      storage.setItem(
        LOCAL_STORAGE_KEYS[scope],
        JSON.stringify(
          createStoredChatStateEnvelope(createLocalRecoveryState(envelope.state), envelope.savedAt, { recovery: true })
        )
      )
    } catch (recoveryError) {
      void recoveryError
    }
  }
}

function clearLocalChatState(scope: ChatStorageScope): void {
  const storage = localChatStorage()
  if (!storage) return
  try {
    storage.removeItem(LOCAL_STORAGE_KEYS[scope])
  } catch (error) {
    void error
  }
}

function v1SavedAt(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value !== 'string') return undefined

  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

async function readV1ChatState(): Promise<StoredChatStateEnvelope | undefined> {
  if (typeof indexedDB === 'undefined') return undefined

  return new Promise((resolve) => {
    let db: IDBDatabase | undefined
    const openRequest = indexedDB.open(DB_NAME)

    openRequest.onerror = () => resolve(undefined)
    openRequest.onupgradeneeded = () => {
      openRequest.transaction?.abort()
    }
    openRequest.onsuccess = () => {
      db = openRequest.result
      try {
        if (db.version >= DB_VERSION) {
          db.close()
          resolve(undefined)
          return
        }
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.close()
          resolve(undefined)
          return
        }

        const tx = db.transaction(STORE_NAME, 'readonly')
        const store = tx.objectStore(STORE_NAME)
        const getRequest = store.get(V1_STATE_KEY)

        getRequest.onsuccess = () => {
          db?.close()
          const record: unknown = getRequest.result
          if (!isRecord(record) || !isChatState(record.state)) {
            resolve(undefined)
            return
          }

          resolve(createStoredChatStateEnvelope(record.state, v1SavedAt(record.updatedAt)))
        }
        getRequest.onerror = () => {
          db?.close()
          resolve(undefined)
        }
      } catch {
        db?.close()
        resolve(undefined)
      }
    }
  })
}

let openPromise: Promise<IDBPDatabase<ChatDB>> | null = null

async function openChatDB(): Promise<IDBPDatabase<ChatDB>> {
  if (!openPromise) {
    openPromise = openChatDBOnce().catch((error) => {
      openPromise = null
      throw error
    })
  }
  return openPromise
}

async function openChatDBOnce(): Promise<IDBPDatabase<ChatDB>> {
  const migratedState = await readV1ChatState()

  const db = await openDB<ChatDB>(DB_NAME, DB_VERSION, {
    upgrade(database, oldVersion, _newVersion, transaction) {
      const hasExistingStore = oldVersion < 2 && database.objectStoreNames.contains(STORE_NAME)
      const shouldRecreateLegacyStore = hasExistingStore && transaction.objectStore(STORE_NAME).keyPath !== null

      if (shouldRecreateLegacyStore) {
        database.deleteObjectStore(STORE_NAME)
      }
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME)
      }
    }
  })

  if (migratedState) {
    try {
      await db.put(STORE_NAME, migratedState, STATE_KEYS.live)
    } catch (error) {
      void error
    }
  }

  return db
}

export async function loadChatState(scope: ChatStorageScope): Promise<ChatState | undefined> {
  const localState = loadLocalChatState(scope)

  try {
    const db = await openChatDB()
    const indexedState = resolveStoredChatState(await db.get(STORE_NAME, STATE_KEYS[scope]))
    const selectedState = newestStoredChatState(indexedState, localState)
    if (
      selectedState &&
      selectedState === localState &&
      !selectedState.isRecovery &&
      (!indexedState || localState.savedAt > indexedState.savedAt)
    ) {
      queueChatStateWrite(scope, createStoredChatStateEnvelope(selectedState.state, selectedState.savedAt), {
        onlyIfNewer: true
      })
    }

    return selectedState?.state
  } catch {
    return localState?.state
  }
}

async function writeChatState(scope: ChatStorageScope, envelope: StoredChatStateEnvelope): Promise<void> {
  try {
    const db = await openChatDB()
    await db.put(STORE_NAME, envelope, STATE_KEYS[scope])
  } catch (_) {
    void _
  }
}

async function writeChatStateIfNewer(scope: ChatStorageScope, envelope: StoredChatStateEnvelope): Promise<void> {
  try {
    const db = await openChatDB()
    const currentState = resolveStoredChatState(await db.get(STORE_NAME, STATE_KEYS[scope]))
    if (currentState && currentState.savedAt > envelope.savedAt) return
    await db.put(STORE_NAME, envelope, STATE_KEYS[scope])
  } catch (_) {
    void _
  }
}

function queueChatStateWrite(
  scope: ChatStorageScope,
  envelope: StoredChatStateEnvelope,
  options: { onlyIfNewer?: boolean } = {}
): Promise<void> {
  const write = saveQueues[scope].then(
    () => (options.onlyIfNewer ? writeChatStateIfNewer(scope, envelope) : writeChatState(scope, envelope)),
    () => (options.onlyIfNewer ? writeChatStateIfNewer(scope, envelope) : writeChatState(scope, envelope))
  )
  saveQueues[scope] = write.catch(() => undefined)
  return write
}

export function saveChatState(scope: ChatStorageScope, state: ChatState): Promise<void> {
  const persistedState = prepareChatStateForStorage(state)
  const envelope = createStoredChatStateEnvelope(persistedState)
  saveLocalChatState(scope, envelope)
  return queueChatStateWrite(scope, envelope)
}

export async function clearChatState(scope: ChatStorageScope): Promise<void> {
  clearLocalChatState(scope)
  try {
    const db = await openChatDB()
    await db.delete(STORE_NAME, STATE_KEYS[scope])
  } catch (_) {
    void _
  }
}
