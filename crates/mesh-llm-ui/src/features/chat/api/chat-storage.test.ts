import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ThreadMessage } from '@/features/app-tabs/types'
import {
  MAX_CHAT_CONVERSATIONS,
  MAX_CHAT_MESSAGE_BODY_CHARS,
  MAX_CHAT_THREAD_MESSAGES,
  clearChatState,
  loadChatState,
  saveChatState,
  trimThreadMessages,
  type ChatState
} from '@/features/chat/api/chat-storage'

type StoredChatStateEnvelope = {
  schemaVersion: 1
  savedAt: number
  recovery?: true
  state: ChatState
}

type StoredChatStateRecord = ChatState | StoredChatStateEnvelope

const mockStore = new Map<string, StoredChatStateRecord>()
const putDelays: Promise<void>[] = []
const LIVE_STATE_KEY = 'chat-state:live'
const LIVE_LOCAL_STORAGE_KEY = 'mesh-llm.chat-state.live'

vi.mock('idb', () => ({
  openDB: vi.fn(async () => ({
    get: vi.fn(async (_storeName: string, key: string) => mockStore.get(key)),
    put: vi.fn(async (_storeName: string, value: StoredChatStateRecord, key: string) => {
      await putDelays.shift()
      mockStore.set(key, value)
    }),
    delete: vi.fn(async (_storeName: string, key: string) => {
      mockStore.delete(key)
    })
  }))
}))

function buildState(id: string, title: string): ChatState {
  return {
    conversations: [{ id, title, subtitle: '', createdAt: Date.now(), updatedAt: String(Date.now()), messages: [] }],
    conversationGroups: [
      { title: 'Today', conversationIds: [id] },
      { title: 'Earlier', conversationIds: [] }
    ],
    threads: { [id]: [] },
    activeConversationId: id
  }
}

function buildStoredEnvelope(state: ChatState, savedAt: number, recovery = false): StoredChatStateEnvelope {
  return {
    schemaVersion: 1,
    savedAt,
    ...(recovery ? { recovery: true as const } : {}),
    state
  }
}

describe('chat-storage', () => {
  beforeEach(() => {
    mockStore.clear()
    putDelays.length = 0
    localStorage.clear()
  })

  it('stores live and harness chat state in separate scopes', async () => {
    const harnessState = buildState('harness-1', 'Harness conversation')
    const liveState = buildState('live-1', 'Live conversation')

    await saveChatState('harness', harnessState)
    await saveChatState('live', liveState)

    await expect(loadChatState('harness')).resolves.toEqual(harnessState)
    await expect(loadChatState('live')).resolves.toEqual(liveState)
  })

  it('restores the latest local conversation when reload happens before IndexedDB finishes saving', async () => {
    let releaseFirstWrite = () => {}
    putDelays.push(
      new Promise<void>((resolve) => {
        releaseFirstWrite = resolve
      })
    )
    const liveState = buildState('live-reload', 'Refresh-safe conversation')
    liveState.threads['live-reload'] = [
      { id: 'message-1', messageRole: 'user', timestamp: 'Now', body: 'This should survive Cmd-R' }
    ]

    const pendingSave = saveChatState('live', liveState)
    await Promise.resolve()

    await expect(loadChatState('live')).resolves.toMatchObject({
      conversations: [{ id: 'live-reload', title: 'Refresh-safe conversation' }],
      threads: { 'live-reload': [{ id: 'message-1', body: 'This should survive Cmd-R' }] },
      activeConversationId: 'live-reload'
    })

    releaseFirstWrite()
    await pendingSave
  })

  it('prefers a newer local conversation over stale IndexedDB during refresh recovery', async () => {
    const oldState = buildState('live-old', 'Old IndexedDB conversation')
    await saveChatState('live', oldState)

    let releaseNewWrite = () => {}
    putDelays.push(
      new Promise<void>((resolve) => {
        releaseNewWrite = resolve
      })
    )
    const newerState = buildState('live-new', 'New local conversation')
    newerState.threads['live-new'] = [
      { id: 'message-new', messageRole: 'assistant', timestamp: 'Later', body: 'Newest local reply' }
    ]

    const pendingSave = saveChatState('live', newerState)
    await Promise.resolve()

    await expect(loadChatState('live')).resolves.toMatchObject({
      conversations: [{ id: 'live-new', title: 'New local conversation' }],
      threads: { 'live-new': [{ id: 'message-new', body: 'Newest local reply' }] },
      activeConversationId: 'live-new'
    })

    releaseNewWrite()
    await pendingSave
  })

  it('queues refresh backfill so it cannot overwrite a newer save', async () => {
    const staleIndexedState = buildState('live-old', 'Old IndexedDB conversation')
    const localState = buildState('live-local', 'Local state to backfill')
    const newerState = buildState('live-newer', 'Newer saved state')
    newerState.threads['live-newer'] = [
      { id: 'newer-message', messageRole: 'user', timestamp: 'After reload', body: 'Newest user message' }
    ]
    mockStore.set(LIVE_STATE_KEY, buildStoredEnvelope(staleIndexedState, 1))
    localStorage.setItem(LIVE_LOCAL_STORAGE_KEY, JSON.stringify(buildStoredEnvelope(localState, 2)))

    let releaseBackfill = () => {}
    putDelays.push(
      new Promise<void>((resolve) => {
        releaseBackfill = resolve
      })
    )

    await expect(loadChatState('live')).resolves.toMatchObject({
      conversations: [{ id: 'live-local', title: 'Local state to backfill' }],
      activeConversationId: 'live-local'
    })

    const newerSave = saveChatState('live', newerState)
    await Promise.resolve()
    releaseBackfill()
    await newerSave

    const storedState = mockStore.get(LIVE_STATE_KEY) as StoredChatStateEnvelope | undefined
    expect(storedState?.state).toMatchObject({
      conversations: [{ id: 'live-newer', title: 'Newer saved state' }],
      threads: { 'live-newer': [{ id: 'newer-message', body: 'Newest user message' }] },
      activeConversationId: 'live-newer'
    })
  })

  it('does not promote quota recovery snapshots over full IndexedDB snapshots', async () => {
    const fullState = buildState('live-active', 'Full active conversation')
    fullState.conversations.push({
      id: 'live-secondary',
      title: 'Full secondary conversation',
      subtitle: '',
      createdAt: Date.now(),
      updatedAt: String(Date.now()),
      messages: []
    })
    fullState.conversationGroups[0]!.conversationIds.push('live-secondary')
    fullState.threads['live-active'] = [
      { id: 'active-message', messageRole: 'assistant', timestamp: 'Now', body: 'Active thread survives recovery' }
    ]
    fullState.threads['live-secondary'] = [
      { id: 'secondary-message', messageRole: 'user', timestamp: 'Earlier', body: 'Secondary thread must stay in IDB' }
    ]

    const originalSetItem = Storage.prototype.setItem
    const setItem = vi.spyOn(Storage.prototype, 'setItem')
    setItem.mockImplementation(function setItemWithQuotaFallback(this: Storage, key: string, value: string) {
      const parsedValue = JSON.parse(value) as Partial<StoredChatStateEnvelope>
      if (key === LIVE_LOCAL_STORAGE_KEY && parsedValue.recovery !== true) {
        throw new DOMException('Quota exceeded', 'QuotaExceededError')
      }
      return Reflect.apply(originalSetItem, this, [key, value])
    })

    try {
      await saveChatState('live', fullState)
    } finally {
      setItem.mockRestore()
    }

    await expect(loadChatState('live')).resolves.toMatchObject({
      conversations: [
        { id: 'live-active', title: 'Full active conversation' },
        { id: 'live-secondary', title: 'Full secondary conversation' }
      ],
      threads: {
        'live-active': [{ id: 'active-message', body: 'Active thread survives recovery' }],
        'live-secondary': [{ id: 'secondary-message', body: 'Secondary thread must stay in IDB' }]
      },
      activeConversationId: 'live-active'
    })
  })

  it('ignores malformed envelopes with non-finite timestamps', async () => {
    const indexedState = buildState('live-indexed', 'Valid IndexedDB state')
    const malformedLocalState = buildState('live-malformed', 'Malformed local state')
    mockStore.set(LIVE_STATE_KEY, buildStoredEnvelope(indexedState, 1))
    localStorage.setItem(
      LIVE_LOCAL_STORAGE_KEY,
      `{"schemaVersion":1,"savedAt":1e309,"state":${JSON.stringify(malformedLocalState)}}`
    )

    await expect(loadChatState('live')).resolves.toMatchObject({
      conversations: [{ id: 'live-indexed', title: 'Valid IndexedDB state' }],
      activeConversationId: 'live-indexed'
    })
    expect(localStorage.getItem(LIVE_LOCAL_STORAGE_KEY)).toBeNull()
  })

  it('prefers legacy local state over legacy IndexedDB state on timestamp ties', async () => {
    const legacyIndexedState = buildState('live-indexed-legacy', 'Legacy IndexedDB state')
    const legacyLocalState = buildState('live-local-legacy', 'Legacy local state')
    mockStore.set(LIVE_STATE_KEY, legacyIndexedState)
    localStorage.setItem(LIVE_LOCAL_STORAGE_KEY, JSON.stringify(legacyLocalState))

    await expect(loadChatState('live')).resolves.toMatchObject({
      conversations: [{ id: 'live-local-legacy', title: 'Legacy local state' }],
      activeConversationId: 'live-local-legacy'
    })
  })

  it('prunes threads for conversations beyond MAX_CHAT_CONVERSATIONS and normalizes active id', async () => {
    const conversations = Array.from({ length: MAX_CHAT_CONVERSATIONS + 5 }, (_, index) => ({
      id: `conv-${index}`,
      title: `Conversation ${index}`,
      subtitle: '',
      createdAt: Date.now(),
      updatedAt: String(Date.now()),
      messages: []
    }))
    const threads: Record<string, []> = {}
    for (const conversation of conversations) {
      threads[conversation.id] = []
    }
    const state: ChatState = {
      conversations,
      conversationGroups: [
        { title: 'Today', conversationIds: conversations.slice(0, 2).map((c) => c.id) },
        { title: 'Earlier', conversationIds: conversations.slice(2).map((c) => c.id) }
      ],
      threads,
      activeConversationId: conversations.at(-1)!.id
    }

    await saveChatState('live', state)

    const saved = await loadChatState('live')
    expect(saved?.conversations).toHaveLength(MAX_CHAT_CONVERSATIONS)
    expect(Object.keys(saved?.threads ?? {})).toHaveLength(MAX_CHAT_CONVERSATIONS)
    expect(saved?.activeConversationId).toBe(conversations[0]?.id)

    const retainedIds = new Set(saved?.conversations.map((c) => c.id))
    for (const groupedId of (saved?.conversationGroups ?? []).flatMap((g) => g.conversationIds)) {
      expect(retainedIds.has(groupedId)).toBe(true)
    }

    for (const orphanId of conversations.slice(MAX_CHAT_CONVERSATIONS).map((c) => c.id)) {
      expect(saved?.threads[orphanId]).toBeUndefined()
    }
  })

  it('bounds retained thread messages and oversized message text', async () => {
    const messages: ThreadMessage[] = Array.from({ length: MAX_CHAT_THREAD_MESSAGES + 5 }, (_, index) => ({
      id: `message-${index}`,
      messageRole: index % 2 === 0 ? 'user' : 'assistant',
      timestamp: `T${index}`,
      body: index === MAX_CHAT_THREAD_MESSAGES + 4 ? 'x'.repeat(MAX_CHAT_MESSAGE_BODY_CHARS + 1000) : `Body ${index}`,
      inspectMessage:
        index === MAX_CHAT_THREAD_MESSAGES + 4
          ? {
              kind: 'assistant',
              id: 'inspect-large',
              text: 'y'.repeat(MAX_CHAT_MESSAGE_BODY_CHARS + 1000),
              at: 'T-final',
              servedBy: 'local',
              route: [],
              model: 'model-a',
              receipt: 'receipt',
              metrics: { rttMs: 1, ttftMs: 1, throughput: '1 tok/s', tokens: 1 },
              decisions: [],
              trace: []
            }
          : undefined
    }))

    const trimmedMessages = trimThreadMessages(messages)
    expect(trimmedMessages).toHaveLength(MAX_CHAT_THREAD_MESSAGES)
    expect(trimmedMessages[0]?.id).toBe('message-5')
    expect(trimmedMessages.at(-1)?.body.length).toBeLessThanOrEqual(MAX_CHAT_MESSAGE_BODY_CHARS)
    expect(trimmedMessages.at(-1)?.body).toContain('Message truncated in local history')
    expect(trimmedMessages.at(-1)?.inspectMessage?.text.length).toBeLessThanOrEqual(MAX_CHAT_MESSAGE_BODY_CHARS)

    const state = buildState('live-1', 'Large thread')
    state.threads['live-1'] = messages
    await saveChatState('live', state)

    const saved = await loadChatState('live')
    expect(saved?.threads['live-1']).toHaveLength(MAX_CHAT_THREAD_MESSAGES)
    expect(saved?.threads['live-1']?.at(-1)?.body.length).toBeLessThanOrEqual(MAX_CHAT_MESSAGE_BODY_CHARS)
  })

  it('clears only the requested scope record', async () => {
    const harnessState = buildState('harness-1', 'Harness conversation')
    const liveState = buildState('live-1', 'Live conversation')

    await saveChatState('harness', harnessState)
    await saveChatState('live', liveState)
    await clearChatState('harness')

    await expect(loadChatState('harness')).resolves.toBeUndefined()
    await expect(loadChatState('live')).resolves.toEqual(liveState)
  })

  it('serializes saves within a scope so older slow writes cannot overwrite newer state', async () => {
    let releaseFirstWrite = () => {}
    putDelays.push(
      new Promise<void>((resolve) => {
        releaseFirstWrite = resolve
      })
    )

    const partialState = buildState('live-1', 'Partial response')
    partialState.threads['live-1'] = [
      { id: 'partial', messageRole: 'assistant', timestamp: 'During stream', body: 'Partial' }
    ]
    const finalState = buildState('live-1', 'Final response')
    finalState.threads['live-1'] = [
      { id: 'final', messageRole: 'assistant', timestamp: 'Done', body: 'Final response' }
    ]

    const partialSave = saveChatState('live', partialState)
    const finalSave = saveChatState('live', finalState)

    await Promise.resolve()
    releaseFirstWrite()
    await Promise.all([partialSave, finalSave])

    await expect(loadChatState('live')).resolves.toMatchObject({
      conversations: [{ id: 'live-1', title: 'Final response' }],
      threads: { 'live-1': [{ id: 'final', body: 'Final response' }] }
    })
  })
})
