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

const mockStore = new Map<string, ChatState>()
const putDelays: Promise<void>[] = []

vi.mock('idb', () => ({
  openDB: vi.fn(async () => ({
    get: vi.fn(async (_storeName: string, key: string) => mockStore.get(key)),
    put: vi.fn(async (_storeName: string, value: ChatState, key: string) => {
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

describe('chat-storage', () => {
  beforeEach(() => {
    mockStore.clear()
    putDelays.length = 0
  })

  it('stores live and harness chat state in separate scopes', async () => {
    const harnessState = buildState('harness-1', 'Harness conversation')
    const liveState = buildState('live-1', 'Live conversation')

    await saveChatState('harness', harnessState)
    await saveChatState('live', liveState)

    await expect(loadChatState('harness')).resolves.toEqual(harnessState)
    await expect(loadChatState('live')).resolves.toEqual(liveState)
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
