import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { CHAT_HARNESS } from '@/features/app-tabs/data'
import { loadChatState, saveChatState, trimThreadMessages, type ChatState } from '@/features/chat/api/chat-storage'
import { useConversations } from '@/features/chat/api/use-conversations'

vi.mock('./chat-storage', () => ({
  MAX_CHAT_CONVERSATIONS: 80,
  trimThreadMessages: vi.fn((messages: Array<unknown>) => messages.map((message) => ({ ...(message as object) }))),
  loadChatState: vi.fn(),
  saveChatState: vi.fn()
}))

function cloneHarness() {
  return JSON.parse(JSON.stringify(CHAT_HARNESS)) as typeof CHAT_HARNESS
}

describe('useConversations', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useRealTimers()
    vi.mocked(loadChatState).mockResolvedValue(undefined)
    vi.mocked(saveChatState).mockResolvedValue(undefined)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('restores the persisted active conversation id on hydration', async () => {
    const persistedState: ChatState = {
      conversations: [...cloneHarness().conversations].reverse(),
      conversationGroups: cloneHarness().conversationGroups,
      threads: cloneHarness().threads,
      activeConversationId: 'c2'
    }
    vi.mocked(loadChatState).mockResolvedValue(persistedState)

    const { result } = renderHook(() => useConversations(cloneHarness(), 'harness'))

    await waitFor(() => {
      expect(result.current.activeConversationId).toBe('c2')
      expect(result.current.conversations[0]?.id).toBe('c2')
    })
  })

  it('restores the persisted live-scoped selection with its thread on hydration', async () => {
    const persistedState: ChatState = {
      conversations: [
        {
          id: 'live-a',
          title: 'Live first',
          subtitle: '',
          createdAt: Date.now(),
          updatedAt: String(Date.now()),
          messages: []
        },
        {
          id: 'live-b',
          title: 'Live selected',
          subtitle: '',
          createdAt: Date.now(),
          updatedAt: String(Date.now()),
          messages: []
        }
      ],
      conversationGroups: [
        { title: 'Today', conversationIds: ['live-a', 'live-b'] },
        { title: 'Earlier', conversationIds: [] }
      ],
      threads: {
        'live-a': [{ id: 'msg-a', messageRole: 'assistant', timestamp: 'Now', body: 'Wrong thread' }],
        'live-b': [{ id: 'msg-b', messageRole: 'assistant', timestamp: 'Later', body: 'Live thread survives reload' }]
      },
      activeConversationId: 'live-b'
    }
    vi.mocked(loadChatState).mockResolvedValue(persistedState)

    const { result } = renderHook(() => useConversations(cloneHarness(), 'live'))

    await waitFor(() => {
      expect(result.current.activeConversationId).toBe('live-b')
      expect(result.current.threads['live-b']?.[0]?.body).toBe('Live thread survives reload')
      expect(result.current.threads['live-a']?.[0]?.body).toBe('Wrong thread')
    })
  })

  it('normalizes restored conversation groups with a rolling 24-hour Earlier cutoff', async () => {
    const now = Date.now()
    const stillToday = new Date(now - 23 * 60 * 60 * 1000).toISOString()
    const earlier = new Date(now - 24 * 60 * 60 * 1000 - 1).toISOString()
    vi.mocked(loadChatState).mockResolvedValue({
      conversations: [
        { id: 'now', title: 'Now', subtitle: '', createdAt: now, updatedAt: new Date(now).toISOString(), messages: [] },
        { id: 'still-today', title: 'Still today', subtitle: '', createdAt: now, updatedAt: stillToday, messages: [] },
        { id: 'earlier', title: 'Earlier', subtitle: '', createdAt: now, updatedAt: earlier, messages: [] }
      ],
      conversationGroups: [
        { title: 'Today', conversationIds: ['now'] },
        { title: 'Earlier', conversationIds: ['still-today', 'earlier'] }
      ],
      threads: { now: [], 'still-today': [], earlier: [] },
      activeConversationId: 'now'
    })

    const { result } = renderHook(() => useConversations(cloneHarness(), 'live'))

    await waitFor(() => {
      expect(result.current.conversationGroups).toEqual([
        { title: 'Today', conversationIds: ['now', 'still-today'] },
        { title: 'Earlier', conversationIds: ['earlier'] }
      ])
    })
  })

  it('falls back safely from malformed persisted state without mutating fixture data', async () => {
    const fallback = cloneHarness()
    const snapshot = cloneHarness()
    vi.mocked(loadChatState).mockResolvedValue({} as ChatState)

    const { result } = renderHook(() => useConversations(fallback, 'harness'))

    await waitFor(() => {
      expect(result.current.conversations[0]?.title).toBe(snapshot.conversations[0]?.title)
    })

    expect(fallback).toEqual(snapshot)
  })

  it('derives titles from the first user line and moves updated conversations to the front', async () => {
    vi.mocked(loadChatState).mockResolvedValue({
      conversations: cloneHarness().conversations,
      conversationGroups: cloneHarness().conversationGroups,
      threads: cloneHarness().threads,
      activeConversationId: 'c2'
    })

    const { result } = renderHook(() => useConversations(cloneHarness(), 'harness'))

    await waitFor(() => {
      expect(result.current.activeConversationId).toBe('c2')
    })

    act(() => {
      result.current.updateThread('c2', [
        {
          id: 'message-1',
          messageRole: 'user',
          timestamp: 'Now',
          body: '\n\nFirst useful line that is intentionally longer than sixty characters for truncation\nsecond line ignored'
        }
      ])
    })

    await waitFor(() => {
      expect(result.current.conversations[0]?.id).toBe('c2')
      expect(result.current.conversations[0]?.title).toBe(
        'First useful line that is intentionally longer than sixty ch'
      )
      expect(result.current.threads.c2?.[0]?.body).toContain('second line ignored')
      expect(saveChatState).toHaveBeenCalledWith('harness', expect.objectContaining({ activeConversationId: 'c2' }))
      expect(trimThreadMessages).toHaveBeenCalled()
    })
  })

  it('renames conversations with normalized titles and persists the reordered list', async () => {
    const fallback = cloneHarness()
    const { result } = renderHook(() => useConversations(fallback, 'harness'))

    await waitFor(() => {
      expect(result.current.conversations[0]?.id).toBe('c1')
      expect(loadChatState).toHaveBeenCalledWith('harness')
    })
    await act(async () => {
      await Promise.resolve()
    })

    act(() => {
      result.current.renameConversation('c2', `  ${'Renamed capacity note '.repeat(10)}  `)
    })

    await waitFor(() => {
      const renamedConversation = result.current.conversations[0]
      expect(renamedConversation?.id).toBe('c2')
      expect(renamedConversation?.title.length).toBeLessThanOrEqual(140)
      expect(renamedConversation?.title).not.toMatch(/^\s|\s$/)
    })

    const renamedState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
    expect(renamedState?.conversations[0]?.id).toBe('c2')
    expect(renamedState?.conversations[0]?.title.length).toBeLessThanOrEqual(140)

    act(() => {
      result.current.renameConversation('c2', '   ')
    })

    await waitFor(() => {
      expect(result.current.conversations[0]).toMatchObject({ id: 'c2', title: 'New chat' })
    })
  })

  it('deletes active conversations, removes their threads, and activates the next row', async () => {
    const fallback = cloneHarness()
    const { result } = renderHook(() => useConversations(fallback, 'harness'))

    await waitFor(() => {
      expect(result.current.activeConversationId).toBe('c1')
      expect(loadChatState).toHaveBeenCalledWith('harness')
    })
    await act(async () => {
      await Promise.resolve()
    })

    act(() => {
      result.current.deleteConversation('c1')
    })

    await waitFor(() => {
      expect(result.current.conversations.some((conversation) => conversation.id === 'c1')).toBe(false)
      expect(result.current.threads.c1).toBeUndefined()
      expect(result.current.activeConversationId).toBe('c2')
    })

    const deletedState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
    expect(deletedState?.activeConversationId).toBe('c2')
    expect(deletedState?.conversations.some((conversation) => conversation.id === 'c1')).toBe(false)
    expect(deletedState?.threads.c1).toBeUndefined()
  })
})
