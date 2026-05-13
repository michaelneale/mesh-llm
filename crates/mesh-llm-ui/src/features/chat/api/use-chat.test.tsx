import { useLayoutEffect } from 'react'
import { render, waitFor } from '@testing-library/react'
import type { UIMessage } from '@tanstack/ai'
import type { ConnectConnectionAdapter } from '@tanstack/ai-client'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { DEFAULT_SYSTEM_PROMPT } from '@/constants/system-prompt'
import { useMeshChat } from '@/features/chat/api/use-chat'

type UseChatOptions = {
  id: string
  connection: ConnectConnectionAdapter
  initialMessages: UIMessage[]
}

function createUserMessage(content: string): UIMessage {
  return {
    id: 'user-1',
    role: 'user',
    parts: [{ type: 'text', content }],
    createdAt: new Date('2026-05-13T00:00:00.000Z')
  }
}

async function drainStream(adapter: ConnectConnectionAdapter, message: UIMessage) {
  for await (const _chunk of adapter.connect([message], undefined, undefined)) {
    // Drain the stream so the request body is built and posted.
  }
}

vi.mock('@tanstack/ai-react', () => ({
  useChat: vi.fn(({ connection, initialMessages }: UseChatOptions) => ({
    messages: initialMessages,
    sendMessage: vi.fn((content: string) => drainStream(connection, createUserMessage(content))),
    setMessages: vi.fn(),
    reload: vi.fn(),
    stop: vi.fn(),
    status: 'ready',
    error: null,
    isLoading: false
  }))
}))

function createSSEStream(lines: string[]) {
  const encoder = new TextEncoder()

  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const line of lines) {
        controller.enqueue(encoder.encode(line))
      }

      controller.close()
    }
  })
}

function SendFirstMessageOnLayout() {
  const chat = useMeshChat({
    conversationId: 'chat-1',
    model: 'auto',
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
    initialMessages: []
  })

  useLayoutEffect(() => {
    void chat.sendMessage('What is MeshLLM?')
  }, [chat])

  return null
}

describe('useMeshChat', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('sends the default system prompt with the first message in a new chat', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(createSSEStream(['data: [DONE]\n']), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    render(<SendFirstMessageOnLayout />)

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    const request = fetchMock.mock.calls[0]?.[1]
    const body = JSON.parse(String(request?.body)) as { input: Array<{ role: string; content: string }> }

    expect(body.input[0]).toEqual({ role: 'system', content: DEFAULT_SYSTEM_PROMPT })
    expect(body.input[1]).toEqual({ role: 'user', content: 'What is MeshLLM?' })
  })
})
