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
  for await (const chunk of adapter.connect([message], undefined, undefined)) {
    void chunk
    // Drain the stream so the request body is built and posted.
  }
}

vi.mock('@tanstack/ai-react', async () => {
  const React = await import('react')

  return {
    useChat: vi.fn(({ connection, initialMessages }: UseChatOptions) => {
      const connectionRef = React.useRef(connection)

      return {
        messages: initialMessages,
        sendMessage: vi.fn((content: string) => drainStream(connectionRef.current, createUserMessage(content))),
        setMessages: vi.fn(),
        reload: vi.fn(),
        stop: vi.fn(),
        status: 'ready',
        error: null,
        isLoading: false
      }
    })
  }
})

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

function SendMessageOnLayout({
  message,
  model,
  systemPrompt
}: {
  message?: string
  model: string
  systemPrompt: string
}) {
  const chat = useMeshChat({
    conversationId: 'chat-1',
    model,
    systemPrompt,
    initialMessages: []
  })

  useLayoutEffect(() => {
    if (message) {
      void chat.sendMessage(message)
    }
  }, [chat, message])

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

  it('sends the latest model and system prompt after rerendering the chat hook', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(createSSEStream(['data: [DONE]\n']), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const { rerender } = render(<SendMessageOnLayout model="model-a" systemPrompt="prompt-a" />)

    rerender(<SendMessageOnLayout message="Use latest values" model="model-b" systemPrompt="prompt-b" />)

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    const request = fetchMock.mock.calls[0]?.[1]
    const body = JSON.parse(String(request?.body)) as {
      model: string
      input: Array<{ role: string; content: string }>
    }

    expect(body.model).toBe('model-b')
    expect(body.input[0]).toEqual({ role: 'system', content: 'prompt-b' })
    expect(body.input[1]).toEqual({ role: 'user', content: 'Use latest values' })
  })
})
