import { EventType, type StreamChunk } from '@tanstack/ai'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createMeshConnectionAdapter } from '@/features/chat/api/mesh-connection'
import type { ChatResponseMetadata } from '@/features/chat/api/response-metadata'

function createMessages() {
  return [
    {
      id: 'user-1',
      role: 'user' as const,
      parts: [{ type: 'text' as const, content: 'Hello mesh' }],
      createdAt: new Date('2026-05-06T00:00:00.000Z')
    }
  ]
}

function createAudioUploadMessage() {
  return [
    {
      id: 'user-1',
      role: 'user' as const,
      parts: [
        { type: 'text' as const, content: 'Listen to this clip' },
        {
          type: 'audio' as const,
          source: { type: 'data' as const, value: 'YXVkaW8=', mimeType: 'audio/mpeg' },
          metadata: { fileName: 'task7-clip.mp3' }
        }
      ],
      createdAt: new Date('2026-05-06T00:00:00.000Z')
    }
  ]
}

function createSSEStream(lines: string[], options?: { error?: Error }) {
  const encoder = new TextEncoder()

  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const line of lines) {
        controller.enqueue(encoder.encode(line))
      }

      if (options?.error) {
        queueMicrotask(() => controller.error(options.error as Error))
        return
      }

      controller.close()
    }
  })
}

function createExplodingReaderStream(line: string, message: string) {
  const encoded = new TextEncoder().encode(line)
  const reader = {
    read: vi
      .fn<() => Promise<{ done: boolean; value?: Uint8Array }>>()
      .mockResolvedValueOnce({ done: false, value: encoded })
      .mockRejectedValueOnce(new Error(message)),
    releaseLock: vi.fn()
  }

  return {
    getReader: () => reader
  }
}

function createAbortError() {
  const error = new Error('aborted')
  error.name = 'AbortError'
  return error
}

function createPendingAbortReaderStream(abortSignal: AbortSignal) {
  const reader = {
    read: vi.fn<() => Promise<ReadableStreamReadResult<Uint8Array>>>(() => {
      return new Promise<ReadableStreamReadResult<Uint8Array>>((_, reject) => {
        if (abortSignal.aborted) {
          reject(createAbortError())
          return
        }

        abortSignal.addEventListener('abort', () => reject(createAbortError()), { once: true })
      })
    }),
    cancel: vi.fn<() => Promise<void>>().mockResolvedValue(undefined),
    releaseLock: vi.fn()
  }

  return {
    reader,
    stream: {
      getReader: () => reader
    }
  }
}

function createPendingReaderErrorStream(abortSignal: AbortSignal, error: Error) {
  const reader = {
    read: vi.fn<() => Promise<ReadableStreamReadResult<Uint8Array>>>(() => {
      return new Promise<ReadableStreamReadResult<Uint8Array>>((_, reject) => {
        abortSignal.addEventListener('abort', () => reject(error), { once: true })
      })
    }),
    cancel: vi.fn<() => Promise<void>>().mockResolvedValue(undefined),
    releaseLock: vi.fn()
  }

  return {
    reader,
    stream: {
      getReader: () => reader
    }
  }
}

describe('createMeshConnectionAdapter', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('uses the latest model getter value and emits streamed text chunks', async () => {
    let currentModel = 'model-a'
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          createSSEStream([
            'data: {"type":"response.output_text.delta","delta":"Hello"}\n',
            'data: {"type":"response.output_text.delta","delta":" world"}\n',
            'data: [DONE]\n'
          ]),
          { status: 200 }
        )
      )
    vi.stubGlobal('fetch', fetchMock)

    const adapter = createMeshConnectionAdapter(() => currentModel)
    currentModel = 'model-b'

    const chunks: StreamChunk[] = []
    for await (const chunk of adapter.connect(createMessages(), undefined, undefined)) {
      chunks.push(chunk)
    }

    const request = fetchMock.mock.calls[0]?.[1]
    const body = JSON.parse(String(request?.body)) as { model: string }

    expect(body.model).toBe('model-b')
    expect(chunks.map((chunk) => chunk.type)).toEqual([
      EventType.RUN_STARTED,
      EventType.TEXT_MESSAGE_START,
      EventType.TEXT_MESSAGE_CONTENT,
      EventType.TEXT_MESSAGE_CONTENT,
      EventType.TEXT_MESSAGE_END,
      EventType.RUN_FINISHED
    ])
  })

  it('includes the latest system prompt in responses requests', async () => {
    let currentSystemPrompt = 'Be concise about mesh routing.'
    const fetchMock = vi.fn().mockResolvedValue(new Response(createSSEStream(['data: [DONE]\n']), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const adapter = createMeshConnectionAdapter('model-a', undefined, () => currentSystemPrompt)
    currentSystemPrompt = 'Prefer operational checklists.'

    const chunks: StreamChunk[] = []
    for await (const chunk of adapter.connect(createMessages(), undefined, undefined)) {
      chunks.push(chunk)
    }

    const request = fetchMock.mock.calls[0]?.[1]
    const body = JSON.parse(String(request?.body)) as { input: Array<{ role: string; content: string }> }

    expect(body.input[0]).toEqual({ role: 'system', content: 'Prefer operational checklists.' })
    expect(body.input[1]).toEqual({ role: 'user', content: 'Hello mesh' })
    expect(chunks.map((chunk) => chunk.type)).toEqual([EventType.RUN_STARTED, EventType.RUN_FINISHED])
  })

  it('passes completed response metadata for the generated assistant message', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          createSSEStream([
            'data: {"type":"response.output_text.delta","delta":"Hello"}\n',
            'data: {"type":"response.completed","response":{"id":"resp-1","model":"backend-model","usage":{"input_tokens":8,"output_tokens":27,"total_tokens":35},"timings":{"decode_time_ms":1765,"ttft_ms":1116,"total_time_ms":2881},"served_by":"lemony-28"}}\n',
            'data: [DONE]\n'
          ]),
          { status: 200 }
        )
      )
    vi.stubGlobal('fetch', fetchMock)
    const metadata: ChatResponseMetadata[] = []
    const adapter = createMeshConnectionAdapter('model-a', (nextMetadata) => metadata.push(nextMetadata))

    const chunks: StreamChunk[] = []
    for await (const chunk of adapter.connect(createMessages(), undefined, undefined)) {
      chunks.push(chunk)
    }

    const startChunk = chunks.find((chunk) => chunk.type === EventType.TEXT_MESSAGE_START)
    if (!startChunk || startChunk.type !== EventType.TEXT_MESSAGE_START) {
      throw new Error('missing assistant message start')
    }

    expect(metadata).toEqual([
      {
        messageId: startChunk.messageId,
        model: 'backend-model',
        usage: { input_tokens: 8, output_tokens: 27, total_tokens: 35 },
        timings: { decode_time_ms: 1765, ttft_ms: 1116, total_time_ms: 2881 },
        servedBy: 'lemony-28'
      }
    ])
  })

  it('backfills missing completion timings from the local response stream clock', async () => {
    vi.spyOn(performance, 'now').mockReturnValueOnce(1000).mockReturnValueOnce(1749).mockReturnValueOnce(3277)
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          createSSEStream([
            'data: {"type":"response.output_text.delta","delta":"Measured"}\n',
            'data: {"type":"response.completed","response":{"id":"resp-1","model":"backend-model","usage":{"input_tokens":11,"output_tokens":59,"total_tokens":70},"served_by":"lemony-28"}}\n',
            'data: [DONE]\n'
          ]),
          { status: 200 }
        )
      )
    vi.stubGlobal('fetch', fetchMock)
    const metadata: ChatResponseMetadata[] = []
    const adapter = createMeshConnectionAdapter('model-a', (nextMetadata) => metadata.push(nextMetadata))

    const chunks: StreamChunk[] = []
    for await (const chunk of adapter.connect(createMessages(), undefined, undefined)) {
      chunks.push(chunk)
    }

    expect(chunks.map((chunk) => chunk.type)).toContain(EventType.TEXT_MESSAGE_END)
    expect(metadata).toHaveLength(1)
    expect(metadata[0]?.timings).toEqual({ decode_time_ms: 1528, ttft_ms: 749, total_time_ms: 2277 })
  })

  it('surfaces partial streamed text before a stream failure', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      body: createExplodingReaderStream(
        'data: {"type":"response.output_text.delta","delta":"Partial"}\n',
        'stream exploded'
      )
    })
    vi.stubGlobal('fetch', fetchMock)

    const adapter = createMeshConnectionAdapter('model-a')
    const iterator = adapter.connect(createMessages(), undefined, undefined)[Symbol.asyncIterator]()

    expect((await iterator.next()).value).toMatchObject({ type: EventType.RUN_STARTED })
    expect((await iterator.next()).value).toMatchObject({ type: EventType.TEXT_MESSAGE_START, role: 'assistant' })
    expect((await iterator.next()).value).toMatchObject({ type: EventType.TEXT_MESSAGE_CONTENT, delta: 'Partial' })
    await expect(iterator.next()).rejects.toThrow('stream exploded')
  })

  it('throws ApiError for backend SSE error events before DONE', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          createSSEStream([
            'event: error\n',
            'data: {"error":{"message":"Router failed while streaming"}}\n',
            'data: [DONE]\n'
          ]),
          { status: 200 }
        )
      )
    vi.stubGlobal('fetch', fetchMock)

    const adapter = createMeshConnectionAdapter('model-a')
    const iterator = adapter.connect(createMessages(), undefined, undefined)[Symbol.asyncIterator]()

    expect((await iterator.next()).value).toMatchObject({ type: EventType.RUN_STARTED })
    await expect(iterator.next()).rejects.toMatchObject({
      name: 'ApiError',
      status: 500,
      body: 'Router failed while streaming',
      message: 'Chat stream failed: Router failed while streaming'
    })
  })

  it('cancels the response reader and stops cleanly when aborted during a pending stream read', async () => {
    const abortController = new AbortController()
    const { reader, stream } = createPendingAbortReaderStream(abortController.signal)
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      body: stream
    })
    vi.stubGlobal('fetch', fetchMock)

    const adapter = createMeshConnectionAdapter('model-a')
    const iterator = adapter.connect(createMessages(), undefined, abortController.signal)[Symbol.asyncIterator]()

    expect((await iterator.next()).value).toMatchObject({ type: EventType.RUN_STARTED })

    const pendingRead = iterator.next()
    await vi.waitFor(() => expect(reader.read).toHaveBeenCalledTimes(1))

    abortController.abort()

    await expect(pendingRead).resolves.toEqual({ done: true, value: undefined })
    expect(reader.cancel).toHaveBeenCalledTimes(1)
    expect(reader.releaseLock).toHaveBeenCalledTimes(1)
  })

  it('does not swallow non-abort read errors that race with an abort signal', async () => {
    const abortController = new AbortController()
    const readError = new Error('stream exploded during abort')
    const { reader, stream } = createPendingReaderErrorStream(abortController.signal, readError)
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      body: stream
    })
    vi.stubGlobal('fetch', fetchMock)

    const adapter = createMeshConnectionAdapter('model-a')
    const iterator = adapter.connect(createMessages(), undefined, abortController.signal)[Symbol.asyncIterator]()

    expect((await iterator.next()).value).toMatchObject({ type: EventType.RUN_STARTED })

    const pendingRead = iterator.next()
    await vi.waitFor(() => expect(reader.read).toHaveBeenCalledTimes(1))

    abortController.abort()

    await expect(pendingRead).rejects.toThrow('stream exploded during abort')
    expect(reader.cancel).toHaveBeenCalledTimes(1)
    expect(reader.releaseLock).toHaveBeenCalledTimes(1)
  })

  it('throws ApiError with parsed backend message on non-ok responses', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { message: 'Backend exploded' } }), {
        status: 503,
        statusText: 'Service Unavailable',
        headers: { 'Content-Type': 'application/json' }
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    const adapter = createMeshConnectionAdapter('model-a')
    const iterator = adapter.connect(createMessages(), undefined, undefined)[Symbol.asyncIterator]()

    expect((await iterator.next()).value).toMatchObject({ type: EventType.RUN_STARTED })

    await expect(iterator.next()).rejects.toMatchObject({
      name: 'ApiError',
      status: 503,
      body: 'Backend exploded',
      message: 'Chat request failed: 503'
    })
  })

  it('stops before /api/responses when attachment upload fails', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { message: 'Upload failed: 503' } }), {
        status: 503,
        statusText: 'Service Unavailable',
        headers: { 'Content-Type': 'application/json' }
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    const adapter = createMeshConnectionAdapter('model-a')
    const iterator = adapter.connect(createAudioUploadMessage(), undefined, undefined)[Symbol.asyncIterator]()

    expect((await iterator.next()).value).toMatchObject({ type: EventType.RUN_STARTED })
    await expect(iterator.next()).rejects.toMatchObject({
      name: 'ApiError',
      status: 503,
      body: 'Upload failed: 503',
      message: 'Upload failed: 503'
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/objects')
  })
})
