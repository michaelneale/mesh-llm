import type { ConnectConnectionAdapter } from '@tanstack/ai-client'
import type { UIMessage, ModelMessage, StreamChunk } from '@tanstack/ai'
import { EventType } from '@tanstack/ai'

import { env } from '@/lib/env'
import { ApiError, parseApiErrorBody } from '@/lib/api/errors'
import { getClientId } from '@/lib/api/client-id'
import { generateRequestId } from '@/lib/api/request-id'
import type { ChatSSEEvent } from '@/lib/api/types'
import { buildResponsesInput } from '@/features/chat/api/build-input'
import type { ChatResponseMetadata } from '@/features/chat/api/response-metadata'

function nowMs() {
  return performance.now()
}

function resolveModel(model: string | (() => string)): string {
  return typeof model === 'function' ? model() : model
}

function resolveSystemPrompt(systemPrompt: string | (() => string) | undefined): string {
  if (!systemPrompt) return ''
  return typeof systemPrompt === 'function' ? systemPrompt() : systemPrompt
}

function parseChatSSEEvent(data: string) {
  try {
    return JSON.parse(data) as ChatSSEEvent
  } catch {
    return undefined
  }
}

function messageFromStreamError(data: string): string {
  try {
    const json = JSON.parse(data) as unknown
    if (json && typeof json === 'object') {
      const obj = json as Record<string, unknown>
      if (obj['error'] && typeof obj['error'] === 'object') {
        const error = obj['error'] as Record<string, unknown>
        if (typeof error['message'] === 'string') return error['message']
      }
      if (typeof obj['error'] === 'string') return obj['error']
      if (typeof obj['message'] === 'string') return obj['message']
    }
  } catch {
    // Not JSON; surface the raw stream payload below.
  }

  return data || 'Chat stream failed'
}

function statusFromStreamError(data: string): number {
  try {
    const json = JSON.parse(data) as unknown
    if (json && typeof json === 'object') {
      const obj = json as Record<string, unknown>
      if (typeof obj['status'] === 'number') return obj['status']
      if (typeof obj['status_code'] === 'number') return obj['status_code']
      if (obj['error'] && typeof obj['error'] === 'object') {
        const error = obj['error'] as Record<string, unknown>
        if (typeof error['status'] === 'number') return error['status']
        if (typeof error['status_code'] === 'number') return error['status_code']
      }
    }
  } catch {
    // Not JSON; stream errors do not always carry HTTP status metadata.
  }

  return 500
}

function isAbortError(error: unknown) {
  return error instanceof Error && error.name === 'AbortError'
}

async function* parseSSEStream(
  body: ReadableStream<Uint8Array>,
  abortSignal?: AbortSignal
): AsyncGenerator<ChatSSEEvent> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let streamFinished = false
  let eventName = 'message'

  function parseLine(line: string): ChatSSEEvent | undefined {
    const trimmed = line.trim()
    if (trimmed === '') {
      eventName = 'message'
      return undefined
    }
    if (trimmed.startsWith('event:')) {
      eventName = trimmed.slice(6).trim() || 'message'
      return undefined
    }
    if (!trimmed.startsWith('data:')) return undefined

    const data = trimmed.slice(5).trimStart()
    if (data === '[DONE]') {
      streamFinished = true
      return undefined
    }
    if (eventName === 'error') {
      const message = messageFromStreamError(data)
      throw new ApiError(statusFromStreamError(data), message, `Chat stream failed: ${message}`)
    }

    return parseChatSSEEvent(data)
  }

  try {
    while (true) {
      if (abortSignal?.aborted) return

      let readResult: ReadableStreamReadResult<Uint8Array>
      try {
        readResult = await reader.read()
      } catch (error) {
        if (isAbortError(error)) return
        throw error
      }

      if (abortSignal?.aborted) return

      const { done, value } = readResult
      if (done) {
        streamFinished = true
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        const event = parseLine(line)
        if (streamFinished) {
          return
        }
        if (event) yield event
      }
    }

    if (buffer.trim()) {
      const event = parseLine(buffer)
      if (event) yield event
    }
  } finally {
    if (!streamFinished) {
      try {
        await reader.cancel()
      } catch (_) {
        void _
      }
    }
    reader.releaseLock()
  }
}

async function* runConnect(
  model: string,
  messages: Array<UIMessage> | Array<ModelMessage>,
  abortSignal?: AbortSignal,
  onResponseMetadata?: (metadata: ChatResponseMetadata) => void,
  systemPrompt = ''
): AsyncGenerator<StreamChunk> {
  const clientId = getClientId()
  const requestId = generateRequestId()
  const messageId = generateRequestId()
  const requestStartedAt = nowMs()

  yield { type: EventType.RUN_STARTED, threadId: requestId, runId: requestId }

  let response: Response
  try {
    const requestBody = await buildResponsesInput(messages, model, clientId, requestId, systemPrompt)
    response = await fetch(`${env.managementApiUrl}/api/responses`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
      signal: abortSignal
    })
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') return
    throw err
  }

  if (!response.ok) {
    const errorBody = await parseApiErrorBody(response)
    throw new ApiError(response.status, errorBody, `Chat request failed: ${response.status}`)
  }

  if (!response.body) {
    throw new Error('Response body is null')
  }

  let messageStarted = false
  let firstDeltaAt: number | undefined

  for await (const event of parseSSEStream(response.body, abortSignal)) {
    if (abortSignal?.aborted) break

    if (event.type === 'response.output_text.delta') {
      firstDeltaAt ??= nowMs()
      if (!messageStarted) {
        messageStarted = true
        yield { type: EventType.TEXT_MESSAGE_START, messageId, role: 'assistant' }
      }
      yield { type: EventType.TEXT_MESSAGE_CONTENT, messageId, delta: event.delta }
      continue
    }

    if (event.type === 'response.completed') {
      const completedAt = nowMs()
      const fallbackTimings = {
        decode_time_ms: firstDeltaAt == null ? undefined : Math.max(0, completedAt - firstDeltaAt),
        ttft_ms: firstDeltaAt == null ? undefined : Math.max(0, firstDeltaAt - requestStartedAt),
        total_time_ms: Math.max(0, completedAt - requestStartedAt)
      }

      onResponseMetadata?.({
        messageId,
        model: event.response.model,
        usage: event.response.usage,
        timings: {
          decode_time_ms: event.response.timings?.decode_time_ms ?? fallbackTimings.decode_time_ms,
          ttft_ms: event.response.timings?.ttft_ms ?? fallbackTimings.ttft_ms,
          total_time_ms: event.response.timings?.total_time_ms ?? fallbackTimings.total_time_ms
        },
        servedBy: event.response.served_by
      })
    }
  }

  if (abortSignal?.aborted) return

  if (messageStarted) {
    yield { type: EventType.TEXT_MESSAGE_END, messageId }
  }

  yield { type: EventType.RUN_FINISHED, threadId: requestId, runId: requestId }
}

export function createMeshConnectionAdapter(
  model: string | (() => string),
  onResponseMetadata?: (metadata: ChatResponseMetadata) => void,
  systemPrompt?: string | (() => string)
): ConnectConnectionAdapter {
  return {
    connect: (_messages, _data, abortSignal) =>
      runConnect(resolveModel(model), _messages, abortSignal, onResponseMetadata, resolveSystemPrompt(systemPrompt))
  }
}
