import type { ConnectConnectionAdapter } from '@tanstack/ai-client'
import type { UIMessage, ModelMessage, StreamChunk } from '@tanstack/ai'
import { EventType } from '@tanstack/ai'

import { env } from '../../../lib/env'
import { ApiError, parseApiErrorBody } from '../../../lib/api/errors'
import { getClientId } from '../../../lib/api/client-id'
import { generateRequestId } from '../../../lib/api/request-id'
import type { ChatSSEEvent } from '../../../lib/api/types'
import { buildResponsesInput } from './build-input'

function parseChatSSEEvent(data: string) {
  try {
    return JSON.parse(data) as ChatSSEEvent
  } catch {
    return undefined
  }
}

async function* parseSSEStream(body: ReadableStream<Uint8Array>): AsyncGenerator<ChatSSEEvent> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data: ')) continue
        const data = trimmed.slice(6)
        if (data === '[DONE]') return
        const event = parseChatSSEEvent(data)
        if (event) yield event
      }
    }

    if (buffer.trim().startsWith('data: ')) {
      const data = buffer.trim().slice(6)
      if (data !== '[DONE]') {
        const event = parseChatSSEEvent(data)
        if (event) yield event
      }
    }
  } finally {
    reader.releaseLock()
  }
}

async function* runConnect(
  model: string,
  messages: Array<UIMessage> | Array<ModelMessage>,
  abortSignal?: AbortSignal,
): AsyncGenerator<StreamChunk> {
  const clientId = getClientId()
  const requestId = generateRequestId()
  const messageId = generateRequestId()

  yield { type: EventType.RUN_STARTED, threadId: requestId, runId: requestId }

  let response: Response
  try {
    response = await fetch(`${env.apiUrl}/api/responses`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildResponsesInput(messages, model, clientId, requestId)),
      signal: abortSignal,
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

  for await (const event of parseSSEStream(response.body)) {
    if (abortSignal?.aborted) break

    if (event.type === 'response.output_text.delta') {
      if (!messageStarted) {
        messageStarted = true
        yield { type: EventType.TEXT_MESSAGE_START, messageId, role: 'assistant' }
      }
      yield { type: EventType.TEXT_MESSAGE_CONTENT, messageId, delta: event.delta }
    }
  }

  if (messageStarted) {
    yield { type: EventType.TEXT_MESSAGE_END, messageId }
  }

  yield { type: EventType.RUN_FINISHED, threadId: requestId, runId: requestId }
}

export function createMeshConnectionAdapter(model: string): ConnectConnectionAdapter {
  return {
    connect: (_messages, _data, abortSignal) =>
      runConnect(model, _messages, abortSignal),
  }
}
