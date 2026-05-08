import { describe, expect, it } from 'vitest'
import {
  mergeThreadMessageMetadata,
  responseMetadataToThreadMessage,
  threadMessageMetadataEquals
} from './response-metadata'
import type { ThreadMessage } from '@/features/app-tabs/types'

describe('responseMetadataToThreadMessage', () => {
  it('formats backend completion metadata for assistant response rows', () => {
    expect(
      responseMetadataToThreadMessage({
        messageId: 'assistant-1',
        model: 'unsloth/MiniMax-M2.5-GGUF:Q4_K_M',
        usage: { input_tokens: 12, output_tokens: 44, total_tokens: 56 },
        timings: { decode_time_ms: 1522, ttft_ms: 1117, total_time_ms: 2639 },
        servedBy: 'carrack'
      })
    ).toEqual({
      model: 'unsloth/MiniMax-M2.5-GGUF:Q4_K_M',
      route: 'carrack',
      routeNode: 'carrack',
      tokens: '44 tok',
      tokPerSec: '28.9 tok/s',
      ttft: '1117ms'
    })
  })
})

describe('mergeThreadMessageMetadata', () => {
  it('prefers backend model metadata while keeping submitted model as fallback', () => {
    const message: ThreadMessage = {
      id: 'assistant-1',
      messageRole: 'assistant',
      timestamp: '2026-05-06T00:00:00.000Z',
      body: 'Hello'
    }

    expect(
      mergeThreadMessageMetadata(
        message,
        { model: 'backend-model', tokens: '27 tok', tokPerSec: '15.3 tok/s', ttft: '1116ms' },
        'submitted-model'
      )
    ).toMatchObject({ model: 'backend-model', tokens: '27 tok', tokPerSec: '15.3 tok/s', ttft: '1116ms' })

    expect(mergeThreadMessageMetadata(message, undefined, 'submitted-model')).toMatchObject({
      model: 'submitted-model'
    })
  })
})

describe('threadMessageMetadataEquals', () => {
  it('detects metadata-only differences for persistence checks', () => {
    const base: ThreadMessage = {
      id: 'assistant-1',
      messageRole: 'assistant',
      timestamp: '2026-05-06T00:00:00.000Z',
      body: 'Hello',
      tokens: '27 tok'
    }

    expect(threadMessageMetadataEquals(base, { ...base })).toBe(true)
    expect(threadMessageMetadataEquals(base, { ...base, tokens: '44 tok' })).toBe(false)
  })
})
