import type { ThreadMessage } from '@/features/app-tabs/types'
import type { ChatTimings, ChatUsage } from '@/lib/api/types'

export type ChatResponseMetadata = {
  messageId: string
  model?: string
  usage?: ChatUsage
  timings?: ChatTimings
  servedBy?: string
}

export type ThreadMessageMetadata = Pick<
  ThreadMessage,
  'model' | 'route' | 'routeNode' | 'tokens' | 'tokPerSec' | 'ttft'
>

const metadataKeys = ['model', 'route', 'routeNode', 'tokens', 'tokPerSec', 'ttft'] satisfies Array<
  keyof ThreadMessageMetadata
>

function formatTokenCount(value: number | undefined): string | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined

  return `${Math.max(0, Math.round(value))} tok`
}

function formatMilliseconds(value: number | undefined): string | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined

  return `${Math.max(0, Math.round(value))}ms`
}

function formatTokPerSec(tokens: number | undefined, decodeTimeMs: number | undefined): string | undefined {
  if (
    typeof tokens !== 'number' ||
    !Number.isFinite(tokens) ||
    typeof decodeTimeMs !== 'number' ||
    !Number.isFinite(decodeTimeMs) ||
    decodeTimeMs <= 0
  ) {
    return undefined
  }

  return `${(tokens / (decodeTimeMs / 1000)).toFixed(1)} tok/s`
}

export function responseMetadataToThreadMessage(metadata: ChatResponseMetadata): ThreadMessageMetadata {
  const outputTokens = metadata.usage?.output_tokens
  const servedBy = metadata.servedBy

  return {
    model: metadata.model,
    route: servedBy,
    routeNode: servedBy,
    tokens: formatTokenCount(outputTokens),
    tokPerSec: formatTokPerSec(outputTokens, metadata.timings?.decode_time_ms),
    ttft: formatMilliseconds(metadata.timings?.ttft_ms)
  }
}

export function mergeThreadMessageMetadata(
  message: ThreadMessage,
  metadata: ThreadMessageMetadata | undefined,
  fallbackModel?: string
): ThreadMessage {
  const model = metadata?.model ?? message.model ?? fallbackModel
  const merged: ThreadMessage = { ...message, ...(metadata ?? {}) }

  return model ? { ...merged, model } : merged
}

export function extractThreadMessageMetadata(message: ThreadMessage): ThreadMessageMetadata | undefined {
  const metadata: ThreadMessageMetadata = {}

  for (const key of metadataKeys) {
    const value = message[key]
    if (value) metadata[key] = value
  }

  return metadataKeys.some((key) => metadata[key] != null) ? metadata : undefined
}

export function threadMessageMetadataEquals(left: ThreadMessage, right: ThreadMessage): boolean {
  return metadataKeys.every((key) => left[key] === right[key])
}
