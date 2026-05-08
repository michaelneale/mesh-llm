import { useMemo } from 'react'
import type { UIMessage } from '@tanstack/ai-react'
import type { ThreadMessage } from '@/features/app-tabs/types'

function parseTimestamp(value: string): Date | undefined {
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? undefined : parsed
}

function formatTimestamp(value: Date | undefined): string {
  return value && !Number.isNaN(value.getTime()) ? value.toISOString() : new Date().toISOString()
}

export function threadMessagesToUIMessages(messages: ThreadMessage[]): UIMessage[] {
  return messages.map((message) => ({
    id: message.id,
    role: message.messageRole,
    createdAt: parseTimestamp(message.timestamp),
    parts: [{ type: 'text', content: message.body }]
  }))
}

export function uiMessagesToThreadMessages(messages: UIMessage[]): ThreadMessage[] {
  return messages
    .filter((m): m is UIMessage & { role: 'user' | 'assistant' } => m.role === 'user' || m.role === 'assistant')
    .map((message) => {
      const textPart = message.parts.find((part) => part.type === 'text')
      const body = textPart?.type === 'text' ? textPart.content : ''
      return {
        id: message.id,
        messageRole: message.role,
        timestamp: formatTimestamp(message.createdAt),
        body
      }
    })
}

export function useChatMessages(messages: UIMessage[]): ThreadMessage[] {
  return useMemo(() => uiMessagesToThreadMessages(messages), [messages])
}
