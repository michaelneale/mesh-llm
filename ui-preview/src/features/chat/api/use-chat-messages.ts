import type { UIMessage } from '@tanstack/ai-react'
import type { ThreadMessage } from '@/features/app-tabs/types'

export function uiMessagesToThreadMessages(messages: UIMessage[]): ThreadMessage[] {
  return messages
    .filter(
      (m): m is UIMessage & { role: 'user' | 'assistant' } =>
        m.role === 'user' || m.role === 'assistant',
    )
    .map((message) => {
      const textPart = message.parts.find((part) => part.type === 'text')
      const body = textPart?.type === 'text' ? textPart.content : ''
      return {
        id: message.id,
        messageRole: message.role,
        timestamp: message.createdAt ? message.createdAt.toISOString() : new Date().toISOString(),
        body,
      }
    })
}

export function useChatMessages(messages: UIMessage[]): ThreadMessage[] {
  return uiMessagesToThreadMessages(messages)
}
