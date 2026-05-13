import { useEffect, useMemo, useRef } from 'react'
import { useChat } from '@tanstack/ai-react'
import type { UseChatReturn } from '@tanstack/ai-react'
import type { ThreadMessage } from '@/features/app-tabs/types'
import { threadMessagesToUIMessages } from '@/features/chat/api/use-chat-messages'
import { createMeshConnectionAdapter } from '@/features/chat/api/mesh-connection'
import type { ChatResponseMetadata } from '@/features/chat/api/response-metadata'

type UseMeshChatOptions = {
  conversationId: string
  model: string
  systemPrompt?: string
  initialMessages: ThreadMessage[]
  onResponseMetadata?: (metadata: ChatResponseMetadata) => void
}

export function useMeshChat({
  conversationId,
  model,
  systemPrompt = '',
  initialMessages,
  onResponseMetadata
}: UseMeshChatOptions): UseChatReturn {
  const previousConversationIdRef = useRef(conversationId)
  const currentModelRef = useRef(model)
  const currentSystemPromptRef = useRef(systemPrompt)

  currentModelRef.current = model
  currentSystemPromptRef.current = systemPrompt

  const connection = useMemo(
    () =>
      createMeshConnectionAdapter(
        () => currentModelRef.current,
        onResponseMetadata,
        () => currentSystemPromptRef.current
      ),
    [onResponseMetadata]
  )
  const hydratedMessages = useMemo(() => threadMessagesToUIMessages(initialMessages), [initialMessages])
  const chat = useChat({ id: conversationId, connection, initialMessages: hydratedMessages })

  useEffect(() => {
    const conversationChanged = previousConversationIdRef.current !== conversationId
    previousConversationIdRef.current = conversationId

    if (conversationChanged || (chat.messages.length === 0 && hydratedMessages.length > 0)) {
      chat.setMessages(hydratedMessages)
    }
  }, [chat, conversationId, hydratedMessages])

  return chat
}
