import { useContext } from 'react'
import { ChatSessionContext } from './chat-session-context'

export function useOptionalChatSession() {
  return useContext(ChatSessionContext)
}

export function useChatSession() {
  const session = useOptionalChatSession()
  if (!session) {
    throw new Error('useChatSession must be used within ChatSessionProvider')
  }
  return session
}
