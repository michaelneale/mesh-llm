import { createContext, type Dispatch, type SetStateAction } from 'react'
import type { ChatHarnessData, Conversation, ThreadMessage } from '@/features/app-tabs/types'
import type { useMeshChat } from './use-chat'
import type { useConversations } from './use-conversations'

export type ChatSessionContextValue = {
  activeConversation: Conversation | undefined
  activeConversationKey: string
  activeMessages: ThreadMessage[]
  chat: ReturnType<typeof useMeshChat>
  chatConversationId: string
  conversations: ReturnType<typeof useConversations>
  createConversation: ReturnType<typeof useConversations>['createConversation']
  deleteConversation: ReturnType<typeof useConversations>['deleteConversation']
  draftConversationId: string
  initialThread: ThreadMessage[]
  isStreaming: boolean
  liveMessagesWithModels: ThreadMessage[]
  liveMode: boolean
  messageCounts: Record<string, number>
  renameConversation: ReturnType<typeof useConversations>['renameConversation']
  selectConversation: ReturnType<typeof useConversations>['selectConversation']
  setDraftConversationId: (conversationId: string) => void
  setMessageModels: Dispatch<SetStateAction<Record<string, string>>>
  setSessionModel: (model: string) => void
  setSystemPrompt: (systemPrompt: string) => void
  systemPrompt: string
  streamingConversationIds: readonly string[]
  updateThread: ReturnType<typeof useConversations>['updateThread']
}

export type ChatSessionProviderProps = {
  children: React.ReactNode
  data?: ChatHarnessData
}

export const ChatSessionContext = createContext<ChatSessionContextValue | null>(null)
