import { LiveLoadingGhostRoot } from '@/components/ui/LiveLoadingGhostRoot'
import {
  ChatActionsLoadingGhost,
  ChatComposerLoadingGhost,
  ChatMessagesLoadingGhost,
  ChatSidebarLoadingGhost
} from '@/features/chat/components/ChatLoadingGhostSections'
import { ChatLayout } from '@/features/chat/layouts/ChatLayout'

export function ChatLiveLoadingGhost() {
  return (
    <LiveLoadingGhostRoot>
      <ChatLayout
        sidebar={<ChatSidebarLoadingGhost />}
        title="Live chat"
        subtitle="Connecting to the backend model catalog"
        actions={<ChatActionsLoadingGhost />}
        composer={<ChatComposerLoadingGhost />}
      >
        <ChatMessagesLoadingGhost />
      </ChatLayout>
    </LiveLoadingGhostRoot>
  )
}
