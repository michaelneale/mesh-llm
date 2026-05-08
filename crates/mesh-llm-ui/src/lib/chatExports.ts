import { attachmentForMessage } from '@/features/chat/lib/message-content'
import { describeImageAttachmentForPrompt, describeRenderedPagesAsText } from '@/features/chat/lib/vision-describe'
import { ChatPage as LegacyChatPage } from '@/features/chat/components/ChatPage'

export {
  attachmentForMessage,
  describeImageAttachmentForPrompt,
  describeRenderedPagesAsText,
  LegacyChatPage as ChatPage
}
