import type { UseChatReturn } from '@tanstack/ai-react'
import { uploadAttachment } from '@/features/chat/api/upload-attachment'

export function useSendMessage(chat: UseChatReturn) {
  async function sendText(text: string): Promise<void> {
    await chat.sendMessage(text)
  }

  async function sendWithAttachment(text: string, file: File): Promise<void> {
    const token = await uploadAttachment(file)
    await chat.sendMessage({
      content: [
        { type: 'text', content: text },
        { type: 'image', source: { type: 'url', value: `mesh://blob/${token}` } }
      ]
    })
  }

  return {
    sendText,
    sendWithAttachment,
    isLoading: chat.isLoading,
    error: chat.error
  }
}
