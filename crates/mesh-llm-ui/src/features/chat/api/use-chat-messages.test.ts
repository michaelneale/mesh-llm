import type { UIMessage } from '@tanstack/ai-react'

import { uiMessagesToThreadMessages } from '@/features/chat/api/use-chat-messages'

describe('uiMessagesToThreadMessages', () => {
  it('preserves first-class thinking parts for the existing thinking renderer', () => {
    const messages: UIMessage[] = [
      {
        id: 'assistant-1',
        role: 'assistant',
        createdAt: new Date('2026-05-14T12:00:00.000Z'),
        parts: [
          { type: 'thinking', content: 'Checked facts.' },
          { type: 'text', content: ' Final answer.' }
        ]
      }
    ]

    expect(uiMessagesToThreadMessages(messages)).toEqual([
      {
        id: 'assistant-1',
        messageRole: 'assistant',
        timestamp: '2026-05-14T12:00:00.000Z',
        body: '<think>Checked facts.</think> Final answer.'
      }
    ])
  })
})
