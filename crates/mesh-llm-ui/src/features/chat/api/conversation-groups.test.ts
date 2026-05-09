import { describe, expect, it } from 'vitest'
import type { Conversation } from '@/features/app-tabs/types'
import { buildConversationGroups } from '@/features/chat/api/conversation-groups'

function conversation(id: string, updatedAt: string): Conversation {
  return { id, title: id, subtitle: '', updatedAt }
}

describe('buildConversationGroups', () => {
  it('keeps conversations out of Earlier until they are more than 24 hours old', () => {
    const now = new Date('2026-05-06T14:00:00.000Z')
    const conversations = [
      conversation('current', now.toISOString()),
      conversation('almost-a-day-old', '2026-05-05T14:00:00.001Z'),
      conversation('exactly-a-day-old', '2026-05-05T14:00:00.000Z'),
      conversation('older-than-a-day', '2026-05-05T13:59:59.999Z')
    ]

    expect(buildConversationGroups(conversations, [], now)).toEqual([
      { title: 'Today', conversationIds: ['current', 'almost-a-day-old', 'exactly-a-day-old'] },
      { title: 'Earlier', conversationIds: ['older-than-a-day'] }
    ])
  })

  it('preserves custom section titles while rebuilding membership from timestamps', () => {
    const now = new Date('2026-05-06T14:00:00.000Z')

    expect(
      buildConversationGroups(
        [conversation('recent', 'Now'), conversation('old', '2026-05-04T14:00:00.000Z')],
        [
          { title: 'Recent', conversationIds: ['old'] },
          { title: 'Archive', conversationIds: ['recent'] }
        ],
        now
      )
    ).toEqual([
      { title: 'Recent', conversationIds: ['recent'] },
      { title: 'Archive', conversationIds: ['old'] }
    ])
  })
})
