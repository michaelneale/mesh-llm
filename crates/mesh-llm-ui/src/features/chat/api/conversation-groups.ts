import type { Conversation, ConversationGroup } from '@/features/app-tabs/types'

const EARLIER_THRESHOLD_MS = 24 * 60 * 60 * 1000
const DEFAULT_CONVERSATION_GROUP_TITLES = ['Today', 'Earlier'] as const

function parseConversationTimestamp(value: string, now: Date): number | undefined {
  const trimmedValue = value.trim()
  const timeOnlyMatch = trimmedValue.match(/^(\d{1,2}):(\d{2})(?:\s*([ap])\.?m\.?)?$/i)

  if (!trimmedValue || /^now$/i.test(trimmedValue)) {
    return now.getTime()
  }

  if (timeOnlyMatch) {
    const hoursValue = Number(timeOnlyMatch[1])
    const minutesValue = Number(timeOnlyMatch[2])
    const period = timeOnlyMatch[3]?.toLowerCase()

    if (hoursValue <= 23 && minutesValue <= 59) {
      const timestamp = new Date(now)
      const hours =
        period === 'p' && hoursValue < 12 ? hoursValue + 12 : period === 'a' && hoursValue === 12 ? 0 : hoursValue
      timestamp.setHours(hours, minutesValue, 0, 0)

      return timestamp.getTime()
    }
  }

  if (/^yesterday$/i.test(trimmedValue)) {
    const yesterday = new Date(now)
    yesterday.setDate(now.getDate() - 1)

    return yesterday.getTime()
  }

  const parsedDate = new Date(trimmedValue)
  if (!Number.isNaN(parsedDate.getTime())) {
    return parsedDate.getTime()
  }

  return undefined
}

function isOlderThanEarlierThreshold(conversation: Conversation, now: Date): boolean {
  const timestamp = parseConversationTimestamp(conversation.updatedAt, now)

  if (timestamp === undefined) return false

  return now.getTime() - timestamp > EARLIER_THRESHOLD_MS
}

export function buildConversationGroups(
  conversations: Conversation[],
  conversationGroups: ConversationGroup[] = [],
  now = new Date()
): ConversationGroup[] {
  const titles = [
    conversationGroups[0]?.title ?? DEFAULT_CONVERSATION_GROUP_TITLES[0],
    conversationGroups[1]?.title ?? DEFAULT_CONVERSATION_GROUP_TITLES[1]
  ]
  const todayConversationIds: string[] = []
  const earlierConversationIds: string[] = []

  for (const conversation of conversations) {
    if (isOlderThanEarlierThreshold(conversation, now)) {
      earlierConversationIds.push(conversation.id)
    } else {
      todayConversationIds.push(conversation.id)
    }
  }

  return [
    { title: titles[0], conversationIds: todayConversationIds },
    { title: titles[1], conversationIds: earlierConversationIds }
  ]
}
