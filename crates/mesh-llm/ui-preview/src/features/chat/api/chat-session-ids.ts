function createDraftConversationId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }

  return `conversation-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function createChatDraftConversationId(): string {
  return createDraftConversationId()
}
