export const statusKeys = {
  all: ['status'] as const,
  detail: () => [...statusKeys.all, 'detail'] as const
}

export const modelKeys = {
  all: ['models'] as const,
  catalog: () => [...modelKeys.all, 'catalog'] as const
}

export const chatKeys = {
  all: ['chat'] as const,
  conversations: () => [...chatKeys.all, 'conversations'] as const,
  messages: (id: string) => [...chatKeys.all, 'messages', id] as const
}
