import { useMemo } from 'react'
import { useChat } from '@tanstack/ai-react'
import type { UseChatReturn } from '@tanstack/ai-react'
import { createMeshConnectionAdapter } from './mesh-connection'

export function useMeshChat(model: string): UseChatReturn {
  const connection = useMemo(() => createMeshConnectionAdapter(model), [model])
  return useChat({ connection })
}
