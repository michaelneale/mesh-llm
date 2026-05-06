import { QueryClientProvider } from '@tanstack/react-query'
import type { QueryClient } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { defaultQueryClient } from './query-client'

export type QueryProviderProps = {
  children: ReactNode
  client?: QueryClient
}

export function QueryProvider({ children, client = defaultQueryClient }: QueryProviderProps) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}
