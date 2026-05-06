import { useQuery } from '@tanstack/react-query'
import { statusKeys } from '@/lib/query/query-keys'
import { fetchStatus } from './status'

export function useStatusQuery(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: statusKeys.detail(),
    queryFn: fetchStatus,
    staleTime: 30_000,
    refetchInterval: 5_000,
    enabled: options?.enabled ?? true
  })
}
