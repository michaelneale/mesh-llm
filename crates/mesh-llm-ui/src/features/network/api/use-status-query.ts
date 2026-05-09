import { useQuery } from '@tanstack/react-query'
import { statusKeys } from '@/lib/query/query-keys'
import { fetchStatus } from '@/features/network/api/status'

function defaultStatusRefetchInterval(): number | false {
  return typeof EventSource === 'undefined' ? 5_000 : false
}

export function useStatusQuery(options?: { enabled?: boolean; refetchInterval?: number | false }) {
  return useQuery({
    queryKey: statusKeys.detail(),
    queryFn: fetchStatus,
    staleTime: 30_000,
    refetchInterval: options?.refetchInterval ?? defaultStatusRefetchInterval(),
    enabled: options?.enabled ?? true
  })
}
