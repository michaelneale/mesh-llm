import { useQuery } from '@tanstack/react-query'
import { modelKeys } from '@/lib/query/query-keys'
import { fetchModels } from '@/features/network/api/models'

export function useModelsQuery(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: modelKeys.catalog(),
    queryFn: fetchModels,
    staleTime: 60_000,
    refetchInterval: 60_000,
    enabled: options?.enabled ?? true
  })
}
