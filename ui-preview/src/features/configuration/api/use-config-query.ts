import { useMemo } from 'react'
import type { UseQueryResult } from '@tanstack/react-query'
import type { ModelsResponse, StatusPayload } from '@/lib/api/types'
import type { ConfigurationHarnessData } from '@/features/app-tabs/types'
import { useStatusQuery } from '@/features/network/api/use-status-query'
import { useModelsQuery } from '@/features/network/api/use-models-query'
import { adaptStatusToConfiguration } from './config-adapter'

type ConfigQueryResult = {
  data: ConfigurationHarnessData | undefined
  isError: boolean
  isFetching: boolean
  isPending: boolean
  statusQuery: UseQueryResult<StatusPayload, Error>
  modelsQuery: UseQueryResult<ModelsResponse, Error>
}

export function useConfigQuery(options?: { enabled?: boolean }): ConfigQueryResult {
  const statusQuery = useStatusQuery(options)
  const modelsQuery = useModelsQuery(options)

  const data = useMemo(
    () => statusQuery.data && modelsQuery.data
      ? adaptStatusToConfiguration(statusQuery.data, modelsQuery.data.mesh_models ?? [])
      : undefined,
    [modelsQuery.data, statusQuery.data],
  )

  return {
    data,
    isPending: statusQuery.isPending || modelsQuery.isPending,
    isFetching: statusQuery.isFetching || modelsQuery.isFetching,
    isError: statusQuery.isError || modelsQuery.isError,
    statusQuery,
    modelsQuery,
  }
}
