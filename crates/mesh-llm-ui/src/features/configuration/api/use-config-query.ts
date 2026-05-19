import { useCallback, useMemo } from 'react'
import { useQuery, useQueryClient, type UseQueryResult } from '@tanstack/react-query'
import type { ModelsResponse, StatusPayload } from '@/lib/api/types'
import type { ConfigurationDefaultsValues, ConfigurationHarnessData } from '@/features/app-tabs/types'
import { useStatusQuery } from '@/features/network/api/use-status-query'
import { useModelsQuery } from '@/features/network/api/use-models-query'
import {
  adaptStatusToConfiguration,
  applyRuntimeControlConfig,
  createConfigurationDefaultsValuesFromMeshConfig,
  fetchRuntimeControlConfig,
  type RuntimeControlApplyResponse,
  type RuntimeControlConfigResult
} from '@/features/configuration/api/config-adapter'

const CONFIG_CONTROL_QUERY_KEY = ['configuration', 'runtime-control'] as const

type ConfigQueryResult = {
  data: ConfigurationHarnessData | undefined
  isError: boolean
  isFetching: boolean
  isPending: boolean
  statusQuery: UseQueryResult<StatusPayload, Error>
  modelsQuery: UseQueryResult<ModelsResponse, Error>
  controlConfigQuery: UseQueryResult<RuntimeControlConfigResult, Error>
  applyDefaults: (defaultsValues: ConfigurationDefaultsValues) => Promise<RuntimeControlApplyResponse | null>
}

export function useConfigQuery(options?: { enabled?: boolean }): ConfigQueryResult {
  const queryClient = useQueryClient()
  const statusQuery = useStatusQuery(options)
  const modelsQuery = useModelsQuery(options)
  const controlConfigQuery = useQuery({
    queryKey: CONFIG_CONTROL_QUERY_KEY,
    queryFn: fetchRuntimeControlConfig,
    staleTime: 30_000,
    retry: false,
    enabled: options?.enabled ?? true
  })

  const defaultsValues = useMemo(
    () =>
      controlConfigQuery.data?.snapshot
        ? createConfigurationDefaultsValuesFromMeshConfig(controlConfigQuery.data.snapshot.config)
        : undefined,
    [controlConfigQuery.data]
  )

  const data = useMemo(
    () =>
      statusQuery.data && modelsQuery.data
        ? adaptStatusToConfiguration(statusQuery.data, modelsQuery.data.mesh_models ?? [], defaultsValues)
        : undefined,
    [defaultsValues, modelsQuery.data, statusQuery.data]
  )

  const applyDefaults = useCallback(
    async (nextDefaultsValues: ConfigurationDefaultsValues) => {
      const endpoint = controlConfigQuery.data?.bootstrap.endpoint?.trim()
      const snapshot = controlConfigQuery.data?.snapshot

      if (!endpoint || !snapshot) return null

      const applied = await applyRuntimeControlConfig(endpoint, snapshot, nextDefaultsValues)
      queryClient.setQueryData<RuntimeControlConfigResult>(CONFIG_CONTROL_QUERY_KEY, (current) => {
        if (!current) return current
        return {
          ...current,
          snapshot: applied.snapshot
        }
      })
      return applied.response
    },
    [controlConfigQuery.data, queryClient]
  )

  return {
    data,
    isPending: statusQuery.isPending || modelsQuery.isPending,
    isFetching: statusQuery.isFetching || modelsQuery.isFetching || controlConfigQuery.isFetching,
    isError: statusQuery.isError || modelsQuery.isError || controlConfigQuery.isError,
    statusQuery,
    modelsQuery,
    controlConfigQuery,
    applyDefaults
  }
}
