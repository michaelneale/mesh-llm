import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  getCommandBarErrorMessage,
  isCommandBarAsyncSource,
  resolveCommandBarModeSource,
  type CommandBarResolvedMode,
} from './command-bar-helpers'
import type { CommandBarBehavior, CommandBarMode } from './command-bar-types'

interface UseCommandBarAsyncResultsOptions<T> {
  behavior: CommandBarBehavior
  isOpen: boolean
  modeById: ReadonlyMap<string, CommandBarMode<T>>
  modes: readonly CommandBarMode<T>[]
  query: string
  resolvedDistinctModeId: string | null
}

interface UseCommandBarAsyncResultsResult<T> {
  asyncErrorMessage?: string
  isLoading: boolean
  resolvedModes: CommandBarResolvedMode<T>[]
}

export function useCommandBarAsyncResults<T>({
  behavior,
  isOpen,
  modeById,
  modes,
  query,
  resolvedDistinctModeId,
}: UseCommandBarAsyncResultsOptions<T>): UseCommandBarAsyncResultsResult<T> {
  const abortControllersRef = useRef(new Map<string, AbortController>())
  const requestTokensByModeIdRef = useRef(new Map<string, number>())
  const previousIsOpenRef = useRef(false)
  const previousQueryRef = useRef(query)
  const previousAsyncModeIdsRef = useRef<string[]>([])
  const [asyncItemsByModeId, setAsyncItemsByModeId] = useState<Record<string, readonly T[]>>({})
  const [asyncLoadingByModeId, setAsyncLoadingByModeId] = useState<Record<string, boolean>>({})
  const [asyncErrorByModeId, setAsyncErrorByModeId] = useState<Record<string, string | null>>({})

  const resolvedModes = useMemo<CommandBarResolvedMode<T>[]>(() => {
    return modes.map((mode) => ({
      ...mode,
      source: resolveCommandBarModeSource(mode, asyncItemsByModeId),
    }))
  }, [asyncItemsByModeId, modes])

  const activeAsyncModeIds = useMemo(() => {
    if (behavior === 'distinct') {
      if (!resolvedDistinctModeId) return []
      const mode = modeById.get(resolvedDistinctModeId)
      return mode && isCommandBarAsyncSource(mode.source) ? [mode.id] : []
    }

    return modes.filter((mode) => isCommandBarAsyncSource(mode.source)).map((mode) => mode.id)
  }, [behavior, modeById, modes, resolvedDistinctModeId])

  const abortInFlightRequests = useCallback((modeIds?: readonly string[]) => {
    const targetModeIds = modeIds ?? Array.from(abortControllersRef.current.keys())

    targetModeIds.forEach((modeId) => {
      const controller = abortControllersRef.current.get(modeId)
      if (!controller) return
      controller.abort()
      abortControllersRef.current.delete(modeId)
    })
  }, [])

  const fetchAsyncResults = useCallback((modeIds: readonly string[], nextQuery: string) => {
    if (modeIds.length === 0) return

    modeIds.forEach((modeId) => {
      const mode = modeById.get(modeId)
      if (!mode || !isCommandBarAsyncSource(mode.source)) return

      abortInFlightRequests([modeId])

      const controller = new AbortController()
      const requestToken = (requestTokensByModeIdRef.current.get(modeId) ?? 0) + 1

      requestTokensByModeIdRef.current.set(modeId, requestToken)
      abortControllersRef.current.set(modeId, controller)
      setAsyncItemsByModeId((current) => ({ ...current, [modeId]: [] }))
      setAsyncLoadingByModeId((current) => ({ ...current, [modeId]: true }))
      setAsyncErrorByModeId((current) => ({ ...current, [modeId]: null }))

      mode.source(nextQuery, controller.signal)
        .then((items) => {
          if (controller.signal.aborted) return
          if (requestToken !== requestTokensByModeIdRef.current.get(modeId)) return

          abortControllersRef.current.delete(modeId)
          setAsyncItemsByModeId((current) => ({ ...current, [modeId]: items }))
          setAsyncErrorByModeId((current) => ({ ...current, [modeId]: null }))
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted) return
          if (requestToken !== requestTokensByModeIdRef.current.get(modeId)) return

          abortControllersRef.current.delete(modeId)
          setAsyncItemsByModeId((current) => ({ ...current, [modeId]: [] }))
          setAsyncErrorByModeId((current) => ({
            ...current,
            [modeId]: getCommandBarErrorMessage(error),
          }))
        })
        .finally(() => {
          if (controller.signal.aborted) return
          if (requestToken !== requestTokensByModeIdRef.current.get(modeId)) return
          setAsyncLoadingByModeId((current) => ({ ...current, [modeId]: false }))
        })
    })
  }, [abortInFlightRequests, modeById])

  useEffect(() => {
    const previouslyOpen = previousIsOpenRef.current
    const previousQuery = previousQueryRef.current
    const previousAsyncModeIds = previousAsyncModeIdsRef.current
    const modesChanged = previousAsyncModeIds.length !== activeAsyncModeIds.length || previousAsyncModeIds.some((modeId, index) => modeId !== activeAsyncModeIds[index])

    previousIsOpenRef.current = isOpen
    previousQueryRef.current = query
    previousAsyncModeIdsRef.current = [...activeAsyncModeIds]

    if (!isOpen) {
      abortInFlightRequests()
      requestTokensByModeIdRef.current.clear()
      return
    }

    if (activeAsyncModeIds.length === 0) return

    if (!previouslyOpen || modesChanged) {
      fetchAsyncResults(activeAsyncModeIds, query)
      return () => {
        abortInFlightRequests(activeAsyncModeIds)
      }
    }

    if (query === previousQuery) {
      return () => {
        abortInFlightRequests(activeAsyncModeIds)
      }
    }

    const timeoutId = window.setTimeout(() => {
      fetchAsyncResults(activeAsyncModeIds, query)
    }, 150)

    return () => {
      window.clearTimeout(timeoutId)
      abortInFlightRequests(activeAsyncModeIds)
    }
  }, [abortInFlightRequests, activeAsyncModeIds, fetchAsyncResults, isOpen, query])

  return {
    resolvedModes,
    isLoading: activeAsyncModeIds.some((modeId) => asyncLoadingByModeId[modeId]),
    asyncErrorMessage: activeAsyncModeIds
      .map((modeId) => asyncErrorByModeId[modeId])
      .find((message): message is string => Boolean(message)),
  }
}
