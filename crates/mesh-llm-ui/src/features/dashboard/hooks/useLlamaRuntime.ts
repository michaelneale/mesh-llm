import { useEffect, useState } from 'react'

import type { LlamaRuntimePayload } from '@/features/app-shell/lib/status-types'

const LLAMA_RUNTIME_REFRESH_MS = 2_500
const LLAMA_RUNTIME_RECONNECT_MS = 1_000

type LlamaRuntimeState = {
  data: LlamaRuntimePayload | null
  loading: boolean
  error: string | null
}

export function useLlamaRuntime(enabled: boolean) {
  const [state, setState] = useState<LlamaRuntimeState>({
    data: null,
    loading: false,
    error: null
  })

  useEffect(() => {
    if (!enabled) {
      setState({ data: null, loading: false, error: null })
      return
    }

    let cancelled = false
    let controller: AbortController | null = null
    let runtimeEvents: EventSource | null = null
    let reconnectTimer: number | null = null
    let fallbackInterval: number | null = null

    const clearReconnectTimer = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
    }

    const clearFallbackInterval = () => {
      if (fallbackInterval !== null) {
        window.clearInterval(fallbackInterval)
        fallbackInterval = null
      }
    }

    const closeRuntimeEvents = () => {
      if (!runtimeEvents) return
      runtimeEvents.onopen = null
      runtimeEvents.onmessage = null
      runtimeEvents.onerror = null
      runtimeEvents.close()
      runtimeEvents = null
    }

    const abortRuntimeRequest = () => {
      controller?.abort()
      controller = null
    }

    async function loadRuntime() {
      abortRuntimeRequest()
      const requestController = new AbortController()
      controller = requestController
      setState((current) => ({ ...current, loading: current.data == null, error: null }))
      try {
        const response = await fetch('/api/runtime/llama', {
          signal: requestController.signal
        })
        if (!response.ok) {
          throw new Error(`Runtime llama request failed with HTTP ${response.status}`)
        }
        const payload = (await response.json()) as LlamaRuntimePayload
        if (!cancelled && controller === requestController) {
          controller = null
          setState({ data: payload, loading: false, error: null })
        }
      } catch (error) {
        if (requestController.signal.aborted || cancelled) return
        if (controller === requestController) {
          controller = null
        }
        setState((current) => ({
          data: current.data,
          loading: false,
          error: error instanceof Error ? error.message : 'Runtime llama request failed'
        }))
      }
    }

    const ensureFallbackPolling = () => {
      if (fallbackInterval !== null) return
      fallbackInterval = window.setInterval(() => void loadRuntime(), LLAMA_RUNTIME_REFRESH_MS)
    }

    const stopFallbackPolling = () => {
      clearFallbackInterval()
    }

    const scheduleReconnect = () => {
      if (cancelled || reconnectTimer !== null) return
      ensureFallbackPolling()
      closeRuntimeEvents()
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null
        connectRuntimeEvents()
      }, LLAMA_RUNTIME_RECONNECT_MS)
    }

    const connectRuntimeEvents = () => {
      if (cancelled || runtimeEvents) return

      let source: EventSource
      try {
        source = new EventSource('/api/runtime/events')
      } catch (error) {
        console.warn(
          'Failed to connect llama runtime stream:',
          error instanceof Error ? error.message : 'failed to create EventSource'
        )
        scheduleReconnect()
        return
      }

      runtimeEvents = source
      source.onopen = () => {
        if (cancelled) return
        abortRuntimeRequest()
        stopFallbackPolling()
        setState((current) => ({ ...current, error: null }))
      }
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as LlamaRuntimePayload
          if (!cancelled) {
            setState({ data: payload, loading: false, error: null })
          }
        } catch {
          // ignore malformed runtime event
        }
      }
      source.onerror = () => {
        if (cancelled) return
        scheduleReconnect()
      }
    }

    void loadRuntime()
    connectRuntimeEvents()
    return () => {
      cancelled = true
      controller?.abort()
      clearReconnectTimer()
      clearFallbackInterval()
      closeRuntimeEvents()
    }
  }, [enabled])

  return state
}
