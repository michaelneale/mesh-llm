import { useEffect, useState } from 'react'
import type { LlamaRuntimePayload } from '@/lib/api/types'
import { env } from '@/lib/env'
import { fetchLlamaRuntimeWithSignal } from '@/features/network/api/runtime'

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
      return undefined
    }

    let cancelled = false
    let runtimeRequest: AbortController | null = null
    let runtimeEvents: EventSource | null = null
    let reconnectTimer: number | null = null
    let fallbackTimer: number | null = null

    const clearReconnectTimer = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
    }

    const clearFallbackTimer = () => {
      if (fallbackTimer !== null) {
        window.clearInterval(fallbackTimer)
        fallbackTimer = null
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
      runtimeRequest?.abort()
      runtimeRequest = null
    }

    async function loadRuntime() {
      abortRuntimeRequest()
      const controller = new AbortController()
      runtimeRequest = controller
      setState((current) => ({ ...current, loading: current.data == null, error: null }))

      try {
        const payload = await fetchLlamaRuntimeWithSignal(controller.signal)
        if (!cancelled && runtimeRequest === controller) {
          runtimeRequest = null
          setState({ data: payload, loading: false, error: null })
        }
      } catch (error) {
        if (cancelled || controller.signal.aborted) return
        if (runtimeRequest === controller) {
          runtimeRequest = null
        }
        setState((current) => ({
          data: current.data,
          loading: false,
          error: error instanceof Error ? error.message : 'Runtime llama request failed'
        }))
      }
    }

    const ensureFallbackPolling = () => {
      if (fallbackTimer !== null) return
      fallbackTimer = window.setInterval(() => void loadRuntime(), LLAMA_RUNTIME_REFRESH_MS)
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

    function connectRuntimeEvents() {
      if (cancelled || runtimeEvents) return

      let source: EventSource
      try {
        source = new EventSource(`${env.managementApiUrl}/api/runtime/events`)
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
        clearFallbackTimer()
        setState((current) => ({ ...current, error: null }))
      }
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data as string) as LlamaRuntimePayload
          if (!cancelled) {
            setState({ data: payload, loading: false, error: null })
          }
        } catch {
          // ignore malformed runtime event payloads
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
      abortRuntimeRequest()
      clearReconnectTimer()
      clearFallbackTimer()
      closeRuntimeEvents()
    }
  }, [enabled])

  return state
}
