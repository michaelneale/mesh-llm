import { useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { statusKeys } from '@/lib/query/query-keys'
import type { StatusPayload } from '@/lib/api/types'
import { env } from '@/lib/env'

const MIN_BACKOFF = 1000
const MAX_BACKOFF = 15_000

export function useStatusStream(options?: { enabled?: boolean }) {
  const queryClient = useQueryClient()
  const enabled = options?.enabled ?? true
  const esRef = useRef<EventSource | null>(null)
  const backoffRef = useRef<number>(MIN_BACKOFF)
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  useEffect(() => {
    if (!enabled) return
    let active = true

    function clearReconnectTimer() {
      if (timerRef.current === undefined) return
      clearTimeout(timerRef.current)
      timerRef.current = undefined
    }

    function closeEventSource() {
      const es = esRef.current
      if (!es) return
      es.onopen = null
      es.onmessage = null
      es.onerror = null
      es.close()
      esRef.current = null
    }

    function scheduleReconnect(connect: () => void) {
      if (!active || timerRef.current !== undefined) return

      timerRef.current = setTimeout(() => {
        timerRef.current = undefined
        if (!active) return
        backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF)
        connect()
      }, backoffRef.current)
    }

    function connect() {
      if (!active) return
      clearReconnectTimer()
      closeEventSource()

      const es = new EventSource(`${env.managementApiUrl}/api/events`)
      esRef.current = es

      es.onopen = () => {
        if (!active) return
        backoffRef.current = MIN_BACKOFF
      }

      es.onmessage = (event: MessageEvent) => {
        if (!active) return
        try {
          const payload = JSON.parse(event.data as string) as StatusPayload
          queryClient.setQueryData(statusKeys.detail(), payload)
        } catch (_) {
          void _
        }
      }

      es.onerror = () => {
        if (esRef.current === es) {
          closeEventSource()
        } else {
          es.close()
        }
        scheduleReconnect(connect)
      }
    }

    connect()

    return () => {
      active = false
      clearReconnectTimer()
      closeEventSource()
    }
  }, [queryClient, enabled])
}
