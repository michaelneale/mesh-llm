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

    function connect() {
      const es = new EventSource(`${env.apiUrl}/api/events`)
      esRef.current = es

      es.onopen = () => {
        backoffRef.current = MIN_BACKOFF
      }

      es.onmessage = (event: MessageEvent) => {
        try {
          const payload = JSON.parse(event.data as string) as StatusPayload
          queryClient.setQueryData(statusKeys.detail(), payload)
        } catch (_) {
          void _
        }
      }

      es.onerror = () => {
        es.close()
        timerRef.current = setTimeout(() => {
          backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF)
          connect()
        }, backoffRef.current)
      }
    }

    connect()

    return () => {
      esRef.current?.close()
      clearTimeout(timerRef.current)
    }
  }, [queryClient, enabled])
}
