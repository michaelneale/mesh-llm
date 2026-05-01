import { env } from '@/lib/env'
import type { StatusPayload } from '@/lib/api/types'
import { ApiError } from '@/lib/api/errors'

export async function fetchStatus(): Promise<StatusPayload> {
  const res = await fetch(`${env.apiUrl}/api/status`)
  if (!res.ok) {
    const body = await res.text()
    throw new ApiError(res.status, body, `HTTP ${res.status}`)
  }
  return res.json() as Promise<StatusPayload>
}
