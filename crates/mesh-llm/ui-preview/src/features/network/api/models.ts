import { env } from '@/lib/env'
import type { ModelsResponse } from '@/lib/api/types'
import { ApiError } from '@/lib/api/errors'

export async function fetchModels(): Promise<ModelsResponse> {
  const res = await fetch(`${env.managementApiUrl}/api/models`)
  if (!res.ok) {
    const body = await res.text()
    throw new ApiError(res.status, body, `HTTP ${res.status}`)
  }
  return res.json() as Promise<ModelsResponse>
}
