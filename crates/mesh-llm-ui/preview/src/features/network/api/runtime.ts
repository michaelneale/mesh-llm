import { ApiError } from '@/lib/api/errors'
import type { LlamaRuntimePayload } from '@/lib/api/types'
import { env } from '@/lib/env'

export async function fetchLlamaRuntime(): Promise<LlamaRuntimePayload> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/llama`)

  if (!response.ok) {
    const body = await response.text()
    throw new ApiError(response.status, body, `HTTP ${response.status}`)
  }

  return response.json() as Promise<LlamaRuntimePayload>
}

export async function fetchLlamaRuntimeWithSignal(signal: AbortSignal): Promise<LlamaRuntimePayload> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/llama`, { signal })

  if (!response.ok) {
    const body = await response.text()
    throw new ApiError(response.status, body, `HTTP ${response.status}`)
  }

  return response.json() as Promise<LlamaRuntimePayload>
}
