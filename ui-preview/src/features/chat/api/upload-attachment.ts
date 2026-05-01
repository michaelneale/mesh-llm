import { env } from '../../../lib/env'
import { ApiError, parseApiErrorBody } from '../../../lib/api/errors'
import { generateRequestId } from '../../../lib/api/request-id'
import type { AttachmentUploadRequest, AttachmentUploadResponse } from '../../../lib/api/types'

export async function uploadAttachment(file: File): Promise<string> {
  const bytes = await file.arrayBuffer()
  const bytes_base64 = btoa(String.fromCharCode(...new Uint8Array(bytes)))

  const body: AttachmentUploadRequest = {
    request_id: generateRequestId(),
    mime_type: file.type,
    file_name: file.name,
    bytes_base64,
  }

  const response = await fetch(`${env.apiUrl}/api/objects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const errorBody = await parseApiErrorBody(response)
    throw new ApiError(response.status, errorBody, `Upload failed: ${response.status}`)
  }

  const data = (await response.json()) as AttachmentUploadResponse
  return data.token
}
