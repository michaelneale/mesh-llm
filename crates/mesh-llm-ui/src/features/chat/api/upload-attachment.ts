import { env } from '@/lib/env'
import { ApiError, parseApiErrorBody } from '@/lib/api/errors'
import { generateRequestId } from '@/lib/api/request-id'
import type { AttachmentUploadRequest, AttachmentUploadResponse } from '@/lib/api/types'

const BASE64_CHUNK_SIZE = 0x8000

export type UploadAttachmentParams = {
  requestId: string
  bytesBase64: string
  mimeType: string
  fileName?: string
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = ''
  for (let index = 0; index < bytes.length; index += BASE64_CHUNK_SIZE) {
    binary += String.fromCharCode(...bytes.subarray(index, index + BASE64_CHUNK_SIZE))
  }
  return btoa(binary)
}

async function uploadAttachmentRequest(params: UploadAttachmentParams): Promise<string> {
  const body: AttachmentUploadRequest = {
    request_id: params.requestId,
    mime_type: params.mimeType,
    file_name: params.fileName ?? '',
    bytes_base64: params.bytesBase64
  }

  const response = await fetch(`${env.managementApiUrl}/api/objects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })

  if (!response.ok) {
    const errorBody = await parseApiErrorBody(response)
    throw new ApiError(response.status, errorBody, `Upload failed: ${response.status}`)
  }

  const data = (await response.json()) as AttachmentUploadResponse
  return data.token
}

export async function uploadAttachment(file: File): Promise<string>
export async function uploadAttachment(params: UploadAttachmentParams): Promise<string>
export async function uploadAttachment(fileOrParams: File | UploadAttachmentParams): Promise<string> {
  if (fileOrParams instanceof File) {
    const bytes = new Uint8Array(await fileOrParams.arrayBuffer())
    return uploadAttachmentRequest({
      requestId: generateRequestId(),
      bytesBase64: bytesToBase64(bytes),
      mimeType: fileOrParams.type,
      fileName: fileOrParams.name
    })
  }

  return uploadAttachmentRequest(fileOrParams)
}
