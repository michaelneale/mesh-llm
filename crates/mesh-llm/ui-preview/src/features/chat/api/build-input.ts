import type { UIMessage, ModelMessage, MessagePart, ContentPart } from '@tanstack/ai'
import type { ResponsesRequest, ResponsesInputMessage, ResponsesInputContentBlock } from '../../../lib/api/types'
import { uploadAttachment } from './upload-attachment'

type PartMetadata = {
  fileName?: string
}

function getPartMetadataFileName(metadata: unknown): string | undefined {
  return typeof metadata === 'object' && metadata !== null && 'fileName' in metadata
    ? (metadata as PartMetadata).fileName
    : undefined
}

async function uploadDataSource(
  source: { type: 'data'; value: string; mimeType: string },
  clientId: string,
  requestId: string,
  fileName?: string
) {
  const token = await uploadAttachment({
    requestId,
    bytesBase64: source.value,
    mimeType: source.mimeType,
    fileName
  })

  return `mesh://blob/${clientId}/${token}`
}

function isUIMessage(msg: UIMessage | ModelMessage): msg is UIMessage {
  return 'parts' in msg
}

async function partToContentBlock(
  part: MessagePart,
  clientId: string,
  requestId: string
): Promise<ResponsesInputContentBlock | null> {
  if (part.type === 'text') {
    return { type: 'input_text', text: part.content }
  }
  if (part.type === 'image') {
    const image_url =
      part.source.type === 'url' ? part.source.value : `data:${part.source.mimeType};base64,${part.source.value}`
    return { type: 'input_image', image_url }
  }
  if (part.type === 'audio') {
    const audio_url =
      part.source.type === 'url'
        ? part.source.value
        : await uploadDataSource(part.source, clientId, requestId, getPartMetadataFileName(part.metadata))
    return { type: 'input_audio', audio_url }
  }
  if (part.type === 'document') {
    const url =
      part.source.type === 'url'
        ? part.source.value
        : await uploadDataSource(part.source, clientId, requestId, getPartMetadataFileName(part.metadata))
    return {
      type: 'input_file',
      url,
      mime_type: part.source.mimeType,
      file_name: getPartMetadataFileName(part.metadata)
    }
  }
  return null
}

async function contentPartToBlock(
  part: ContentPart,
  clientId: string,
  requestId: string
): Promise<ResponsesInputContentBlock | null> {
  if (part.type === 'text') {
    return { type: 'input_text', text: part.content }
  }
  if (part.type === 'image') {
    const image_url =
      part.source.type === 'url' ? part.source.value : `data:${part.source.mimeType};base64,${part.source.value}`
    return { type: 'input_image', image_url }
  }
  if (part.type === 'audio') {
    const audio_url =
      part.source.type === 'url'
        ? part.source.value
        : await uploadDataSource(part.source, clientId, requestId, getPartMetadataFileName(part.metadata))
    return { type: 'input_audio', audio_url }
  }
  if (part.type === 'document') {
    const url =
      part.source.type === 'url'
        ? part.source.value
        : await uploadDataSource(part.source, clientId, requestId, getPartMetadataFileName(part.metadata))
    return {
      type: 'input_file',
      url,
      mime_type: part.source.mimeType,
      file_name: getPartMetadataFileName(part.metadata)
    }
  }
  return null
}

async function convertUIMessage(
  msg: UIMessage,
  clientId: string,
  requestId: string
): Promise<ResponsesInputMessage | null> {
  if (msg.role === 'system') return null
  const role = msg.role as 'user' | 'assistant'
  const content: ResponsesInputContentBlock[] = (
    await Promise.all(msg.parts.map((part) => partToContentBlock(part, clientId, requestId)))
  ).filter((b): b is ResponsesInputContentBlock => b !== null)
  if (content.length === 0) return null
  return { role, content: content.length === 1 && content[0]?.type === 'input_text' ? content[0].text : content }
}

async function convertModelMessage(
  msg: ModelMessage,
  clientId: string,
  requestId: string
): Promise<ResponsesInputMessage | null> {
  if (msg.role === 'tool') return null
  const role = msg.role as 'user' | 'assistant'

  if (msg.content === null) return null

  if (typeof msg.content === 'string') {
    if (msg.content.length === 0) return null
    return { role, content: msg.content }
  }

  const content: ResponsesInputContentBlock[] = (
    await Promise.all(msg.content.map((part) => contentPartToBlock(part, clientId, requestId)))
  ).filter((b): b is ResponsesInputContentBlock => b !== null)

  if (content.length === 0) return null
  return { role, content: content.length === 1 && content[0]?.type === 'input_text' ? content[0].text : content }
}

export async function buildResponsesInput(
  messages: Array<UIMessage> | Array<ModelMessage>,
  model: string,
  clientId: string,
  requestId: string,
  systemPrompt = ''
): Promise<ResponsesRequest> {
  const input: ResponsesInputMessage[] = (
    await Promise.all(
      (messages as ReadonlyArray<UIMessage | ModelMessage>).map((msg) =>
        isUIMessage(msg) ? convertUIMessage(msg, clientId, requestId) : convertModelMessage(msg, clientId, requestId)
      )
    )
  ).filter((m): m is ResponsesInputMessage => m !== null)
  const trimmedSystemPrompt = systemPrompt.trim()
  const messagesWithSystemPrompt: ResponsesInputMessage[] = trimmedSystemPrompt
    ? [{ role: 'system', content: trimmedSystemPrompt }, ...input]
    : input

  return {
    model,
    client_id: clientId,
    request_id: requestId,
    input: messagesWithSystemPrompt,
    stream: true,
    stream_options: { include_usage: true }
  }
}
