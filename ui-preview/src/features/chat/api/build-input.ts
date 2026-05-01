import type { UIMessage, ModelMessage, MessagePart, ContentPart } from '@tanstack/ai'
import type {
  ResponsesRequest,
  ResponsesInputMessage,
  ResponsesInputContentBlock,
} from '../../../lib/api/types'

function isUIMessage(msg: UIMessage | ModelMessage): msg is UIMessage {
  return 'parts' in msg
}

function partToContentBlock(part: MessagePart): ResponsesInputContentBlock | null {
  if (part.type === 'text') {
    return { type: 'input_text', text: part.content }
  }
  if (part.type === 'image') {
    const url =
      part.source.type === 'url'
        ? part.source.value
        : `data:${part.source.mimeType};base64,${part.source.value}`
    return { type: 'input_image', url }
  }
  if (part.type === 'document') {
    const url =
      part.source.type === 'url'
        ? part.source.value
        : `data:${part.source.mimeType};base64,${part.source.value}`
    return {
      type: 'input_file',
      url,
      mime_type: part.source.mimeType,
    }
  }
  return null
}

function contentPartToBlock(part: ContentPart): ResponsesInputContentBlock | null {
  if (part.type === 'text') {
    return { type: 'input_text', text: part.content }
  }
  if (part.type === 'image') {
    const url =
      part.source.type === 'url'
        ? part.source.value
        : `data:${part.source.mimeType};base64,${part.source.value}`
    return { type: 'input_image', url }
  }
  if (part.type === 'document') {
    const url =
      part.source.type === 'url'
        ? part.source.value
        : `data:${part.source.mimeType};base64,${part.source.value}`
    return {
      type: 'input_file',
      url,
      mime_type: part.source.mimeType,
    }
  }
  return null
}

function convertUIMessage(msg: UIMessage): ResponsesInputMessage | null {
  if (msg.role === 'system') return null
  const role = msg.role as 'user' | 'assistant'
  const content: ResponsesInputContentBlock[] = msg.parts
    .map(partToContentBlock)
    .filter((b): b is ResponsesInputContentBlock => b !== null)
  if (content.length === 0) return null
  return { role, content }
}

function convertModelMessage(msg: ModelMessage): ResponsesInputMessage | null {
  if (msg.role === 'tool') return null
  const role = msg.role as 'user' | 'assistant'

  if (msg.content === null) return null

  if (typeof msg.content === 'string') {
    if (msg.content.length === 0) return null
    return { role, content: [{ type: 'input_text', text: msg.content }] }
  }

  const content: ResponsesInputContentBlock[] = msg.content
    .map(contentPartToBlock)
    .filter((b): b is ResponsesInputContentBlock => b !== null)

  if (content.length === 0) return null
  return { role, content }
}

export function buildResponsesInput(
  messages: Array<UIMessage> | Array<ModelMessage>,
  model: string,
  clientId: string,
  requestId: string,
): ResponsesRequest {
  const input: ResponsesInputMessage[] = (
    messages as ReadonlyArray<UIMessage | ModelMessage>
  )
    .map((msg) => (isUIMessage(msg) ? convertUIMessage(msg) : convertModelMessage(msg)))
    .filter((m): m is ResponsesInputMessage => m !== null)

  return {
    model,
    client_id: clientId,
    request_id: requestId,
    input,
    stream: true,
    stream_options: { include_usage: true },
  }
}
