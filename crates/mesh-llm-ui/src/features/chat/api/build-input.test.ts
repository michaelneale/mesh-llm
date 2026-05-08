import type { UIMessage } from '@tanstack/ai'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const uploadAttachmentMock = vi.hoisted(() => vi.fn())

vi.mock('@/features/chat/api/upload-attachment', () => ({
  uploadAttachment: uploadAttachmentMock
}))

import { buildResponsesInput } from '@/features/chat/api/build-input'

function createMessage(parts: UIMessage['parts']): UIMessage {
  return {
    id: 'user-1',
    role: 'user',
    createdAt: new Date('2026-05-06T00:00:00.000Z'),
    parts
  }
}

describe('buildResponsesInput', () => {
  beforeEach(() => {
    uploadAttachmentMock.mockReset()
  })

  it('uploads audio and generic file parts with the shared request id and client id', async () => {
    uploadAttachmentMock.mockResolvedValueOnce('audio-token').mockResolvedValueOnce('file-token')

    const request = await buildResponsesInput(
      [
        createMessage([
          { type: 'text', content: 'Inspect these attachments' },
          {
            type: 'audio',
            source: { type: 'data', value: 'YXVkaW8=', mimeType: 'audio/mpeg' },
            metadata: { fileName: 'note.mp3' }
          },
          {
            type: 'document',
            source: { type: 'data', value: 'ZmlsZQ==', mimeType: 'application/octet-stream' },
            metadata: { fileName: 'notes.bin' }
          }
        ])
      ],
      'model-a',
      'client-123',
      'request-456'
    )

    expect(uploadAttachmentMock).toHaveBeenNthCalledWith(1, {
      requestId: 'request-456',
      bytesBase64: 'YXVkaW8=',
      mimeType: 'audio/mpeg',
      fileName: 'note.mp3'
    })
    expect(uploadAttachmentMock).toHaveBeenNthCalledWith(2, {
      requestId: 'request-456',
      bytesBase64: 'ZmlsZQ==',
      mimeType: 'application/octet-stream',
      fileName: 'notes.bin'
    })
    expect(request.input).toEqual([
      {
        role: 'user',
        content: [
          { type: 'input_text', text: 'Inspect these attachments' },
          { type: 'input_audio', audio_url: 'mesh://blob/client-123/audio-token' },
          {
            type: 'input_file',
            url: 'mesh://blob/client-123/file-token',
            mime_type: 'application/octet-stream',
            file_name: 'notes.bin'
          }
        ]
      }
    ])
  })

  it('preserves simple text messages as a string payload', async () => {
    const request = await buildResponsesInput(
      [createMessage([{ type: 'text', content: 'Hello mesh' }])],
      'model-a',
      'client-123',
      'request-456'
    )

    expect(uploadAttachmentMock).not.toHaveBeenCalled()
    expect(request.input).toEqual([{ role: 'user', content: 'Hello mesh' }])
  })

  it('prepends the saved system prompt as a responses system message', async () => {
    const request = await buildResponsesInput(
      [createMessage([{ type: 'text', content: 'Explain the cluster status' }])],
      'model-a',
      'client-123',
      'request-456',
      'Answer like a mesh-llm operator.'
    )

    expect(request.input).toEqual([
      { role: 'system', content: 'Answer like a mesh-llm operator.' },
      { role: 'user', content: 'Explain the cluster status' }
    ])
  })

  it('passes through url-backed audio and document parts without reuploading', async () => {
    const request = await buildResponsesInput(
      [
        createMessage([
          { type: 'audio', source: { type: 'url', value: 'mesh://blob/client-123/audio-token' } },
          {
            type: 'document',
            source: { type: 'url', value: 'mesh://blob/client-123/file-token', mimeType: 'text/plain' },
            metadata: { fileName: 'notes.txt' }
          }
        ])
      ],
      'model-a',
      'client-123',
      'request-456'
    )

    expect(uploadAttachmentMock).not.toHaveBeenCalled()
    expect(request.input).toEqual([
      {
        role: 'user',
        content: [
          { type: 'input_audio', audio_url: 'mesh://blob/client-123/audio-token' },
          {
            type: 'input_file',
            url: 'mesh://blob/client-123/file-token',
            mime_type: 'text/plain',
            file_name: 'notes.txt'
          }
        ]
      }
    ])
  })

  it('converts image parts to backend-compatible image_url blocks without uploading', async () => {
    const request = await buildResponsesInput(
      [
        createMessage([
          { type: 'image', source: { type: 'data', value: 'aW1hZ2U=', mimeType: 'image/png' } },
          { type: 'image', source: { type: 'url', value: 'mesh://blob/client-123/image-token' } }
        ])
      ],
      'model-a',
      'client-123',
      'request-456'
    )

    expect(uploadAttachmentMock).not.toHaveBeenCalled()
    expect(request.input).toEqual([
      {
        role: 'user',
        content: [
          { type: 'input_image', image_url: 'data:image/png;base64,aW1hZ2U=' },
          { type: 'input_image', image_url: 'mesh://blob/client-123/image-token' }
        ]
      }
    ])
  })
})
