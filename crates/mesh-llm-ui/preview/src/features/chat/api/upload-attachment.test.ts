import { beforeEach, describe, expect, it, vi } from 'vitest'

import { uploadAttachment } from './upload-attachment'

describe('uploadAttachment', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('posts the attachment bytes to /api/objects using the provided request id', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ token: 'blob-token' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(
      uploadAttachment({
        requestId: 'request-123',
        bytesBase64: 'aGVsbG8=',
        mimeType: 'text/plain',
        fileName: 'hello.txt'
      })
    ).resolves.toBe('blob-token')

    const [, request] = fetchMock.mock.calls[0] ?? []
    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/objects')
    expect(request).toMatchObject({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
    expect(JSON.parse(String(request.body))).toEqual({
      request_id: 'request-123',
      mime_type: 'text/plain',
      file_name: 'hello.txt',
      bytes_base64: 'aGVsbG8='
    })
  })

  it('supports the legacy file overload', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ token: 'blob-token' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    const file = new File([Uint8Array.from([104, 105])], 'greeting.txt', { type: 'text/plain' })

    await expect(uploadAttachment(file)).resolves.toBe('blob-token')

    const [, request] = fetchMock.mock.calls[0] ?? []
    const body = JSON.parse(String(request.body)) as {
      request_id: string
      mime_type: string
      file_name: string
      bytes_base64: string
    }
    expect(body.request_id).toBeTruthy()
    expect(body.mime_type).toBe('text/plain')
    expect(body.file_name).toBe('greeting.txt')
    expect(body.bytes_base64).toBe('aGk=')
  })

  it('base64-encodes large files without a single oversized argument spread', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ token: 'blob-token' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    const bytes = new Uint8Array(70_000)
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = index % 256
    }

    await expect(uploadAttachment(new File([bytes], 'large.bin', { type: 'application/octet-stream' }))).resolves.toBe(
      'blob-token'
    )

    const [, request] = fetchMock.mock.calls[0] ?? []
    const body = JSON.parse(String(request.body)) as { bytes_base64: string }
    const decoded = atob(body.bytes_base64)
    expect(decoded).toHaveLength(bytes.length)
    expect(decoded.charCodeAt(0)).toBe(bytes[0])
    expect(decoded.charCodeAt(32_768)).toBe(bytes[32_768])
    expect(decoded.charCodeAt(decoded.length - 1)).toBe(bytes.at(-1))
  })

  it('throws ApiError when the upload fails', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { message: 'Upload denied' } }), {
        status: 403,
        headers: { 'Content-Type': 'application/json' }
      })
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(
      uploadAttachment({
        requestId: 'request-123',
        bytesBase64: 'aGVsbG8=',
        mimeType: 'text/plain',
        fileName: 'hello.txt'
      })
    ).rejects.toMatchObject({
      name: 'ApiError',
      status: 403,
      body: 'Upload denied',
      message: 'Upload failed: 403'
    })
  })
})
