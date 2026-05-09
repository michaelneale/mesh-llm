import { expect, test, type Page, type TestInfo } from '@playwright/test'

const DATA_MODE_STORAGE_KEY = 'mesh-llm-ui-preview:data-mode:v1'
const API_ORIGIN = 'http://127.0.0.1:3131'
const CLIP_FIXTURE_PATH = decodeURIComponent(new URL('../fixtures/clip.mp3', import.meta.url).pathname)

const LIVE_STATUS = {
  node_id: '26d1bcf099',
  node_state: 'serving',
  model_name: 'unsloth/Qwen3.5-2B-GGUF:Q4_K_M',
  peers: [],
  models: [],
  my_vram_gb: 24,
  gpus: [],
  serving_models: [],
  hostname: '127.0.0.1',
  api_port: 9337,
  token:
    'eyJpZCI6IjI2ZDFiY2YwOTk4NTJiOGMzOWQ5NDZlNjQ3OTU0MGRhN2Y1OTI1YmM5YzExNmRjZmVmNmQ2NmNkZTU2MmRlNGEiLCJhZGRycyI6W3siUmVsYXkiOiJodHRwczovL3VzdzEtMi5yZWxheS5taWNoYWVsbmVhbGUubWVzaC1sbG0uaXJvaC5saW5rLi8ifSx7IklwIjoiMTAuNC4wLjE1NDo2Mzc3OCJ9LHsiSXAiOiIxMC40LjAuMTk4OjYzNzc4In0seyJJcCI6IjY0LjEzNy4xNTcuMTM5OjAifV19',
  llama_ready: true
}

const LIVE_MODELS = {
  mesh_models: [
    {
      name: 'Qwen3.5-2B-GGUF',
      status: 'ready',
      size_gb: 1.3,
      node_count: 1,
      capabilities: { vision: false, moe: false },
      quantization: 'Q4_K_M',
      context_length: 32768,
      family: 'qwen',
      tags: ['chat'],
      params_b: 2,
      disk_gb: 1.3
    }
  ]
}

type StreamScenario = {
  deltas?: string[]
  delayMs?: number
  failAfterChunkCount?: number
  errorMessage?: string
  status?: number
}

type BackendScenario = {
  statusPayload?: Record<string, unknown>
  modelsPayload?: Record<string, unknown>
  objectResponse?: {
    status: number
    body?: Record<string, unknown>
    errorMessage?: string
  }
  responseScenarios?: StreamScenario[]
}

type BackendState = {
  requestOrder: string[]
  objectRequests: Array<Record<string, unknown>>
  responseRequests: Array<Record<string, unknown>>
  responseAbortCount: number
}

function resolveAppUrl(pathname: string, testInfo: TestInfo) {
  const configuredBaseUrl = typeof testInfo.project.use.baseURL === 'string' ? testInfo.project.use.baseURL : undefined
  const env = typeof process !== 'undefined' && process.env ? process.env : {}
  const host = env.PLAYWRIGHT_HOST ?? '127.0.0.1'
  const port = env.PLAYWRIGHT_PORT ?? '5173'
  const baseUrl = configuredBaseUrl ?? `http://${host}:${port}`

  return new URL(pathname, baseUrl).toString()
}

async function installLiveBackend(page: Page, scenario: BackendScenario = {}) {
  await page.addInitScript(
    ({ storageKey, apiOrigin, statusPayload, modelsPayload, objectResponse, responseScenarios }) => {
      window.localStorage.setItem(storageKey, 'live')

      class MockEventSource {
        static readonly CONNECTING = 0
        static readonly OPEN = 1
        static readonly CLOSED = 2

        readonly CONNECTING = MockEventSource.CONNECTING
        readonly OPEN = MockEventSource.OPEN
        readonly CLOSED = MockEventSource.CLOSED
        readonly url: string
        readonly withCredentials = false
        readyState = MockEventSource.OPEN
        onopen: ((event: Event) => unknown) | null = null
        onmessage: ((event: MessageEvent<string>) => unknown) | null = null
        onerror: ((event: Event) => unknown) | null = null

        constructor(url: string | URL) {
          this.url = String(url)

          queueMicrotask(() => {
            const openHandler = this.onopen
            if (typeof openHandler === 'function') {
              openHandler(new Event('open'))
            }

            const messageHandler = this.onmessage
            if (typeof messageHandler === 'function') {
              messageHandler(new MessageEvent('message', { data: JSON.stringify(statusPayload) }))
            }
          })
        }

        addEventListener() {}
        removeEventListener() {}
        dispatchEvent() {
          return true
        }
        close() {
          this.readyState = MockEventSource.CLOSED
        }
      }

      Object.defineProperty(window, 'EventSource', {
        configurable: true,
        writable: true,
        value: MockEventSource
      })

      const backendState: {
        requestOrder: string[]
        objectRequests: Array<Record<string, unknown>>
        responseRequests: Array<Record<string, unknown>>
        responseAbortCount: number
      } = {
        requestOrder: [],
        objectRequests: [],
        responseRequests: [],
        responseAbortCount: 0
      }

      Object.defineProperty(window, '__liveParityBackendState', {
        configurable: true,
        writable: true,
        value: backendState
      })

      const originalFetch = window.fetch.bind(window)

      function jsonResponse(payload: unknown, status = 200) {
        return new Response(JSON.stringify(payload), {
          status,
          headers: { 'Content-Type': 'application/json' }
        })
      }

      window.fetch = async (input, init) => {
        const request = input instanceof Request ? input : new Request(input, init)
        const url = new URL(request.url, window.location.href)

        if (url.origin !== apiOrigin) {
          return originalFetch(input, init)
        }

        if (url.pathname === '/api/status') {
          return jsonResponse(statusPayload)
        }

        if (url.pathname === '/api/models') {
          return jsonResponse(modelsPayload)
        }

        if (url.pathname === '/api/objects') {
          backendState.requestOrder.push('objects')
          backendState.objectRequests.push((await request.clone().json()) as Record<string, unknown>)

          const uploadStatus = objectResponse?.status ?? 200

          if (uploadStatus >= 400) {
            return jsonResponse(
              { error: { message: objectResponse?.errorMessage ?? `Upload failed: ${uploadStatus}` } },
              uploadStatus
            )
          }

          return jsonResponse(objectResponse?.body ?? { token: 'blob-token-123' }, uploadStatus)
        }

        if (url.pathname === '/api/responses') {
          backendState.requestOrder.push('responses')
          backendState.responseRequests.push((await request.clone().json()) as Record<string, unknown>)

          const scenarioIndex = backendState.responseRequests.length - 1
          const responseScenario = responseScenarios[scenarioIndex] ??
            responseScenarios[responseScenarios.length - 1] ?? { deltas: [] }
          const responseStatus = responseScenario.status ?? 200

          if (responseStatus >= 400) {
            return jsonResponse(
              { error: { message: responseScenario.errorMessage ?? `Chat request failed: ${responseStatus}` } },
              responseStatus
            )
          }

          const deltas = responseScenario.deltas ?? ['Streaming reply from mock backend']
          const delayMs = responseScenario.delayMs ?? 0

          return new Response(
            new ReadableStream({
              start(controller) {
                const encoder = new TextEncoder()
                let closed = false
                let chunkIndex = 0
                let timer: number | undefined

                const cleanup = () => {
                  if (request.signal) {
                    request.signal.removeEventListener('abort', abortStream)
                  }
                  if (timer !== undefined) {
                    window.clearTimeout(timer)
                  }
                }

                const closeStream = () => {
                  if (closed) return
                  closed = true
                  cleanup()
                  controller.close()
                }

                const failStream = () => {
                  if (closed) return
                  closed = true
                  cleanup()
                  controller.error(new Error(responseScenario.errorMessage ?? 'Streaming request failed'))
                }

                const abortStream = () => {
                  backendState.responseAbortCount += 1
                  closeStream()
                }

                const pump = () => {
                  if (closed) return

                  if (request.signal?.aborted) {
                    abortStream()
                    return
                  }

                  if (
                    responseScenario.failAfterChunkCount !== undefined &&
                    chunkIndex === responseScenario.failAfterChunkCount
                  ) {
                    failStream()
                    return
                  }

                  if (chunkIndex < deltas.length) {
                    controller.enqueue(
                      encoder.encode(
                        `data: ${JSON.stringify({ type: 'response.output_text.delta', delta: deltas[chunkIndex] })}\n\n`
                      )
                    )
                    chunkIndex += 1
                    timer = window.setTimeout(pump, delayMs)
                    return
                  }

                  controller.enqueue(encoder.encode('data: [DONE]\n\n'))
                  closeStream()
                }

                request.signal?.addEventListener('abort', abortStream, { once: true })
                pump()
              }
            }),
            {
              status: responseStatus,
              headers: { 'Content-Type': 'text/event-stream' }
            }
          )
        }

        return originalFetch(input, init)
      }
    },
    {
      storageKey: DATA_MODE_STORAGE_KEY,
      apiOrigin: API_ORIGIN,
      statusPayload: scenario.statusPayload ?? LIVE_STATUS,
      modelsPayload: scenario.modelsPayload ?? LIVE_MODELS,
      objectResponse: scenario.objectResponse ?? { status: 200, body: { token: 'blob-token-123' } },
      responseScenarios: scenario.responseScenarios ?? [{ deltas: ['Streaming reply from mock backend'] }]
    }
  )
}

async function readBackendState(page: Page): Promise<BackendState> {
  return page.evaluate(() => (window as unknown as { __liveParityBackendState: BackendState }).__liveParityBackendState)
}

test('live shell metadata shows status-backed API and invite instructions', async ({ page }, testInfo) => {
  await installLiveBackend(page)

  await page.goto(resolveAppUrl('/chat', testInfo))
  await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible()

  await page.getByRole('button', { name: 'API target instructions' }).click()
  const apiCard = page.getByLabel('API access')
  await expect(page.getByRole('heading', { name: 'API access' })).toBeVisible()
  await expect(
    apiCard.getByText(`http://${LIVE_STATUS.hostname}:${LIVE_STATUS.api_port}/v1`, { exact: true })
  ).toBeVisible()
  await expect(
    apiCard.getByText(`curl http://${LIVE_STATUS.hostname}:${LIVE_STATUS.api_port}/v1/models`, { exact: true })
  ).toBeVisible()
  await expect(
    apiCard.getByText(`OPENAI_BASE_URL=http://${LIVE_STATUS.hostname}:${LIVE_STATUS.api_port}/v1`, { exact: true })
  ).toBeVisible()
  await expect(page.getByText('test harness')).toHaveCount(0)

  await page.getByRole('button', { name: 'Mesh join and invite instructions' }).click()
  const joinCard = page.getByLabel('Join or invite')
  await expect(page.getByRole('heading', { name: 'Join or invite' })).toBeVisible()
  await expect(joinCard.getByText(LIVE_STATUS.token, { exact: true })).toBeVisible()
  await expect(joinCard.getByText(`mesh-llm --auto --join ${LIVE_STATUS.token}`, { exact: true })).toBeVisible()
  await expect(joinCard.getByText(`mesh-llm client --join ${LIVE_STATUS.token}`, { exact: true })).toBeVisible()
  await expect(page.getByText('<mesh-invite-token>')).toHaveCount(0)
})

test('missing invite token degrades safely in live shell mode', async ({ page }, testInfo) => {
  const statusWithoutToken = { ...LIVE_STATUS, token: '' }
  await installLiveBackend(page, { statusPayload: statusWithoutToken })

  await page.goto(resolveAppUrl('/', testInfo))
  await page.getByRole('button', { name: 'Mesh join and invite instructions' }).click()

  const joinCard = page.getByLabel('Join or invite')
  await expect(joinCard.getByText('Invite token unavailable', { exact: true })).toBeVisible()
  await expect(joinCard.getByText('Auto join command unavailable', { exact: true })).toBeVisible()
  await expect(joinCard.getByText('Client-only join command unavailable', { exact: true })).toBeVisible()
  await expect(joinCard.getByText('<mesh-invite-token>')).toHaveCount(0)

  await page.getByRole('button', { name: 'API target instructions' }).click()
  await expect(page.getByRole('heading', { name: 'API access' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Mesh join and invite instructions' })).toBeEnabled()
})

test('live chat send, stream, stop, and retry work end to end', async ({ page }, testInfo) => {
  await installLiveBackend(page, {
    responseScenarios: [
      {
        deltas: ['Partial assistant reply', ' that should not finish'],
        delayMs: 250
      },
      {
        deltas: ['Retried assistant reply'],
        delayMs: 20
      }
    ]
  })

  await page.goto(resolveAppUrl('/chat', testInfo))
  await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible()

  await page.locator('#prompt-composer').fill('hello mesh')
  await page.getByTestId('chat-send').click()

  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toBeVisible()
  await expect(page.getByText('Generating response…')).toBeVisible()
  await expect(page.getByText('Partial assistant reply')).toBeVisible()

  await page.getByRole('button', { name: 'Stop', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Send' })).toBeVisible()
  await expect(page.getByText('Partial assistant reply')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Retry last' })).toBeEnabled()

  await page.getByRole('button', { name: 'Retry last' }).click()

  await expect(page.getByText('Retried assistant reply')).toBeVisible()
  await expect(page.getByText('Partial assistant reply')).toHaveCount(0)

  const backendState = await readBackendState(page)
  expect(backendState.responseRequests).toHaveLength(2)
  expect(backendState.responseAbortCount).toBeGreaterThanOrEqual(1)
})

test('streaming failure leaves partial text and recoverable chat state', async ({ page }, testInfo) => {
  await installLiveBackend(page, {
    responseScenarios: [
      {
        deltas: ['Partial stream text'],
        delayMs: 50,
        failAfterChunkCount: 1,
        errorMessage: 'Streaming request failed after partial text'
      }
    ]
  })

  await page.goto(resolveAppUrl('/chat', testInfo))
  await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible()

  await page.locator('#prompt-composer').fill('hello mesh')
  await page.getByTestId('chat-send').click()

  await expect(page.getByText('Partial stream text')).toBeVisible()
  await expect(page.getByTestId('chat-send')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Retry last' })).toBeEnabled()
  await expect(page.getByText('Generating response…')).toHaveCount(0)
})

test('live chat uploads an attachment before sending the parity request', async ({ page }, testInfo) => {
  await installLiveBackend(page)

  await page.goto(resolveAppUrl('/chat', testInfo))
  await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible()
  await expect(page.getByTestId('chat-send')).toBeVisible()

  await page.locator('input[type="file"]').setInputFiles(CLIP_FIXTURE_PATH)
  await expect(page.getByText('1 attachment ready')).toBeVisible()

  await page.locator('#prompt-composer').fill('Keep this prompt')
  await page.getByTestId('chat-send').click()

  await expect(page.getByText('Streaming reply from mock backend')).toBeVisible()

  const backendState = await readBackendState(page)
  expect(backendState.requestOrder).toEqual(['objects', 'responses'])
  expect(backendState.objectRequests).toHaveLength(1)
  expect(backendState.objectRequests[0]).toMatchObject({
    file_name: 'clip.mp3',
    mime_type: 'audio/mpeg'
  })

  expect(backendState.responseRequests).toHaveLength(1)
  expect(backendState.responseRequests[0]).toMatchObject({
    model: LIVE_STATUS.model_name,
    stream: true
  })

  const responseInput = backendState.responseRequests[0]?.input
  expect(Array.isArray(responseInput)).toBe(true)
  const firstMessage = (responseInput as Array<Record<string, unknown>>)[0]
  expect(firstMessage?.role).toBe('user')
  const content = firstMessage?.content as Array<Record<string, unknown>>
  expect(content[0]).toEqual({ type: 'input_text', text: 'Keep this prompt' })
  expect(content[1]).toMatchObject({ type: 'input_audio' })
  expect(String(content[1]?.audio_url ?? '')).toMatch(/^mesh:\/\/blob\/[^/]+\/blob-token-123$/)
})

test('attachment upload failure keeps the prompt and skips the chat request', async ({ page }, testInfo) => {
  await installLiveBackend(page, {
    objectResponse: {
      status: 503,
      errorMessage: 'Upload backend unavailable'
    }
  })

  await page.goto(resolveAppUrl('/chat', testInfo))
  await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible()

  await page.locator('input[type="file"]').setInputFiles(CLIP_FIXTURE_PATH)
  await expect(page.getByText('1 attachment ready')).toBeVisible()

  await page.locator('#prompt-composer').fill('Keep this prompt')
  await page.getByTestId('chat-send').click()

  await expect(page.getByRole('alert')).toContainText('Upload failed: 503')
  await expect(page.locator('#prompt-composer')).toHaveValue('Keep this prompt')

  const backendState = await readBackendState(page)
  expect(backendState.requestOrder).toEqual(['objects'])
  expect(backendState.responseRequests).toHaveLength(0)
})
