import { devices, expect, test, type Page } from '@playwright/test'

const DATA_MODE_STORAGE_KEY = 'mesh-llm-ui-preview:data-mode:v1'
const API_ORIGIN = 'http://127.0.0.1:3131'
const IPHONE_KEYBOARD_VIEWPORT = { width: 390, height: 520 }
const IPHONE_14 = devices['iPhone 14']
const LIVE_STATUS = {
  node_id: 'mobile-chat-node',
  node_state: 'serving',
  model_name: 'unsloth/Qwen3.5-2B-GGUF:Q4_K_M',
  peers: [],
  models: [],
  my_vram_gb: 24,
  gpus: [],
  serving_models: [],
  hostname: '127.0.0.1',
  api_port: 9337,
  token: 'mobile-chat-test-token',
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

test.use({
  viewport: IPHONE_14.viewport,
  deviceScaleFactor: IPHONE_14.deviceScaleFactor,
  hasTouch: IPHONE_14.hasTouch,
  isMobile: IPHONE_14.isMobile,
  userAgent: IPHONE_14.userAgent
})

async function installLiveChatBackend(page: Page) {
  await page.addInitScript(
    ({ storageKey, apiOrigin, statusPayload, modelsPayload }) => {
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
            this.onopen?.(new Event('open'))
            this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(statusPayload) }))
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

      const originalFetch = window.fetch.bind(window)

      function jsonResponse(payload: unknown) {
        return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } })
      }

      window.fetch = async (input, init) => {
        const request = input instanceof Request ? input : new Request(input, init)
        const url = new URL(request.url, window.location.href)

        if ((url.origin === apiOrigin || url.origin === window.location.origin) && url.pathname === '/api/status') {
          return jsonResponse(statusPayload)
        }
        if ((url.origin === apiOrigin || url.origin === window.location.origin) && url.pathname === '/api/models') {
          return jsonResponse(modelsPayload)
        }

        return originalFetch(input, init)
      }
    },
    {
      storageKey: DATA_MODE_STORAGE_KEY,
      apiOrigin: API_ORIGIN,
      statusPayload: LIVE_STATUS,
      modelsPayload: LIVE_MODELS
    }
  )
}

test.describe('mobile chat composer viewport', () => {
  test('keeps the focused composer visible when the keyboard reduces the visual viewport', async ({ page }) => {
    await installLiveChatBackend(page)

    await page.goto('/chat')

    const composer = page.locator('#prompt-composer')
    await expect(composer).toBeVisible()

    await composer.focus()
    await expect(composer).toBeFocused()

    await page.setViewportSize(IPHONE_KEYBOARD_VIEWPORT)

    await expect
      .poll(async () => {
        const composerBox = await composer.boundingBox()
        const layoutBox = await page.getByTestId('chat-layout').boundingBox()
        const viewport = page.viewportSize()
        const composerFontSize = await composer.evaluate((element) => Number.parseFloat(getComputedStyle(element).fontSize))

        if (!composerBox || !layoutBox || !viewport) return false

        const composerBottom = composerBox.y + composerBox.height
        const layoutBottom = layoutBox.y + layoutBox.height

        return (
          composerFontSize >= 16 &&
          composerBottom <= viewport.height - 8 &&
          layoutBottom <= viewport.height + 1 &&
          layoutBox.height >= 320
        )
      })
      .toBe(true)
  })
})
