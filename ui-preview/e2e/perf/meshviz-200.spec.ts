import { expect, test, type Page } from '@playwright/test'

const MEASUREMENT_MS = 6000
const FRAME_BUDGET_MS = 16.7
const JANK_FRAME_MS = 33.3
const MIN_AVG_FPS = 30
const MAX_P95_FRAME_MS = 50
const MAX_JANK_FRAME_RATIO = 0.2
const NODE_SIZE_DELTA_TOLERANCE_PX = 1

type BrowserMeasurement = {
  scenario: 'meshviz-pan-zoom-200'
  nodeCount: number
  durationMs: number
  frameCount: number
  avgFps: number
  p50FrameMs: number
  p95FrameMs: number
  maxFrameMs: number
  framesOver16_7ms: number
  framesOver33_3ms: number
  longTaskCount: number
  longTaskTotalMs: number
  longAnimationFrameCount: number
  longAnimationFrameTotalMs: number
}

type CssTranslate = {
  x: number
  y: number
}

declare global {
  interface Window {
    __meshVizMeasurement?: Promise<BrowserMeasurement>
  }
}

test.use({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 })

test('measures MeshViz pan and zoom performance with 200 nodes', async ({ page }, testInfo) => {
  await page.goto('/__meshviz-perf')
  await expect(page.getByRole('heading', { name: 'MeshViz 200-node benchmark' })).toBeVisible()

  const canvas = page.getByTestId('mesh-canvas')
  await expect(canvas).toBeVisible()
  await expect(page.getByTestId('mesh-node-core')).toHaveCount(200)

  await page.evaluate(
    ({ durationMs, frameBudgetMs, jankFrameMs }) => {
      const frameDeltas: number[] = []
      const longTasks: number[] = []
      const longAnimationFrames: number[] = []
      const observers: PerformanceObserver[] = []
      let previousFrameTimestamp: number | undefined
      let lastFrameTimestamp: number | undefined
      let frameCount = 0
      let animationFrameId = 0
      let completed = false

      const round = (value: number) => Math.round(value * 100) / 100
      const sum = (values: number[]) => values.reduce((total, value) => total + value, 0)
      const percentile = (values: number[], percentileValue: number) => {
        if (values.length === 0) return 0
        const sorted = [...values].sort((left, right) => left - right)
        const index = Math.min(sorted.length - 1, Math.ceil((percentileValue / 100) * sorted.length) - 1)
        return sorted[index] ?? 0
      }
      const observePerformanceEntries = (entryType: string, durations: number[]) => {
        if (!PerformanceObserver.supportedEntryTypes.includes(entryType)) return
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) durations.push(entry.duration)
        })
        observer.observe({ type: entryType, buffered: true })
        observers.push(observer)
      }
      const createMeasurement = (measuredDurationMs: number): BrowserMeasurement => ({
        scenario: 'meshviz-pan-zoom-200',
        nodeCount: document.querySelectorAll('[data-testid="mesh-node-core"]').length,
        durationMs: round(measuredDurationMs),
        frameCount,
        avgFps: round(frameDeltas.length / Math.max(measuredDurationMs / 1000, 0.001)),
        p50FrameMs: round(percentile(frameDeltas, 50)),
        p95FrameMs: round(percentile(frameDeltas, 95)),
        maxFrameMs: round(frameDeltas.length > 0 ? Math.max(...frameDeltas) : 0),
        framesOver16_7ms: frameDeltas.filter((delta) => delta > frameBudgetMs).length,
        framesOver33_3ms: frameDeltas.filter((delta) => delta > jankFrameMs).length,
        longTaskCount: longTasks.length,
        longTaskTotalMs: round(sum(longTasks)),
        longAnimationFrameCount: longAnimationFrames.length,
        longAnimationFrameTotalMs: round(sum(longAnimationFrames))
      })

      observePerformanceEntries('longtask', longTasks)
      observePerformanceEntries('long-animation-frame', longAnimationFrames)

      window.__meshVizMeasurement = new Promise((resolve) => {
        const startedAt = performance.now()
        const finish = (finishedAt: number) => {
          if (completed) return
          completed = true
          cancelAnimationFrame(animationFrameId)
          observers.forEach((observer) => {
            observer.disconnect()
          })
          resolve(createMeasurement(finishedAt - startedAt))
        }
        const tick = (timestamp: number) => {
          if (previousFrameTimestamp !== undefined) frameDeltas.push(timestamp - previousFrameTimestamp)
          previousFrameTimestamp = timestamp
          lastFrameTimestamp = timestamp
          frameCount += 1
          if (timestamp - startedAt >= durationMs) {
            finish(timestamp)
            return
          }
          animationFrameId = requestAnimationFrame(tick)
        }

        animationFrameId = requestAnimationFrame(tick)
        window.setTimeout(() => finish(lastFrameTimestamp ?? performance.now()), durationMs + 500)
      })
    },
    { durationMs: MEASUREMENT_MS, frameBudgetMs: FRAME_BUDGET_MS, jankFrameMs: JANK_FRAME_MS }
  )

  const box = await canvas.boundingBox()
  if (!box) throw new Error('Expected MeshViz canvas to have a bounding box')

  const centerX = box.x + box.width / 2
  const centerY = box.y + box.height / 2
  await page.mouse.move(centerX, centerY)
  await page.mouse.down()
  await dragTo(page, centerX, centerY, centerX - 280, centerY + 90, 90, 16)
  await dragTo(page, centerX - 280, centerY + 90, centerX + 260, centerY - 70, 90, 16)
  await page.mouse.up()

  await page.mouse.move(centerX, centerY)
  for (let index = 0; index < 16; index += 1) {
    await page.mouse.wheel(0, index < 8 ? -120 : 120)
  }

  const metrics = await page.evaluate(async () => {
    if (!window.__meshVizMeasurement) throw new Error('MeshViz measurement did not start')
    return window.__meshVizMeasurement
  })

  await testInfo.attach('meshviz-200-node-performance.json', {
    body: JSON.stringify(metrics, null, 2),
    contentType: 'application/json'
  })

  console.info(`MeshViz 200-node perf: ${JSON.stringify(metrics)}`)
  expect(metrics.nodeCount).toBe(200)
  expect(metrics.durationMs).toBeGreaterThan(MEASUREMENT_MS * 0.75)
  expect(metrics.frameCount).toBeGreaterThan(0)
  expect(metrics.avgFps).toBeGreaterThanOrEqual(MIN_AVG_FPS)
  expect(metrics.p95FrameMs).toBeLessThanOrEqual(MAX_P95_FRAME_MS)
  expect(metrics.framesOver33_3ms).toBeLessThanOrEqual(Math.ceil(metrics.frameCount * MAX_JANK_FRAME_RATIO))
})

test('keeps in-flight traffic packets aligned during live panning', async ({ page }) => {
  await page.addInitScript(() => {
    Math.random = () => 0
  })

  await page.goto('/__meshviz-perf')
  await expect(page.getByRole('heading', { name: 'MeshViz 200-node benchmark' })).toBeVisible()

  const canvas = page.getByTestId('mesh-canvas')
  await expect(canvas).toBeVisible()
  await expect(page.getByTestId('mesh-node-core')).toHaveCount(200)
  await expect(page.getByTestId('mesh-node-core').first()).toHaveAttribute('data-node-lifecycle', 'present')

  const box = await canvas.boundingBox()
  if (!box) throw new Error('Expected MeshViz canvas to have a bounding box')

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  for (let index = 0; index < 4; index += 1) {
    await page.mouse.wheel(0, -120)
  }
  await expect
    .poll(async () => page.getByTestId('mesh-packet-layer').evaluate((element) => element.style.transform))
    .toContain('scale(')

  await page.keyboard.press('z')

  const packet = page.locator('.mesh-packet').first()
  await expect.poll(async () => page.locator('.mesh-packet').count(), { timeout: 10_000 }).toBeGreaterThan(0)
  await expect(packet).toBeVisible()

  const packetLayer = page.getByTestId('mesh-packet-layer')
  const initialPacketPosition = parseCssTranslate(await packet.evaluate((element) => element.style.transform))

  const panStart = await findEmptyCanvasPoint(page)

  await page.mouse.move(panStart.x, panStart.y)
  await page.mouse.down()
  await page.mouse.move(panStart.x - 220, panStart.y + 80)

  await expect.poll(async () => packetLayer.evaluate((element) => element.style.transform)).toContain('translate3d')

  const layerTransform = await packetLayer.evaluate((element) => element.style.transform)

  const liveLayerPosition = parseCssTranslate(layerTransform)
  const packetPositionDuringPan = parseCssTranslate(await packet.evaluate((element) => element.style.transform))

  await page.mouse.up()

  expect(Math.abs(liveLayerPosition.x)).toBeGreaterThan(120)
  expect(Math.abs(packetPositionDuringPan.x - initialPacketPosition.x)).toBeLessThan(120)
  expect(Math.abs(packetPositionDuringPan.y - initialPacketPosition.y)).toBeLessThan(120)
})

test('keeps glowing node visuals stable during live zoom', async ({ page }) => {
  await page.goto('/__meshviz-perf')
  await expect(page.getByRole('heading', { name: 'MeshViz 200-node benchmark' })).toBeVisible()

  const canvas = page.getByTestId('mesh-canvas')
  await expect(canvas).toBeVisible()
  await expect(page.getByTestId('mesh-node-core')).toHaveCount(200)

  const box = await canvas.boundingBox()
  if (!box) throw new Error('Expected MeshViz canvas to have a bounding box')

  const initialCoreBox = await page.getByTestId('mesh-node-core').first().boundingBox()
  if (!initialCoreBox) throw new Error('Expected a MeshViz node core bounding box')

  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
  for (let index = 0; index < 5; index += 1) {
    await page.mouse.wheel(0, -120)
  }

  await page.waitForFunction(
    () => document.querySelector<HTMLElement>('[data-testid="mesh-packet-layer"]')?.style.transform.includes('scale('),
    undefined,
    { polling: 10, timeout: 1000 }
  )

  const liveNodeVisuals = await page.evaluate(() => {
    const packetLayer = document.querySelector('[data-testid="mesh-packet-layer"]')
    const nodeCore = document.querySelector('[data-testid="mesh-node-core"]')

    if (!(packetLayer instanceof HTMLElement) || !(nodeCore instanceof HTMLElement)) {
      throw new Error('Expected MeshViz packet layer and node core elements')
    }

    const coreBox = nodeCore.getBoundingClientRect()

    return {
      layerTransform: packetLayer.style.transform,
      coreWidth: coreBox.width,
      coreHeight: coreBox.height
    }
  })

  expect(liveNodeVisuals.layerTransform).toContain('scale(')
  expect(Math.abs(liveNodeVisuals.coreWidth - initialCoreBox.width)).toBeLessThan(NODE_SIZE_DELTA_TOLERANCE_PX)
  expect(Math.abs(liveNodeVisuals.coreHeight - initialCoreBox.height)).toBeLessThan(NODE_SIZE_DELTA_TOLERANCE_PX)
})

async function dragTo(
  page: Page,
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  steps: number,
  delayMs: number
) {
  for (let step = 1; step <= steps; step += 1) {
    const progress = step / steps
    await page.mouse.move(startX + (endX - startX) * progress, startY + (endY - startY) * progress)
    if (delayMs > 0) {
      await page.waitForTimeout(delayMs)
    }
  }
}

async function findEmptyCanvasPoint(page: Page): Promise<CssTranslate> {
  return page.evaluate(() => {
    const canvasElement = document.querySelector('[data-testid="mesh-canvas"]')

    if (!(canvasElement instanceof HTMLElement)) {
      throw new Error('Expected MeshViz canvas element')
    }

    const bounds = canvasElement.getBoundingClientRect()
    for (let y = bounds.top + 24; y < bounds.bottom - 24; y += 24) {
      for (let x = bounds.left + 24; x < bounds.right - 24; x += 24) {
        const target = document.elementFromPoint(x, y)

        if (target && !target.closest('button, a, input, label')) {
          return { x, y }
        }
      }
    }

    throw new Error('Expected an empty MeshViz canvas point')
  })
}

function parseCssTranslate(transform: string): CssTranslate {
  const match = transform.match(/translate3d\((-?\d+(?:\.\d+)?)px,\s*(-?\d+(?:\.\d+)?)px,/)

  if (!match) {
    throw new Error(`Expected translate3d transform, got: ${transform}`)
  }

  const [, xValue, yValue] = match

  if (!xValue || !yValue) {
    throw new Error(`Expected x/y translate values, got: ${transform}`)
  }

  return {
    x: Number.parseFloat(xValue),
    y: Number.parseFloat(yValue)
  }
}
