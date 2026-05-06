import { createRef } from 'react'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ChatTab } from '@/features/chat/pages/ChatTab'
import { buildTOML } from '@/features/configuration/lib/build-toml'
import { ConfigurationTab } from '@/features/configuration/pages/ConfigurationTab'
import { MeshViz, type MeshVizHandle } from '@/features/network/components/MeshViz'
import { MESH_VIZ_DOT_COLOR_SCHEMES } from '@/features/network/lib/mesh-viz-dot-color-schemes'
import { DashboardPage } from '@/features/network/pages/DashboardPage'
import { buildDashboardMeshNodes } from '@/features/network/lib/dashboard-mesh-nodes'
import { APP_STORAGE_KEYS, CFG_NODES, INITIAL_ASSIGNS, MESH_NODES, PEERS } from '@/features/app-tabs/data'
import type { MeshNode, Peer } from '@/features/app-tabs/types'
import { env } from '@/lib/env'
import { FeatureFlagProvider } from '@/lib/feature-flags'

let resizeCallback: ResizeObserverCallback | undefined
let meshCanvasWidth = 800
let meshCanvasHeight = 420
let fullscreenElement: Element | null = null
const DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT = 20
const DEBUG_PLACEMENT_MIN_DISTANCE_PERCENT = 7
const DEBUG_PLACEMENT_CLUSTER_PADDING_PERCENT = 24
const DEBUG_PLACEMENT_CLUSTER_GROWTH_PERCENT = 4

class ControlledResizeObserver implements ResizeObserver {
  constructor(callback: ResizeObserverCallback) {
    resizeCallback = callback
  }

  observe(_target: Element, _options?: ResizeObserverOptions): void {}

  unobserve(_target: Element): void {}

  disconnect(): void {}
}

const controlledResizeObserver: ResizeObserver = {
  observe(): void {},
  unobserve(): void {},
  disconnect(): void {}
}

function createMatchMedia(matches: boolean) {
  return vi.fn(
    (query: string): MediaQueryList => ({
      matches,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: () => false
    })
  )
}

function setMeshCanvasSize(width: number, height: number) {
  meshCanvasWidth = width
  meshCanvasHeight = height
}

function triggerMeshResize() {
  if (resizeCallback) {
    resizeCallback([], controlledResizeObserver)
  }
}

function setFullscreenElement(element: Element | null) {
  fullscreenElement = element
}

function getMeshCanvas() {
  const canvasElement = document.querySelector('.mesh-canvas')

  if (!(canvasElement instanceof HTMLElement)) {
    throw new Error('Expected mesh canvas element')
  }

  return canvasElement
}

function getMeshElement(selector: string) {
  const element = document.querySelector(selector)

  if (!(element instanceof HTMLElement)) {
    throw new Error(`Expected ${selector} element`)
  }

  return element
}

function getMeshPackets() {
  return Array.from(document.querySelectorAll('.mesh-packet')).filter(
    (element): element is HTMLElement => element instanceof HTMLElement
  )
}

function getTomlSource() {
  return screen.getByRole('textbox', { name: /configuration toml source/i }) as HTMLTextAreaElement
}

function countTomlOccurrences(value: string) {
  return getTomlSource().value.split(value).length - 1
}

async function expectTomlOccurrences(user: ReturnType<typeof userEvent.setup>, value: string, expected: number) {
  await user.click(screen.getByRole('tab', { name: 'TOML Output' }))
  expect(countTomlOccurrences(value)).toBe(expected)
  await user.click(screen.getByRole('tab', { name: 'Model Deployment' }))
}

function getMeshLinkPairs() {
  return screen.getAllByTestId('mesh-link').map((element) => {
    if (element.tagName.toLowerCase() !== 'line') {
      throw new Error('Expected mesh link to be an SVG line')
    }

    const source = element.dataset.sourceNodeId
    const target = element.dataset.targetNodeId

    if (!source || !target) {
      throw new Error('Expected mesh link source and target data attributes')
    }

    return [source, target].sort() as [string, string]
  })
}

function linkKey(pair: [string, string]) {
  return pair.join('::')
}

function linkDegree(pairs: [string, string][], nodeId: string) {
  return pairs.filter((pair) => pair.includes(nodeId)).length
}

function nonClientLinkDegree(pairs: [string, string][], nodeId: string, clientIds: Set<string>) {
  return pairs.filter((pair) => pair.includes(nodeId) && pair.every((id) => !clientIds.has(id))).length
}

function getNodeButton(label: string) {
  const button = screen.getByRole('button', { name: label })

  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`Expected ${label} to be a button`)
  }

  return button
}

function getMeshNodeLabel(nodeId: string) {
  const label = screen
    .getAllByTestId('mesh-node-label')
    .find((element) => element.getAttribute('data-node-id') === nodeId)

  if (!(label instanceof HTMLElement)) {
    throw new Error(`Expected mesh node label for ${nodeId}`)
  }

  return label
}

function getMeshNodeContextHighlight(nodeId: string) {
  const highlight = screen
    .getAllByTestId('mesh-node-context-highlight')
    .find((element) => element.getAttribute('data-node-id') === nodeId)

  if (!(highlight instanceof HTMLElement)) {
    throw new Error(`Expected mesh node context highlight for ${nodeId}`)
  }

  return highlight
}

function getMeshNodeCore(nodeId: string) {
  const core = screen
    .getAllByTestId('mesh-node-core')
    .find((element) => element.getAttribute('data-node-id') === nodeId)

  if (!(core instanceof HTMLElement)) {
    throw new Error(`Expected mesh node core for ${nodeId}`)
  }

  return core
}

function expectMeshNodeCoreFill(nodeId: string, color: string, mixPercent: '14%' | '18%') {
  const core = getMeshNodeCore(nodeId)

  expect(core.style.color).toBe(color)
  expect(core.style.backgroundColor).toBe(`color-mix(in oklab, currentColor ${mixPercent}, var(--color-panel-strong))`)
}

function getMeshNodeCoreOverlay(nodeId: string) {
  const overlay = screen
    .getAllByTestId('mesh-node-core-overlay')
    .find((element) => element.getAttribute('data-node-id') === nodeId)

  if (!(overlay instanceof HTMLElement)) {
    throw new Error(`Expected mesh node core overlay for ${nodeId}`)
  }

  return overlay
}

async function openDebugMenu(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: /^debug$/i }))
}

async function openTrafficDebugMenu(user: ReturnType<typeof userEvent.setup>) {
  await openDebugMenu(user)
}

async function openAddDebugNodesMenu(user: ReturnType<typeof userEvent.setup>) {
  await openDebugMenu(user)
  const addNodesTrigger = screen.getByRole('button', { name: /^add nodes$/i })

  fireEvent.pointerEnter(addNodesTrigger)
  await screen.findByRole('menuitem', { name: /debug client/i })
}

async function openRemoveDebugNodesMenu(user: ReturnType<typeof userEvent.setup>) {
  await openDebugMenu(user)
  const removeNodesTrigger = screen.getByRole('button', { name: /^remove nodes$/i })

  fireEvent.pointerEnter(removeNodesTrigger)
  await screen.findByRole('menuitem', { name: /debug client/i })
}

function pixelValue(value: string) {
  return Number.parseFloat(value.replace('px', ''))
}

function debugNodeCoordinates() {
  return screen.getAllByRole('button', { name: /view debug-/i }).map((button) => {
    const x = Number(button.getAttribute('data-node-x'))
    const y = Number(button.getAttribute('data-node-y'))

    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      throw new Error('Expected debug node to expose deterministic coordinates')
    }

    return { x, y }
  })
}

function nearestNodeDistance(point: Pick<MeshNode, 'x' | 'y'>, nodes: Array<Pick<MeshNode, 'x' | 'y'>>) {
  return Math.min(...nodes.map((node) => Math.hypot(point.x - node.x, point.y - node.y)))
}

function placementCentroid(nodes: Array<Pick<MeshNode, 'x' | 'y'>>) {
  return {
    x: nodes.reduce((sum, node) => sum + node.x, 0) / nodes.length,
    y: nodes.reduce((sum, node) => sum + node.y, 0) / nodes.length
  }
}

function distanceFrom(point: Pick<MeshNode, 'x' | 'y'>, origin: Pick<MeshNode, 'x' | 'y'>) {
  return Math.hypot(point.x - origin.x, point.y - origin.y)
}

function placementClusterRadius(nodes: Array<Pick<MeshNode, 'x' | 'y'>>, debugCount: number) {
  const centroid = placementCentroid(nodes)
  const baseRadius = Math.max(...nodes.map((node) => distanceFrom(node, centroid)))

  return Math.max(
    DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT * 2,
    baseRadius +
      DEBUG_PLACEMENT_CLUSTER_PADDING_PERCENT +
      Math.sqrt(debugCount + 1) * DEBUG_PLACEMENT_CLUSTER_GROWTH_PERCENT
  )
}

const OVERSIZED_MESH_NODES = [
  ...MESH_NODES,
  { id: 'north-edge', label: 'NORTH EDGE', subLabel: 'TEST EDGE', x: 6, y: 2, status: 'online' as const },
  { id: 'south-edge', label: 'SOUTH EDGE', subLabel: 'TEST EDGE', x: 94, y: 98, status: 'online' as const }
]
const EXPANDED_BOUNDARY_MESH_NODES = [
  ...MESH_NODES,
  { id: 'west-boundary', label: 'WEST EDGE', subLabel: 'TEST EDGE', x: -40, y: 50, status: 'online' as const },
  { id: 'east-boundary', label: 'EAST EDGE', subLabel: 'TEST EDGE', x: 180, y: 50, status: 'online' as const }
]
const HUGE_BOUNDARY_MESH_NODES = [
  ...MESH_NODES,
  { id: 'far-west-boundary', label: 'FAR WEST', subLabel: 'TEST EDGE', x: -400, y: 50, status: 'online' as const },
  { id: 'far-east-boundary', label: 'FAR EAST', subLabel: 'TEST EDGE', x: 700, y: 50, status: 'online' as const }
]

beforeEach(() => {
  env.isDevelopment = true
  resizeCallback = undefined
  setMeshCanvasSize(800, 420)
  globalThis.ResizeObserver = ControlledResizeObserver
  window.matchMedia = createMatchMedia(false)

  Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
    configurable: true,
    get() {
      return this.classList.contains('mesh-canvas') ? meshCanvasWidth : 0
    }
  })
  Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
    configurable: true,
    get() {
      return this.classList.contains('mesh-canvas') ? meshCanvasHeight : 0
    }
  })

  HTMLElement.prototype.setPointerCapture = vi.fn()
  HTMLElement.prototype.releasePointerCapture = vi.fn()
  HTMLElement.prototype.hasPointerCapture = vi.fn(() => true)
  HTMLElement.prototype.requestFullscreen = vi.fn(() => Promise.resolve())
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined })
  document.documentElement.removeAttribute('data-theme')
  document.body.style.userSelect = ''
  window.localStorage.removeItem(APP_STORAGE_KEYS.featureFlagOverrides)
  setFullscreenElement(null)
  Object.defineProperty(document, 'fullscreenElement', {
    configurable: true,
    get: () => fullscreenElement
  })
})

function createMockDataTransfer() {
  const data = new Map<string, string>()

  return {
    dropEffect: 'none',
    effectAllowed: 'all',
    get types() {
      return Array.from(data.keys())
    },
    getData: vi.fn((type: string) => data.get(type) ?? ''),
    setData: vi.fn((type: string, value: string) => {
      data.set(type, value)
    }),
    setDragImage: vi.fn()
  }
}

describe('app surfaces', () => {
  it('renders the network component composition', () => {
    render(<DashboardPage />)
    expect(screen.getByRole('heading', { name: /your private mesh/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /model catalog/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /connected peers/i })).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /view carrack node/i }).length).toBeGreaterThan(0)
  })

  it('copies the dashboard connect command to the clipboard', async () => {
    const user = userEvent.setup()
    const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })

    render(<DashboardPage />)

    await user.click(screen.getByRole('button', { name: 'Copy' }))

    expect(writeText).toHaveBeenCalledWith('mesh-llm --auto --join <mesh-invite-token>')
    await waitFor(() => expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument())
  })

  it('places newly joined dashboard peers with the shared clustered mesh rule', () => {
    const joinedPeer: Peer = {
      id: 'p4',
      hostname: 'new-worker',
      region: 'iad-1',
      status: 'online',
      hostedModels: [],
      sharePct: 12,
      latencyMs: 2.4,
      loadPct: 18,
      role: 'peer',
      version: '0.64.0',
      vramGB: 24,
      toksPerSec: 11.2
    }
    const meshNodes = buildDashboardMeshNodes([...PEERS, joinedPeer], 'joined-peer-placement-test')
    const repeatedMeshNodes = buildDashboardMeshNodes([...PEERS, joinedPeer], 'joined-peer-placement-test')
    const joinedNode = meshNodes.find((node) => node.peerId === joinedPeer.id)
    const repeatedJoinedNode = repeatedMeshNodes.find((node) => node.peerId === joinedPeer.id)
    const baseCentroid = placementCentroid(MESH_NODES)

    expect(joinedNode).toBeDefined()
    expect(repeatedJoinedNode).toBeDefined()
    expect(joinedNode?.x).not.toBe(0)
    expect(joinedNode?.y).not.toBe(0)
    expect(joinedNode?.renderKind).toBe('worker')
    expect(joinedNode?.meshState).toBe('standby')
    expect(joinedNode?.vramGB).toBe(24)

    for (const pinnedNode of MESH_NODES) {
      const generatedNode = meshNodes.find((node) => node.peerId === pinnedNode.peerId || node.id === pinnedNode.id)

      expect(generatedNode?.x).toBe(pinnedNode.x)
      expect(generatedNode?.y).toBe(pinnedNode.y)
    }

    expect(nearestNodeDistance(joinedNode as MeshNode, MESH_NODES)).toBeLessThanOrEqual(
      DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT + 0.01
    )
    expect(distanceFrom(joinedNode as MeshNode, baseCentroid)).toBeLessThanOrEqual(
      placementClusterRadius(MESH_NODES, 0) + 0.01
    )
    expect(repeatedJoinedNode?.x).toBe(joinedNode?.x)
    expect(repeatedJoinedNode?.y).toBe(joinedNode?.y)
  })

  it('opens dashboard drawers from lists and MeshViz node popovers from mesh clicks', async () => {
    const user = userEvent.setup()
    render(<DashboardPage />)

    const gemmaModelRow = screen.getByRole('button', { name: /view gemma-4-26b-a4b-it-ud model/i })
    await user.click(gemmaModelRow)
    expect(gemmaModelRow).toHaveAttribute('data-active', 'true')
    let drawer = screen.getByRole('dialog')
    expect(within(drawer).getByText(/availability/i)).toBeInTheDocument()
    expect(within(drawer).getAllByText(/64k/i).length).toBeGreaterThan(0)
    expect(within(drawer).getByText(/files/i)).toBeInTheDocument()
    expect(within(drawer).getByText(/active peers/i)).toBeInTheDocument()

    await user.click(within(drawer).getByRole('button', { name: /close/i }))
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    expect(gemmaModelRow).not.toHaveAttribute('data-active')
    const carrackNodeButton = screen.getByRole('button', { name: 'View CARRACK node' })
    await user.hover(carrackNodeButton)
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    expect(carrackNodeButton).not.toHaveAttribute('data-context-open')
    expect(getMeshNodeContextHighlight('self')).toHaveClass('opacity-0')
    const selfCoreFill = getMeshNodeCore('self').style.color
    expect(getMeshNodeCoreOverlay('self')).toHaveClass('opacity-0')

    await user.click(carrackNodeButton)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(carrackNodeButton).toHaveAttribute('data-context-open', 'true')
    expect(getMeshNodeContextHighlight('self')).toHaveClass('opacity-100', 'duration-150')
    expect(getMeshNodeContextHighlight('self').style.background).toContain('currentcolor')
    expect(getMeshNodeContextHighlight('self').style.background).not.toContain('--color-accent')
    expect(getMeshNodeCore('self').style.color).toBe(selfCoreFill)
    const radarPing = getMeshElement('.mesh-radar-ping')
    expect(radarPing.style.color).toContain('oklch(')
    expect(radarPing.style.color).not.toContain('--color-accent')
    expect(getMeshNodeCoreOverlay('self')).toHaveClass('opacity-45', 'duration-150')
    expect(getMeshNodeCoreOverlay('self').style.backgroundColor).toContain('currentcolor')
    expect(getMeshNodeCoreOverlay('self').style.backgroundColor).not.toContain('--color-accent')

    let popover = await screen.findByRole('tooltip')
    expect(within(popover).getByText(/CARRACK/i)).toBeInTheDocument()
    expect(within(popover).getByText(/990232e1c1/i)).toBeInTheDocument()
    expect(within(popover).getByText(/VRAM/i)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'View LEMONY-28 node' }))
    popover = await screen.findByRole('tooltip')
    expect(getMeshNodeContextHighlight('self')).toHaveClass('opacity-0')
    expect(getMeshNodeContextHighlight('lemony')).toHaveClass('opacity-100', 'duration-150')
    const lemonyCoreFill = getMeshNodeCore('lemony').style.color
    expect(getMeshNodeCore('self').style.color).toBe(selfCoreFill)
    expect(getMeshNodeCore('lemony').style.color).toBe(lemonyCoreFill)
    expect(getMeshNodeCoreOverlay('self')).toHaveClass('opacity-0')
    expect(getMeshNodeCoreOverlay('lemony')).toHaveClass('opacity-45', 'duration-150')
    expect(within(popover).getByText(/lemony-28/i)).toBeInTheDocument()
    expect(within(popover).getByText(/e5c42cc0ad/i)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'View lemony-28 node' }))
    drawer = screen.getByRole('dialog')
    expect(within(drawer).getByText(/node metadata/i)).toBeInTheDocument()
    expect(within(drawer).getByText(/hosted models/i)).toBeInTheDocument()
    expect(within(drawer).getByText(/hardware/i)).toBeInTheDocument()
    expect(within(drawer).getByRole('heading', { name: /ownership/i })).toBeInTheDocument()
    expect(within(drawer).getByText(/gemma-4-26B-A4B-it-UD/i)).toBeInTheDocument()
  })

  it('keeps MeshViz viewport interactions stable across wheel, pan, resize, and reset', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))
    const initialLeft = pixelValue(lemony29.style.left)
    const initialTop = pixelValue(lemony29.style.top)

    fireEvent.wheel(canvas, { deltaY: 0, clientX: 240, clientY: 210 })
    expect(pixelValue(lemony29.style.left)).toBeCloseTo(initialLeft, 4)
    expect(pixelValue(lemony29.style.top)).toBeCloseTo(initialTop, 4)

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 150, clientY: 130 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeCloseTo(initialLeft + 50, 1))
    expect(pixelValue(lemony29.style.top)).toBeGreaterThan(initialTop + 20)
    expect(pixelValue(lemony29.style.top)).toBeLessThanOrEqual(initialTop + 30)

    setMeshCanvasSize(801, 420)
    triggerMeshResize()
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(initialLeft + 40))

    await userEvent.click(screen.getByRole('button', { name: /reset view/i }))
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeLessThan(initialLeft + 10))
  })

  it('pinch-zooms MeshViz on touch pointers', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))
    const initialLeft = pixelValue(lemony29.style.left)

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, pointerType: 'touch', clientX: 350, clientY: 210 })
    fireEvent.pointerDown(canvas, { button: 0, pointerId: 2, pointerType: 'touch', clientX: 450, clientY: 210 })
    fireEvent.pointerMove(canvas, { pointerId: 2, pointerType: 'touch', clientX: 550, clientY: 210 })
    fireEvent.pointerUp(canvas, { pointerId: 2, pointerType: 'touch' })
    fireEvent.pointerUp(canvas, { pointerId: 1, pointerType: 'touch' })

    await waitFor(() => expect(Math.abs(pixelValue(lemony29.style.left) - initialLeft)).toBeGreaterThan(20))
  })

  it('connects clients to their closest non-client without consuming non-client link capacity', () => {
    const sparseNodes: MeshNode[] = [
      { id: 'client-a', label: 'CLIENT A', x: 0, y: 0, status: 'online', renderKind: 'client', client: true },
      { id: 'client-b', label: 'CLIENT B', x: 0, y: 20, status: 'online', renderKind: 'client', client: true },
      { id: 'host', label: 'HOST', x: 50, y: 50, status: 'online', host: true, renderKind: 'worker' },
      { id: 'worker-1', label: 'WORKER 1', x: 40, y: 50, status: 'online', renderKind: 'worker' },
      { id: 'worker-2', label: 'WORKER 2', x: 60, y: 50, status: 'online', renderKind: 'worker' },
      { id: 'worker-3', label: 'WORKER 3', x: 50, y: 40, status: 'online', renderKind: 'worker' },
      { id: 'worker-4', label: 'WORKER 4', x: 50, y: 60, status: 'online', renderKind: 'worker' },
      { id: 'worker-5', label: 'WORKER 5', x: 90, y: 90, status: 'online', renderKind: 'worker' }
    ]

    render(<MeshViz nodes={sparseNodes} selfId="host" height={420} />)

    const pairs = getMeshLinkPairs()
    const pairKeys = pairs.map(linkKey)
    const clientIds = new Set(['client-a', 'client-b'])

    expect(pairKeys).toContain('client-a::worker-1')
    expect(pairKeys).toContain('client-b::worker-1')
    expect(linkDegree(pairs, 'client-a')).toBe(1)
    expect(linkDegree(pairs, 'client-b')).toBe(1)
    expect(pairKeys).not.toContain('client-a::client-b')
    expect(linkDegree(pairs, 'worker-1')).toBeGreaterThan(3)

    for (const node of sparseNodes.filter((meshNode) => !meshNode.client)) {
      expect(nonClientLinkDegree(pairs, node.id, clientIds)).toBeLessThanOrEqual(3)
    }
  })

  it('connects a host to its three closest available nodes', () => {
    const nearestNodes: MeshNode[] = [
      { id: 'host', label: 'HOST', x: 50, y: 50, status: 'online', host: true, renderKind: 'worker' },
      { id: 'worker-1', label: 'WORKER 1', x: 40, y: 50, status: 'online', renderKind: 'worker' },
      { id: 'worker-2', label: 'WORKER 2', x: 60, y: 50, status: 'online', renderKind: 'worker' },
      { id: 'worker-3', label: 'WORKER 3', x: 50, y: 40, status: 'online', renderKind: 'worker' },
      { id: 'worker-4', label: 'WORKER 4', x: 50, y: 60, status: 'online', renderKind: 'worker' },
      { id: 'worker-5', label: 'WORKER 5', x: 90, y: 90, status: 'online', renderKind: 'worker' }
    ]

    render(<MeshViz nodes={nearestNodes} selfId="host" height={420} />)

    const pairs = getMeshLinkPairs()
    const pairKeys = pairs.map(linkKey)

    expect(linkDegree(pairs, 'host')).toBe(3)
    expect(pairKeys).toContain('host::worker-1')
    expect(pairKeys).toContain('host::worker-2')
    expect(pairKeys).toContain('host::worker-3')
    expect(pairKeys).not.toContain('host::worker-4')
    expect(pairKeys).not.toContain('host::worker-5')
  })

  it('keeps host and worker backbone links when nearby clients attach to the same node', () => {
    const crowdedClientNodes: MeshNode[] = [
      { id: 'host', label: 'HOST', x: 50, y: 50, status: 'online', host: true, renderKind: 'worker' },
      { id: 'client-1', label: 'CLIENT 1', x: 48, y: 49, status: 'online', renderKind: 'client', client: true },
      { id: 'client-2', label: 'CLIENT 2', x: 49, y: 48, status: 'online', renderKind: 'client', client: true },
      { id: 'client-3', label: 'CLIENT 3', x: 51, y: 52, status: 'online', renderKind: 'client', client: true },
      { id: 'worker-1', label: 'WORKER 1', x: 46, y: 50, status: 'online', renderKind: 'worker' },
      { id: 'worker-2', label: 'WORKER 2', x: 54, y: 50, status: 'online', renderKind: 'worker' },
      { id: 'worker-3', label: 'WORKER 3', x: 50, y: 46, status: 'online', renderKind: 'worker' },
      { id: 'worker-4', label: 'WORKER 4', x: 80, y: 80, status: 'online', renderKind: 'worker' }
    ]

    render(<MeshViz nodes={crowdedClientNodes} selfId="host" height={420} />)

    const pairs = getMeshLinkPairs()
    const pairKeys = pairs.map(linkKey)
    const clientIds = new Set(['client-1', 'client-2', 'client-3'])

    expect(pairKeys).toContain('client-1::host')
    expect(pairKeys).toContain('client-2::host')
    expect(pairKeys).toContain('client-3::host')
    expect(pairKeys).toContain('host::worker-1')
    expect(pairKeys).toContain('host::worker-2')
    expect(pairKeys).toContain('host::worker-3')
    expect(linkDegree(pairs, 'host')).toBeGreaterThan(3)
    expect(nonClientLinkDegree(pairs, 'host', clientIds)).toBe(3)
  })

  it('prevents page text selection while panning MeshViz', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()

    await waitFor(() => expect(getNodeButton('View LEMONY-29 node').style.left).not.toBe(''))

    const pointerDownWasNotCanceled = fireEvent.pointerDown(canvas, {
      button: 0,
      pointerId: 1,
      clientX: 100,
      clientY: 100,
      cancelable: true
    })

    expect(pointerDownWasNotCanceled).toBe(false)
    await waitFor(() => expect(document.body.style.userSelect).toBe('none'))
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 130, clientY: 120, cancelable: true })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(document.body.style.userSelect).toBe(''))
  })

  it('limits MeshViz pan travel to the node bounds plus the dead-zone', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))
    const initialLeft = pixelValue(lemony29.style.left)

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 2100, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeLessThan(initialLeft + 160))

    await userEvent.click(screen.getByRole('button', { name: /reset view/i }))
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeCloseTo(initialLeft, 1))

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 2, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 2, clientX: -1900, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 2 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(initialLeft - 160))
  })

  it('allows MeshViz bounds to reach fullscreen edges while panning', async () => {
    setMeshCanvasSize(1600, 900)
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={900} />)
    const canvas = getMeshCanvas()
    const carrack = getNodeButton('View CARRACK node')
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(carrack.style.left)).toBeGreaterThan(0))

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 2100, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(pixelValue(carrack.style.left)).toBeGreaterThan(1450))
    expect(pixelValue(carrack.style.left)).toBeLessThan(1530)

    await userEvent.click(screen.getByRole('button', { name: /reset view/i }))

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 2, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 2, clientX: -1900, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 2 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeLessThan(130))
    expect(pixelValue(lemony29.style.left)).toBeGreaterThan(70)
  })

  it('lowers the MeshViz max zoom-out limit as the graph boundary grows', async () => {
    render(<MeshViz nodes={EXPANDED_BOUNDARY_MESH_NODES} selfId="self" height={420} />)
    const westEdge = getNodeButton('View WEST EDGE node')
    const eastEdge = getNodeButton('View EAST EDGE node')

    await waitFor(() => {
      expect(screen.getByTestId('mesh-max-zoom-label')).toHaveTextContent('Max Zoom: 0.40')
      expect(pixelValue(westEdge.style.left)).toBeGreaterThanOrEqual(48)
      expect(pixelValue(eastEdge.style.left)).toBeLessThanOrEqual(752)
    })
  })

  it('keeps deriving the MeshViz max zoom-out limit for very large graph boundaries', async () => {
    render(<MeshViz nodes={HUGE_BOUNDARY_MESH_NODES} selfId="self" height={420} />)

    await waitFor(() => expect(screen.getByTestId('mesh-max-zoom-label')).toHaveTextContent('Max Zoom: 0.08'))
  })

  it('only shows the MeshViz max zoom label in debug mode', () => {
    env.isDevelopment = false

    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    expect(screen.queryByTestId('mesh-max-zoom-label')).not.toBeInTheDocument()
  })

  it('reclamps oversized MeshViz bounds to preserve viewport intersection when the canvas shrinks', async () => {
    const user = userEvent.setup()
    setMeshCanvasSize(1600, 900)
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={900} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await openTrafficDebugMenu(user)
    await user.click(screen.getByRole('button', { name: /debug boundaries/i }))
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: -1900, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeLessThan(130))

    await act(async () => {
      setMeshCanvasSize(800, 420)
      triggerMeshResize()
    })

    await waitFor(() => {
      const nodeBoundsBox = screen.getByTestId('mesh-node-bounds-box')
      const boundsX = Number(nodeBoundsBox.getAttribute('x'))
      const boundsY = Number(nodeBoundsBox.getAttribute('y'))
      const boundsWidth = Number(nodeBoundsBox.getAttribute('width'))
      const boundsHeight = Number(nodeBoundsBox.getAttribute('height'))

      expect(boundsX).toBeLessThanOrEqual(800)
      expect(boundsX + boundsWidth).toBeGreaterThanOrEqual(0)
      expect(boundsY).toBeLessThanOrEqual(420)
      expect(boundsY + boundsHeight).toBeGreaterThanOrEqual(0)
    })
  })

  it('resets MeshViz to a centered fit when exiting fullscreen', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))
    const initialLeft = pixelValue(lemony29.style.left)

    await act(async () => {
      setFullscreenElement(canvas)
      setMeshCanvasSize(1600, 900)
      triggerMeshResize()
      fireEvent(document, new Event('fullscreenchange'))
    })

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: -1900, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeLessThan(140))

    await act(async () => {
      setMeshCanvasSize(800, 420)
      triggerMeshResize()
      setFullscreenElement(null)
      fireEvent(document, new Event('fullscreenchange'))
    })

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeCloseTo(initialLeft, 1))
  })

  it('lets oversized MeshViz bounds pan until the inner bounds still touch the viewport', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={OVERSIZED_MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony = getNodeButton('View LEMONY-28 node')

    await openTrafficDebugMenu(user)
    await user.click(screen.getByRole('button', { name: /debug boundaries/i }))
    await waitFor(() => expect(pixelValue(lemony.style.top)).toBeGreaterThan(0))

    for (let index = 0; index < 12; index += 1) {
      await user.click(screen.getByRole('button', { name: /zoom in/i }))
    }

    await waitFor(() => expect(pixelValue(lemony.style.top)).toBeLessThan(0))
    const centeredNodeBoundsBox = screen.getByTestId('mesh-centered-bounds-box')
    const centeredBoundsY = Number(centeredNodeBoundsBox.getAttribute('y'))
    const centeredBoundsHeight = Number(centeredNodeBoundsBox.getAttribute('height'))

    expect(centeredBoundsHeight).toBeGreaterThan(420)

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 100, clientY: 2100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })

    await waitFor(() => {
      const nodeBoundsBox = screen.getByTestId('mesh-centered-bounds-box')
      const boundsY = Number(nodeBoundsBox.getAttribute('y'))
      const boundsHeight = Number(nodeBoundsBox.getAttribute('height'))

      expect(boundsY).toBeGreaterThan(centeredBoundsY + 100)
      expect(boundsY).toBeLessThanOrEqual(420)
      expect(boundsY + boundsHeight).toBeGreaterThanOrEqual(0)
    })
  })

  it('lets oversized MeshViz bounds pan while preserving a viewport intersection', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={OVERSIZED_MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony = getNodeButton('View LEMONY-28 node')

    await openTrafficDebugMenu(user)
    await user.click(screen.getByRole('button', { name: /debug boundaries/i }))
    await waitFor(() => expect(pixelValue(lemony.style.top)).toBeGreaterThan(0))
    const focusX = pixelValue(lemony.style.left)
    const focusY = pixelValue(lemony.style.top) + 18

    for (let index = 0; index < 12; index += 1) {
      fireEvent.wheel(canvas, { deltaY: -100, clientX: focusX, clientY: focusY, cancelable: true })
    }

    await waitFor(() => expect(pixelValue(lemony.style.top) + 18).toBeGreaterThanOrEqual(0))
    expect(pixelValue(lemony.style.top) + 18).toBeLessThanOrEqual(420)

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 100, clientY: -2100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })

    await waitFor(() => {
      const nodeBoundsBox = screen.getByTestId('mesh-centered-bounds-box')
      const boundsY = Number(nodeBoundsBox.getAttribute('y'))
      const boundsHeight = Number(nodeBoundsBox.getAttribute('height'))

      expect(boundsY).toBeLessThanOrEqual(420)
      expect(boundsY + boundsHeight).toBeGreaterThanOrEqual(-0.5)
    })
  })

  it('drops stale zoom focus when the focused edge node leaves the mesh', async () => {
    const user = userEvent.setup()
    const nodesWithoutLemony = MESH_NODES.filter((node) => node.id !== 'lemony')
    const { rerender } = render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony = getNodeButton('View LEMONY-28 node')

    await openTrafficDebugMenu(user)
    await user.click(screen.getByRole('button', { name: /debug boundaries/i }))
    await waitFor(() => expect(pixelValue(lemony.style.top)).toBeGreaterThan(0))
    const focusX = pixelValue(lemony.style.left)
    const focusY = pixelValue(lemony.style.top) + 18

    for (let index = 0; index < 12; index += 1) {
      fireEvent.wheel(canvas, { deltaY: -100, clientX: focusX, clientY: focusY, cancelable: true })
    }

    rerender(<MeshViz nodes={nodesWithoutLemony} selfId="self" height={420} />)

    await waitFor(() => {
      const nodeBoundsBox = screen.getByTestId('mesh-node-bounds-box')
      const boundsY = Number(nodeBoundsBox.getAttribute('y'))
      const boundsHeight = Number(nodeBoundsBox.getAttribute('height'))

      expect(boundsY).toBeGreaterThanOrEqual(0)
      expect(boundsY + boundsHeight).toBeLessThanOrEqual(420)
    })
  })

  it('lets zoomed MeshViz centered bounds pan naturally until the hard viewport limit', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={OVERSIZED_MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()

    await openTrafficDebugMenu(user)
    await user.click(screen.getByRole('button', { name: /debug boundaries/i }))

    for (let index = 0; index < 12; index += 1) {
      await user.click(screen.getByRole('button', { name: /zoom in/i }))
    }

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 100, clientY: 2100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })

    const nodeBoundsBox = screen.getByTestId('mesh-centered-bounds-box')
    const boundsY = Number(nodeBoundsBox.getAttribute('y'))
    const boundsHeight = Number(nodeBoundsBox.getAttribute('height'))

    expect(boundsHeight).toBeGreaterThan(420)
    expect(boundsY).toBeLessThanOrEqual(420)
    expect(boundsY).toBeGreaterThan(410)
    expect(boundsY + boundsHeight).toBeGreaterThanOrEqual(0)
  })

  it('transitions MeshViz to recalculated bounds when an edge node is removed', async () => {
    const edgeNodes = [
      ...MESH_NODES,
      { ...MESH_NODES[1], id: 'edge-node', peerId: 'edge-peer', label: 'EDGE', x: 94, y: 76 }
    ]
    const { rerender } = render(<MeshViz nodes={edgeNodes} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: -1900, clientY: 100 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeLessThan(80))

    rerender(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    const exitingEdge = getNodeButton('View EDGE node')
    expect(exitingEdge).toBeInTheDocument()
    expect(exitingEdge).toHaveAttribute('data-node-lifecycle', 'leaving')
    expect(pixelValue(lemony29.style.left)).toBeLessThan(90)
    expect(pixelValue(lemony29.style.left)).toBeLessThan(90)
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /view edge node/i })).not.toBeInTheDocument()
    })
    expect(pixelValue(lemony29.style.left)).toBeLessThan(200)
  })

  it('toggles a development overlay for MeshViz bounds and dead-zone visualization', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    await openTrafficDebugMenu(user)
    expect(screen.getByText('Visuals')).toBeInTheDocument()
    const boundsToggle = screen.getByRole('button', { name: /debug boundaries/i })
    expect(boundsToggle).toHaveAttribute('aria-pressed', 'false')
    expect(screen.queryByTestId('mesh-node-bounds-box')).not.toBeInTheDocument()
    expect(screen.queryByTestId('mesh-pan-dead-zone-box')).not.toBeInTheDocument()
    expect(screen.queryByTestId('mesh-centered-bounds-box')).not.toBeInTheDocument()

    await user.click(boundsToggle)

    const nodeBoundsBox = screen.getByTestId('mesh-node-bounds-box')
    const centeredBoundsBox = screen.getByTestId('mesh-centered-bounds-box')

    expect(nodeBoundsBox).toBeInTheDocument()
    expect(screen.getByTestId('mesh-pan-dead-zone-box')).toBeInTheDocument()
    expect(centeredBoundsBox).toBeInTheDocument()
    expect(nodeBoundsBox).toHaveAttribute('stroke', 'color-mix(in oklab, var(--color-good) 78%, transparent)')
    expect(centeredBoundsBox).toHaveAttribute('stroke', 'color-mix(in oklab, var(--color-warn) 82%, transparent)')
    expect(centeredBoundsBox).toHaveAttribute('stroke-dasharray', '6 5')

    const nodeBoundsX = Number(nodeBoundsBox.getAttribute('x'))
    const nodeBoundsY = Number(nodeBoundsBox.getAttribute('y'))
    const nodeBoundsWidth = Number(nodeBoundsBox.getAttribute('width'))
    const nodeBoundsHeight = Number(nodeBoundsBox.getAttribute('height'))
    const centeredBoundsX = Number(centeredBoundsBox.getAttribute('x'))
    const centeredBoundsY = Number(centeredBoundsBox.getAttribute('y'))
    const centeredBoundsWidth = Number(centeredBoundsBox.getAttribute('width'))
    const centeredBoundsHeight = Number(centeredBoundsBox.getAttribute('height'))

    expect(centeredBoundsWidth).toBeCloseTo(nodeBoundsWidth / 2)
    expect(centeredBoundsHeight).toBeCloseTo(nodeBoundsHeight / 2)
    expect(centeredBoundsX).toBeCloseTo(nodeBoundsX + nodeBoundsWidth / 4)
    expect(centeredBoundsY).toBeCloseTo(nodeBoundsY + nodeBoundsHeight / 4)

    await openTrafficDebugMenu(user)
    expect(screen.getByRole('button', { name: /debug boundaries/i })).toHaveAttribute('aria-pressed', 'true')
  })

  it('toggles MeshViz between line and dot grid styles from the debug visuals menu', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    expectMeshNodeCoreFill('self', 'oklch(0.66 0.22 28)', '18%')
    expect(screen.getByTestId('mesh-viz-line-grid')).toHaveAttribute(
      'stroke',
      'color-mix(in oklab, var(--color-foreground) 7.2%, transparent)'
    )
    expect(screen.queryByTestId('mesh-viz-dot-grid')).not.toBeInTheDocument()
    expect(screen.queryByTestId('mesh-viz-accent-dot-grid')).not.toBeInTheDocument()

    await openDebugMenu(user)
    const gridStyleToggle = screen.getByRole('button', { name: /toggle grid style \(lines\)/i })
    expect(gridStyleToggle).toHaveAttribute('aria-pressed', 'false')
    expect(gridStyleToggle).toHaveAttribute('aria-keyshortcuts', 'Control+G')
    expect(gridStyleToggle).toHaveTextContent('Ctrl+G')

    await user.click(gridStyleToggle)

    expect(screen.queryByTestId('mesh-viz-line-grid')).not.toBeInTheDocument()
    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.64 0.025 252 / 9%)')
    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('cx', '0')
    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('cy', '0')
    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('r', '1.35')
    const accentDot = screen.getByTestId('mesh-viz-accent-dot-grid')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.72 0.115 220 / 13%)')
    expect(Number(accentDot.getAttribute('cx'))).toBeGreaterThan(0)
    expect(accentDot.getAttribute('cx')).toBe(accentDot.getAttribute('cy'))
    expect(accentDot).toHaveAttribute('r', '1.25')
    const tertiaryDot = screen.getByTestId('mesh-viz-tertiary-dot-grid')
    expect(tertiaryDot).toHaveAttribute('fill', 'oklch(0.76 0.105 72 / 7%)')
    expect(tertiaryDot).toHaveAttribute('cx', '0')
    expect(tertiaryDot.getAttribute('cy')).toBe(accentDot.getAttribute('cy'))
    expect(tertiaryDot).toHaveAttribute('r', '0.85')

    await openDebugMenu(user)
    const activeGridStyleToggle = screen.getByRole('button', { name: /toggle grid style \(dots\)/i })
    expect(activeGridStyleToggle).toHaveAttribute('aria-pressed', 'true')
    expect(activeGridStyleToggle).toHaveAttribute('aria-keyshortcuts', 'Control+G')
    const dotThemeOptions = screen.getByRole('group', { name: /dot theme options/i })
    expect(dotThemeOptions).toHaveAttribute('aria-keyshortcuts', 'Control+C')
    const dotThemeLabel = screen.getByTestId('mesh-viz-dot-theme-label')
    expect(dotThemeLabel).toHaveTextContent('Dot Theme')
    expect(dotThemeLabel).toHaveClass('text-[length:var(--density-type-caption)]', 'text-foreground')
    expect(dotThemeLabel).not.toHaveClass('font-mono')
    expect(screen.getByRole('button', { name: /cycle dot theme/i })).toHaveTextContent('Ctrl+C')
    const ashSignalSwatch = screen.getByRole('button', { name: /dot theme 1: ash signal/i })
    const coolTraceSwatch = screen.getByRole('button', { name: /dot theme 2: cool trace/i })
    const warmTraceSwatch = screen.getByRole('button', { name: /dot theme 3: warm trace/i })

    expect(ashSignalSwatch).toHaveAttribute('aria-pressed', 'true')
    expect(coolTraceSwatch).toHaveAttribute('aria-pressed', 'false')
    expect(warmTraceSwatch).toHaveAttribute('aria-pressed', 'false')
    expect(ashSignalSwatch).toHaveClass('border-transparent', 'bg-panel-strong/45')
    expect(coolTraceSwatch).toHaveClass('border-transparent')
    expect(coolTraceSwatch).not.toHaveClass('border-foreground')
    expect(ashSignalSwatch.firstElementChild).toBe(screen.getByTestId('mesh-viz-dot-theme-1-index'))
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-1')).toHaveAttribute(
      'data-color-value',
      'oklch(0.64 0.025 252)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-1')).toHaveClass('opacity-100')
    expect(screen.getByTestId('mesh-viz-dot-theme-2-color-1')).toHaveClass('opacity-45')
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-2')).toHaveAttribute(
      'data-color-value',
      'oklch(0.72 0.115 220)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-3')).toHaveAttribute(
      'data-color-value',
      'oklch(0.76 0.105 72)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-4')).toHaveAttribute(
      'data-color-value',
      'oklch(0.66 0.22 28)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-index')).toHaveClass('text-foreground', 'opacity-100')
    expect(screen.getByTestId('mesh-viz-dot-theme-2-index')).toHaveClass('text-fg-faint', 'opacity-55')

    await user.click(coolTraceSwatch)

    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.62 0.024 252 / 8%)')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.74 0.12 190 / 12%)')
    expect(screen.getByTestId('mesh-viz-tertiary-dot-grid')).toHaveAttribute('fill', 'oklch(0.72 0.13 275 / 10%)')
    expectMeshNodeCoreFill('self', 'oklch(0.8 0.12 28)', '18%')
    expectMeshNodeCoreFill('lemony', 'oklch(0.74 0.12 190)', '14%')

    await openDebugMenu(user)
    expect(screen.getByRole('button', { name: /dot theme 2: cool trace/i })).toHaveAttribute('aria-pressed', 'true')
    await user.click(screen.getByRole('button', { name: /dot theme 3: warm trace/i }))

    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.62 0.024 252 / 8%)')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.78 0.125 74 / 10%)')
    expect(screen.getByTestId('mesh-viz-tertiary-dot-grid')).toHaveAttribute('fill', 'oklch(0.70 0.12 260 / 12%)')
    expectMeshNodeCoreFill('self', 'oklch(0.76 0.12 155)', '18%')

    await openDebugMenu(user)
    await user.click(screen.getByRole('button', { name: /dot theme 1: ash signal/i }))

    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.64 0.025 252 / 9%)')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.72 0.115 220 / 13%)')
    expect(screen.getByTestId('mesh-viz-tertiary-dot-grid')).toHaveAttribute('fill', 'oklch(0.76 0.105 72 / 7%)')
    expectMeshNodeCoreFill('self', 'oklch(0.66 0.22 28)', '18%')
  })

  it('keeps MeshViz palette colors independent from global accent tokens', () => {
    const paletteColors = Object.values(MESH_VIZ_DOT_COLOR_SCHEMES).flatMap((schemes) =>
      schemes.flatMap((scheme) => [...scheme.colors, ...scheme.nodeColors])
    )

    expect(paletteColors).not.toHaveLength(0)
    for (const color of paletteColors) {
      expect(color).not.toContain('var(')
      expect(color).not.toContain('--color-accent')
    }
  })

  it('uses light-mode MeshViz dot color schemes by index', async () => {
    const user = userEvent.setup()
    document.documentElement.dataset.theme = 'light'
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    fireEvent.keyDown(window, { key: 'g', ctrlKey: true })

    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.52 0.022 252 / 12%)')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.54 0.12 220 / 12%)')
    expect(screen.getByTestId('mesh-viz-tertiary-dot-grid')).toHaveAttribute('fill', 'oklch(0.58 0.18 28 / 9%)')

    await openDebugMenu(user)
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-1')).toHaveAttribute(
      'data-color-value',
      'oklch(0.68 0.018 252)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-2')).toHaveAttribute(
      'data-color-value',
      'oklch(0.54 0.12 220)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-3')).toHaveAttribute(
      'data-color-value',
      'oklch(0.58 0.18 28)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-1-color-4')).toHaveAttribute(
      'data-color-value',
      'oklch(0.48 0.01 252)'
    )
    expect(screen.getByTestId('mesh-viz-dot-theme-2-color-4')).toHaveAttribute(
      'data-color-value',
      'oklch(0.48 0.01 252)'
    )
    expect(screen.getByRole('button', { name: /dot theme 1: paper signal/i })).toHaveAttribute('aria-pressed', 'true')

    await user.click(screen.getByRole('button', { name: /dot theme 2: field trace/i }))

    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.55 0.02 252 / 11%)')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.53 0.13 145 / 13%)')
    expect(screen.getByTestId('mesh-viz-tertiary-dot-grid')).toHaveAttribute('fill', 'oklch(0.50 0.12 265 / 10%)')
    expectMeshNodeCoreFill('self', 'oklch(0.48 0.01 252)', '18%')
    expectMeshNodeCoreFill('lemony', 'oklch(0.53 0.13 145)', '14%')

    await openDebugMenu(user)
    expect(screen.getByTestId('mesh-viz-dot-theme-3-color-4')).toHaveAttribute(
      'data-color-value',
      'oklch(0.48 0.01 252)'
    )
    await user.click(screen.getByRole('button', { name: /dot theme 3: amber trace/i }))

    expect(screen.getByTestId('mesh-viz-dot-grid')).toHaveAttribute('fill', 'oklch(0.55 0.02 252 / 11%)')
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.56 0.13 74 / 11%)')
    expect(screen.getByTestId('mesh-viz-tertiary-dot-grid')).toHaveAttribute('fill', 'oklch(0.50 0.12 245 / 9%)')
    expectMeshNodeCoreFill('lemony', 'oklch(0.56 0.13 74)', '14%')
    expectMeshNodeCoreFill('self', 'oklch(0.48 0.01 252)', '18%')
  })

  it('adds separately tracked DEBUG nodes from the MeshViz debug menu', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    expect(screen.queryByRole('button', { name: /view debug-/i })).not.toBeInTheDocument()

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug client/i }))

    expect(screen.getByText(/3 nodes \+ 1 debug/i)).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /view debug-/i })).toHaveLength(1)

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug worker/i }))
    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug host/i }))

    expect(screen.getByText(/3 nodes \+ 3 debug/i)).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /view debug-/i })).toHaveLength(3)
    expect(screen.getByRole('button', { name: /view debug-client-1 node/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /view debug-worker-2 node/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /view debug-host-3 node/i })).toBeInTheDocument()
  })

  it('fades MeshViz node labels at dense counts and reveals the hovered label', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const labelLayer = screen.getByTestId('mesh-node-label-layer')

    expect(labelLayer).toHaveClass('z-[40]')
    expect(labelLayer).toContainElement(getMeshNodeLabel('self'))
    expect(labelLayer.compareDocumentPosition(getMeshNodeCore('self')) & Node.DOCUMENT_POSITION_PRECEDING).toBeTruthy()
    expect(getMeshNodeLabel('self')).toHaveClass('opacity-100')

    fireEvent.keyDown(window, { key: '1', ctrlKey: true })
    fireEvent.keyDown(window, { key: '2', ctrlKey: true })
    fireEvent.keyDown(window, { key: '3', ctrlKey: true })

    await waitFor(() => expect(screen.getAllByRole('button', { name: /view debug-/i })).toHaveLength(3))
    expect(getMeshNodeLabel('self')).toHaveClass('opacity-100', 'duration-[500ms]')
    expect(getMeshNodeLabel('debug-client-1')).toHaveClass('opacity-100')

    fireEvent.keyDown(window, { key: '1', ctrlKey: true })
    fireEvent.keyDown(window, { key: '2', ctrlKey: true })

    await waitFor(() => expect(screen.getAllByRole('button', { name: /view debug-/i })).toHaveLength(5))
    expect(Number(getMeshNodeLabel('self').style.opacity)).toBeGreaterThan(0)
    await waitFor(() => expect(getMeshNodeLabel('self')).toHaveClass('opacity-0'))
    expect(getMeshNodeLabel('self')).toHaveClass('absolute', 'top-full')
    expect(getMeshNodeLabel('self')).toHaveClass('duration-[500ms]')
    expect(getMeshNodeLabel('debug-client-1')).toHaveClass('opacity-0', 'duration-[500ms]')

    await user.hover(getNodeButton('View CARRACK node'))

    await waitFor(() => expect(getMeshNodeLabel('self')).toHaveClass('opacity-100', 'duration-[300ms]'))
    expect(getMeshNodeLabel('debug-client-1')).toHaveClass('opacity-0')
  })

  it('removes DEBUG nodes from the nested MeshViz debug menu', async () => {
    const user = userEvent.setup()
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug client/i }))
    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug worker/i }))
    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug host/i }))

    expect(screen.getAllByRole('button', { name: /view debug-/i })).toHaveLength(3)

    await openRemoveDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug worker/i }))

    expect(screen.getByText(/3 nodes \+ 2 debug/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /view debug-client-1 node/i })).toBeInTheDocument()
    const removedWorkerNode = screen.getByRole('button', { name: /view debug-worker-2 node/i })
    expect(removedWorkerNode).toHaveAttribute('data-node-lifecycle', 'leaving')
    expect(removedWorkerNode).toBeDisabled()
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /view debug-worker-2 node/i })).not.toBeInTheDocument()
    })
    expect(screen.getByRole('button', { name: /view debug-host-3 node/i })).toBeInTheDocument()

    await openRemoveDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug host/i }))

    expect(screen.getByText(/3 nodes \+ 1 debug/i)).toBeInTheDocument()
    await waitFor(() => expect(screen.getAllByRole('button', { name: /view debug-/i })).toHaveLength(1))
    expect(screen.getByRole('button', { name: /view debug-client-1 node/i })).toBeInTheDocument()
  })

  it('places new DEBUG nodes deterministically inside a sparse cluster envelope', async () => {
    const user = userEvent.setup()
    const { unmount } = render(
      <MeshViz meshId="deterministic-test-mesh" nodes={MESH_NODES} selfId="self" height={420} />
    )

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug client/i }))

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug worker/i }))

    await openAddDebugNodesMenu(user)
    await user.click(screen.getByRole('menuitem', { name: /debug host/i }))

    const firstRunCoordinates = debugNodeCoordinates()
    const placementNodes: Array<Pick<MeshNode, 'x' | 'y'>> = [...MESH_NODES]
    const baseCentroid = placementCentroid(MESH_NODES)

    expect(firstRunCoordinates).toHaveLength(3)

    for (const [index, coordinate] of firstRunCoordinates.entries()) {
      const nearestDistance = nearestNodeDistance(coordinate, placementNodes)

      expect(nearestNodeDistance(coordinate, placementNodes)).toBeLessThanOrEqual(
        DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT + 0.01
      )
      expect(nearestDistance).toBeGreaterThanOrEqual(DEBUG_PLACEMENT_MIN_DISTANCE_PERCENT - 0.01)
      expect(distanceFrom(coordinate, baseCentroid)).toBeLessThanOrEqual(
        placementClusterRadius(MESH_NODES, index) + 0.01
      )
      placementNodes.push(coordinate)
    }

    unmount()

    const repeatUser = userEvent.setup()
    render(<MeshViz meshId="deterministic-test-mesh" nodes={MESH_NODES} selfId="self" height={420} />)

    await openAddDebugNodesMenu(repeatUser)
    await repeatUser.click(screen.getByRole('menuitem', { name: /debug client/i }))

    await openAddDebugNodesMenu(repeatUser)
    await repeatUser.click(screen.getByRole('menuitem', { name: /debug worker/i }))

    await openAddDebugNodesMenu(repeatUser)
    await repeatUser.click(screen.getByRole('menuitem', { name: /debug host/i }))

    expect(debugNodeCoordinates()).toEqual(firstRunCoordinates)
  })

  it('keeps growing DEBUG node placement sparse but clustered', () => {
    render(<MeshViz meshId="clustered-debug-mesh" nodes={MESH_NODES} selfId="self" height={420} />)

    for (let index = 0; index < 12; index += 1) {
      fireEvent.keyDown(window, { key: '1', ctrlKey: true })
      fireEvent.keyDown(window, { key: '2', ctrlKey: true })
      fireEvent.keyDown(window, { key: '3', ctrlKey: true })
    }

    const coordinates = debugNodeCoordinates()
    const placementNodes: Array<Pick<MeshNode, 'x' | 'y'>> = [...MESH_NODES]
    const baseCentroid = placementCentroid(MESH_NODES)
    const finalClusterRadius = placementClusterRadius(MESH_NODES, coordinates.length - 1)

    expect(coordinates).toHaveLength(36)
    for (const [index, coordinate] of coordinates.entries()) {
      const nearestDistance = nearestNodeDistance(coordinate, placementNodes)

      expect(nearestDistance).toBeLessThanOrEqual(DEBUG_PLACEMENT_MAX_DISTANCE_PERCENT + 0.01)
      expect(distanceFrom(coordinate, baseCentroid)).toBeLessThanOrEqual(
        placementClusterRadius(MESH_NODES, index) + 0.01
      )
      placementNodes.push(coordinate)
    }

    expect(
      coordinates.some(
        (coordinate) => nearestNodeDistance(coordinate, MESH_NODES) >= DEBUG_PLACEMENT_MIN_DISTANCE_PERCENT - 0.01
      )
    ).toBe(true)
    expect(coordinates.every((coordinate) => distanceFrom(coordinate, baseCentroid) <= finalClusterRadius + 0.01)).toBe(
      true
    )
  })

  it('re-fits MeshViz after topology coordinates change even after manual panning', async () => {
    const { rerender } = render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))
    const initialLeft = pixelValue(lemony29.style.left)

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 1, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 150, clientY: 130 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })
    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeCloseTo(initialLeft + 50, 1))
    const pannedLeft = pixelValue(lemony29.style.left)
    const shiftedNodes = MESH_NODES.map((node) => (node.id === 'lemony-29' ? { ...node, x: 82, y: 82 } : node))

    rerender(<MeshViz nodes={shiftedNodes} selfId="self" height={420} />)

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(pannedLeft + 100))
  })

  it('eases MeshViz out to include a newly added node outside the current viewport', async () => {
    const { rerender } = render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const lemony29 = getNodeButton('View LEMONY-29 node')

    await waitFor(() => expect(pixelValue(lemony29.style.left)).toBeGreaterThan(0))

    fireEvent.wheel(canvas, { deltaY: -100, clientX: 400, clientY: 210 })

    const joinedNode: MeshNode = {
      id: 'joined-outside-view',
      label: 'JOINED',
      subLabel: 'JOINED PEER',
      status: 'online',
      role: 'peer',
      renderKind: 'worker',
      x: 160,
      y: 82
    }

    rerender(<MeshViz nodes={[...MESH_NODES, joinedNode]} selfId="self" height={420} />)

    const joinedButton = getNodeButton('View JOINED node')

    await waitFor(() => {
      expect(pixelValue(joinedButton.style.left)).toBeGreaterThanOrEqual(0)
      expect(pixelValue(joinedButton.style.left)).toBeLessThanOrEqual(800)
      expect(pixelValue(joinedButton.style.top)).toBeGreaterThanOrEqual(0)
      expect(pixelValue(joinedButton.style.top)).toBeLessThanOrEqual(420)
    })
  })

  it('honors reduced-motion preferences for MeshViz animations', async () => {
    window.matchMedia = createMatchMedia(true)
    const meshRef = createRef<MeshVizHandle>()

    render(<MeshViz ref={meshRef} nodes={MESH_NODES} selfId="self" height={420} />)
    const radarPing = getMeshElement('.mesh-radar-ping')

    await waitFor(() => expect(radarPing.style.opacity).toBe('0'))
    expect(radarPing.style.transform).toBe('scale(1)')

    await act(async () => {
      expect(meshRef.current?.playTraffic('self', 'lemony')).toBe(true)
    })

    expect(getMeshPackets()).toHaveLength(0)
  })

  it('keeps repeated MeshViz traffic instances independent on the same transition', async () => {
    const meshRef = createRef<MeshVizHandle>()
    const sourceNodeColor = MESH_VIZ_DOT_COLOR_SCHEMES.dark[0].nodeColors[3]

    render(<MeshViz ref={meshRef} nodes={MESH_NODES} selfId="self" height={420} />)

    await act(async () => {
      expect(meshRef.current?.playTraffic('self', 'lemony')).toBe(true)
      expect(meshRef.current?.playTraffic('self', 'lemony')).toBe(true)
    })

    const packets = getMeshPackets()

    expect(packets).toHaveLength(2)
    expect(packets.every((packet) => packet.style.opacity === '0.92')).toBe(true)
    expect(packets.every((packet) => packet.style.transition.includes('opacity'))).toBe(true)
    expect(packets.every((packet) => packet.style.background.includes(sourceNodeColor))).toBe(true)
  })

  it('does not create MeshViz traffic packets for invalid transitions', async () => {
    const meshRef = createRef<MeshVizHandle>()

    render(<MeshViz ref={meshRef} nodes={MESH_NODES} selfId="self" height={420} />)

    await act(async () => {
      expect(meshRef.current?.playTraffic('self', 'self')).toBe(false)
      expect(meshRef.current?.playTraffic('self', 'missing-node')).toBe(false)
    })

    expect(getMeshPackets()).toHaveLength(0)
  })

  it('repositions in-flight MeshViz traffic packets when the canvas resizes', async () => {
    const meshRef = createRef<MeshVizHandle>()

    render(<MeshViz ref={meshRef} nodes={MESH_NODES} selfId="self" height={420} />)

    await act(async () => {
      expect(meshRef.current?.playTraffic('self', 'lemony')).toBe(true)
    })

    const packet = getMeshPackets()[0]

    if (!packet) {
      throw new Error('Expected an in-flight mesh packet')
    }

    const initialTransform = packet.style.transform

    setMeshCanvasSize(1000, 500)
    await act(async () => {
      triggerMeshResize()
    })

    expect(packet.style.transform).not.toBe(initialTransform)
  })

  it('keeps MeshViz utility controls functional without blocking empty canvas space', async () => {
    const user = userEvent.setup()
    const requestFullscreen = vi.fn(() => Promise.resolve())
    HTMLElement.prototype.requestFullscreen = requestFullscreen

    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    await user.click(screen.getByRole('button', { name: /fullscreen/i }))
    expect(requestFullscreen).toHaveBeenCalledTimes(1)

    const devControlGroup = screen.getByRole('group', { name: /mesh debug controls/i })
    const debugMenuButton = screen.getByRole('button', { name: /^debug$/i })

    expect(devControlGroup).toHaveClass('pointer-events-none')
    expect(debugMenuButton).toHaveClass('pointer-events-auto')

    await user.click(debugMenuButton)
    expect(screen.getByRole('button', { name: /random traffic/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /self traffic/i })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: /debug node actions/i })).toBeInTheDocument()
    const addNodesTrigger = screen.getByRole('button', { name: /^add nodes$/i })
    const disabledRemoveNodesTrigger = screen.getByRole('button', { name: /^remove nodes$/i })

    expect(addNodesTrigger).toHaveAttribute('aria-haspopup', 'menu')
    expect(addNodesTrigger).toHaveAttribute('aria-expanded', 'false')
    expect(disabledRemoveNodesTrigger).toHaveAttribute('aria-haspopup', 'menu')
    expect(disabledRemoveNodesTrigger).toHaveAttribute('aria-expanded', 'false')
    expect(disabledRemoveNodesTrigger).toBeDisabled()
    expect(screen.queryByRole('menuitem', { name: /debug client/i })).not.toBeInTheDocument()

    await user.click(addNodesTrigger)
    expect(addNodesTrigger).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('menu', { name: /add debug nodes/i })).toHaveClass('absolute', 'left-full')
    expect(screen.getByRole('menuitem', { name: /debug client/i })).toHaveAttribute('aria-keyshortcuts', 'Control+1')
    expect(screen.getByRole('menuitem', { name: /debug worker/i })).toHaveAttribute('aria-keyshortcuts', 'Control+2')
    expect(screen.getByRole('menuitem', { name: /debug host/i })).toHaveAttribute('aria-keyshortcuts', 'Control+3')

    await user.click(screen.getByRole('menuitem', { name: /debug client/i }))
    await user.click(debugMenuButton)
    const removeNodesTrigger = screen.getByRole('button', { name: /^remove nodes$/i })

    expect(removeNodesTrigger).toBeEnabled()
    expect(removeNodesTrigger).toHaveAttribute('aria-haspopup', 'menu')
    expect(removeNodesTrigger).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByRole('menuitem', { name: /debug client/i })).not.toBeInTheDocument()

    await user.click(removeNodesTrigger)
    expect(removeNodesTrigger).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('menu', { name: /remove debug nodes/i })).toHaveClass('absolute', 'left-full')
    expect(screen.getByRole('menuitem', { name: /debug client/i })).toHaveAttribute('aria-keyshortcuts', 'Shift+1')
    expect(screen.getByRole('menuitem', { name: /debug worker/i })).toHaveAttribute('aria-keyshortcuts', 'Shift+2')
    expect(screen.getByRole('menuitem', { name: /debug host/i })).toHaveAttribute('aria-keyshortcuts', 'Shift+3')
    expect(screen.getByRole('button', { name: /debug boundaries/i })).toHaveAttribute('aria-keyshortcuts', 'Control+B')
    expect(screen.getByRole('button', { name: /toggle grid style \(lines\)/i })).toHaveAttribute(
      'aria-keyshortcuts',
      'Control+G'
    )
    expect(screen.getByRole('group', { name: /dot theme options/i })).toHaveAttribute('aria-keyshortcuts', 'Control+C')
    expect(screen.getByRole('button', { name: /cycle dot theme/i })).toHaveAttribute('aria-keyshortcuts', 'Control+C')
    expect(screen.getByRole('button', { name: /dot theme 1: ash signal/i })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText('Z')).toBeInTheDocument()
    expect(screen.getByText('X')).toBeInTheDocument()
    expect(screen.getByText('Ctrl+G')).toBeInTheDocument()
    expect(screen.getByText('Ctrl+C')).toBeInTheDocument()
    expect(screen.getByText('Shift+1')).toBeInTheDocument()
    expect(screen.getByText('Shift+2')).toBeInTheDocument()
    expect(screen.getByText('Shift+3')).toBeInTheDocument()
  })

  it('supports MeshViz debug hotkeys outside text editing targets', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)

    expect(screen.queryByRole('button', { name: /view debug-/i })).not.toBeInTheDocument()

    fireEvent.keyDown(window, { key: '1', ctrlKey: true })
    expect(screen.getByText(/3 nodes \+ 1 debug/i)).toBeInTheDocument()

    fireEvent.keyDown(window, { key: '2', ctrlKey: true })
    expect(screen.getByText(/3 nodes \+ 2 debug/i)).toBeInTheDocument()

    fireEvent.keyDown(window, { key: '3', ctrlKey: true })
    expect(screen.getByText(/3 nodes \+ 3 debug/i)).toBeInTheDocument()

    fireEvent.keyDown(window, { key: '!', code: 'Digit1', shiftKey: true })
    expect(screen.getByText(/3 nodes \+ 2 debug/i)).toBeInTheDocument()

    fireEvent.keyDown(window, { key: '@', code: 'Digit2', shiftKey: true })
    expect(screen.getByText(/3 nodes \+ 1 debug/i)).toBeInTheDocument()

    fireEvent.keyDown(window, { key: '#', code: 'Digit3', shiftKey: true })
    expect(screen.queryByText(/3 nodes \+ \d debug/i)).not.toBeInTheDocument()

    fireEvent.keyDown(window, { key: 'b', ctrlKey: true })
    expect(screen.getByTestId('mesh-centered-bounds-box')).toBeInTheDocument()

    fireEvent.keyDown(window, { key: 'g', ctrlKey: true })
    expect(screen.queryByTestId('mesh-viz-line-grid')).not.toBeInTheDocument()
    expect(screen.getByTestId('mesh-viz-dot-grid')).toBeInTheDocument()

    const dotThemeEvent = new KeyboardEvent('keydown', { key: 'c', ctrlKey: true, bubbles: true, cancelable: true })
    await act(async () => {
      window.dispatchEvent(dotThemeEvent)
    })
    expect(dotThemeEvent.defaultPrevented).toBe(true)
    expect(screen.getByTestId('mesh-viz-accent-dot-grid')).toHaveAttribute('fill', 'oklch(0.74 0.12 190 / 12%)')

    const randomTrafficEvent = new KeyboardEvent('keydown', { key: 'z', bubbles: true, cancelable: true })
    window.dispatchEvent(randomTrafficEvent)
    expect(randomTrafficEvent.defaultPrevented).toBe(true)

    const selfTrafficEvent = new KeyboardEvent('keydown', { key: 'x', bubbles: true, cancelable: true })
    window.dispatchEvent(selfTrafficEvent)
    expect(selfTrafficEvent.defaultPrevented).toBe(true)

    const input = document.createElement('input')
    document.body.append(input)
    input.focus()
    fireEvent.keyDown(input, { key: '1', ctrlKey: true })
    expect(screen.queryByText(/3 nodes \+ \d debug/i)).not.toBeInTheDocument()
    input.remove()
  })

  it('doubles MeshViz viewport and debug controls while the canvas is fullscreen', async () => {
    render(<MeshViz nodes={MESH_NODES} selfId="self" height={420} />)
    const canvas = getMeshCanvas()
    const debugButton = screen.getByRole('button', { name: /^debug$/i })
    const zoomInButton = screen.getByRole('button', { name: /zoom in/i })
    const zoomOutButton = screen.getByRole('button', { name: /zoom out/i })
    const resetButton = screen.getByRole('button', { name: /reset view/i })

    expect(debugButton).toHaveClass('gap-1.5', 'px-2.5', 'py-1', 'text-[length:var(--density-type-annotation)]')
    expect(zoomInButton).toHaveClass('size-[26px]')
    expect(zoomOutButton).toHaveClass('size-[26px]')
    expect(resetButton).toHaveClass('size-[26px]')

    setFullscreenElement(canvas)
    fireEvent(document, new Event('fullscreenchange'))

    await waitFor(() =>
      expect(debugButton).toHaveClass('gap-3', 'px-5', 'py-2', 'text-[length:var(--density-type-caption)]')
    )
    await waitFor(() => expect(zoomInButton).toHaveClass('size-[52px]'))
    expect(zoomOutButton).toHaveClass('size-[52px]')
    expect(resetButton).toHaveClass('size-[52px]')

    setFullscreenElement(null)
    fireEvent(document, new Event('fullscreenchange'))

    await waitFor(() =>
      expect(debugButton).toHaveClass('gap-1.5', 'px-2.5', 'py-1', 'text-[length:var(--density-type-annotation)]')
    )
    await waitFor(() => expect(zoomInButton).toHaveClass('size-[26px]'))
  })

  it('renders chat and opens transparency from a message', async () => {
    const user = userEvent.setup()
    window.localStorage.setItem(
      APP_STORAGE_KEYS.featureFlagOverrides,
      JSON.stringify({ chat: { transparencyTab: true } })
    )

    render(
      <FeatureFlagProvider>
        <ChatTab />
      </FeatureFlagProvider>
    )

    await user.click(screen.getByRole('button', { name: /inspect transparency/i }))
    expect(screen.getByText(/inbound route/i)).toBeInTheDocument()
    expect(screen.getByText(/link healthy/i)).toBeInTheDocument()
  })

  it('switches chat conversations from the sidebar', async () => {
    const user = userEvent.setup()
    render(<ChatTab />)
    expect(screen.getByText(/newsletter about local ai/i)).toBeInTheDocument()
    expect(screen.queryByText(/pooled placement plan/i)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /model capacity draft/i }))
    expect(screen.getByText(/pooled placement plan/i)).toBeInTheDocument()
    expect(screen.getByText(/use pooled placement on perseus.local/i)).toBeInTheDocument()
    expect(screen.queryByText(/newsletter about local ai/i)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /routing latency notes/i }))
    expect(screen.getByText(/newsletter about local ai/i)).toBeInTheDocument()
    expect(screen.queryByText(/pooled placement plan/i)).not.toBeInTheDocument()
  })

  it('renders configuration controls and keeps placement reflected in TOML output', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)
    const configurationHeading = screen.getByRole('heading', { name: 'Configuration', level: 1 })
    expect(configurationHeading).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /toml output/i })).toBeInTheDocument()
    expect(screen.getByText('⌫')).toBeInTheDocument()
    expect(screen.getAllByText(/placement/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/add model/i).length).toBeGreaterThan(0)
    const configurationHeader = configurationHeading.closest('header')
    const nodeRail = screen.getByRole('navigation', { name: /configuration nodes/i })
    if (!configurationHeader) throw new Error('Expected configuration header')
    const keyboardShortcuts = within(nodeRail).getByRole('region', { name: /keyboard shortcuts/i })
    expect(within(keyboardShortcuts).queryByText('Keyboard:')).not.toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Navigate')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Toggle Section')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('␣')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Selected Model')).toBeInTheDocument()
    expect(within(keyboardShortcuts).queryByText('Adjust')).not.toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Actions')).toBeInTheDocument()
    const keyboardLegendText = keyboardShortcuts.textContent ?? ''
    expect(keyboardLegendText.indexOf('Select Model')).toBeLessThan(keyboardLegendText.indexOf('Selected Model'))
    expect(keyboardLegendText.indexOf('First/Last Model')).toBeLessThan(keyboardLegendText.indexOf('Selected Model'))
    expect(within(keyboardShortcuts).getByText('Adjust Context')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Jump Context')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Move GPU')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Add model')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Toggle Placement')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Undo')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Redo')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Save config')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Revert')).toBeInTheDocument()
    expect(keyboardLegendText.indexOf('Actions')).toBeLessThan(keyboardLegendText.indexOf('Add model'))
    expect(keyboardLegendText.indexOf('Add model')).toBeLessThan(keyboardLegendText.indexOf('Undo'))
    expect(within(keyboardShortcuts).queryByText('Alt')).not.toBeInTheDocument()
    expect(within(keyboardShortcuts).queryByText('Ctrl')).not.toBeInTheDocument()
    expect(within(keyboardShortcuts).queryByText('Shift')).not.toBeInTheDocument()
    expect(within(keyboardShortcuts).queryByText('Tab')).not.toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('⇥')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getAllByText('⌥')).toHaveLength(2)
    expect(within(keyboardShortcuts).getAllByText('⌃')).toHaveLength(4)
    expect(within(keyboardShortcuts).getAllByText('⇧')).toHaveLength(3)
    expect(within(keyboardShortcuts).getByText('Z')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('R')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getAllByText('S').length).toBeGreaterThan(0)
    expect(within(keyboardShortcuts).getByText('X')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('Selected model')).toBeInTheDocument()
    expect(within(keyboardShortcuts).getByText('⌫')).toBeInTheDocument()
    expect(within(configurationHeader).queryByRole('region', { name: /keyboard shortcuts/i })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /undo/i })).toHaveAttribute('aria-keyshortcuts', 'Control+Z')
    expect(screen.getByRole('button', { name: /redo/i })).toHaveAttribute('aria-keyshortcuts', 'Control+R')
    expect(screen.getByRole('button', { name: /revert/i })).toHaveAttribute('aria-keyshortcuts', 'Control+X')
    expect(screen.getByRole('button', { name: /save config/i })).toHaveAttribute('aria-keyshortcuts', 'Control+S')
    expect(screen.getByRole('button', { name: /save config/i })).toBeDisabled()
    expect(buildTOML(CFG_NODES, INITIAL_ASSIGNS)).not.toContain('placement = "pooled"')
    expect(screen.getByRole('button', { name: /remove llama-3\.3-70b-q4_k_m/i })).toBeInTheDocument()

    await user.click(configurationHeading)
    expect(screen.queryByRole('button', { name: /remove llama-3\.3-70b-q4_k_m/i })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /qwen3-4b-q4_k_m, 2\.6 gb weights/i }))
    expect(screen.getByRole('button', { name: /remove qwen3-4b-q4_k_m/i })).toBeInTheDocument()

    const carrackKeyboardTarget = screen.getByRole('button', {
      name: /collapse carrack\. use up and down arrows to select gpu slots/i
    })
    expect(carrackKeyboardTarget).toHaveTextContent('▾')
    expect(carrackKeyboardTarget).not.toHaveTextContent(/carrack/i)
    expect(carrackKeyboardTarget).not.toHaveClass('focus-visible:outline-accent')
    const carrackSection = carrackKeyboardTarget.closest('section')
    if (!carrackSection) throw new Error('Expected carrack section')
    expect(carrackSection).toHaveAttribute('data-config-node-selected', 'true')
    expect(carrackSection.className).not.toContain('shadow-[0_0_0_1px_var(--color-accent)]')
    carrackKeyboardTarget.focus()
    expect(carrackKeyboardTarget).toHaveFocus()
    expect(screen.queryByRole('button', { name: /remove qwen3\.5-27b-ud-q4_k_xl/i })).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'perseus.local' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'triton.lab' })).toBeInTheDocument()
    expect(screen.getByText('Peers')).toBeInTheDocument()
    expect(screen.getByText('read-only')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i }))
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m/i })).toBeInTheDocument()
    await user.keyboard('{Alt>}{ArrowRight}{/Alt}')
    expect(
      screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights, 0\.4 gb context cache/i })
    ).toHaveTextContent('17,408 ctx')
    await user.keyboard('{Alt>}{ArrowLeft}{/Alt}')
    expect(
      screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights, 0\.4 gb context cache/i })
    ).toHaveTextContent('16,384 ctx')
    await user.keyboard('{Alt>}{Shift>}{ArrowRight}{/Shift}{/Alt}')
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '32,768 ctx'
    )
    await user.keyboard('{Alt>}{Shift>}{ArrowLeft}{/Shift}{/Alt}')
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '16,384 ctx'
    )
    await user.keyboard('{Shift>}{ArrowDown}{/Shift}')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 3/i })).toBeInTheDocument()
    await user.keyboard('{Shift>}{ArrowUp}{/Shift}')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()
    await user.keyboard('{ArrowUp}')
    expect(screen.getByRole('button', { name: /remove llama-3\.3-70b-q4_k_m/i })).toBeInTheDocument()
    await user.keyboard('{ArrowDown}')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m/i })).toBeInTheDocument()
    await user.keyboard('{Delete}')
    expect(screen.queryByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /remove qwen3\.5-27b-q4_k_m/i })).not.toBeInTheDocument()

    await user.click(document.body)
    expect(screen.queryByRole('button', { name: /remove qwen3-4b-q4_k_m/i })).not.toBeInTheDocument()

    const perseusSection = screen.getByRole('heading', { name: 'perseus.local' }).closest('section')
    const tritonSection = screen.getByRole('heading', { name: 'triton.lab' }).closest('section')

    if (!perseusSection || !tritonSection) throw new Error('Expected remote context sections')

    expect(screen.getByRole('button', { name: 'Add model to perseus.local' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Add model to triton.lab' })).toBeDisabled()
    expect(within(perseusSection).queryByText('read-only')).not.toBeInTheDocument()
    expect(within(tritonSection).queryByText('read-only')).not.toBeInTheDocument()

    const reservedLane = within(carrackSection).getAllByRole('button', { name: /system reserved space/i })[0]
    await user.click(reservedLane)
    expect(reservedLane).toHaveAttribute('aria-pressed', 'true')
    expect(within(carrackSection).getByRole('heading', { name: /system reserved space/i })).toBeInTheDocument()
    expect(
      within(carrackSection).getByText(/invariant system reserved space and has no configurable settings/i)
    ).toBeInTheDocument()
    expect(within(carrackSection).queryByRole('button', { name: /remove qwen3-4b-q4_k_m/i })).not.toBeInTheDocument()

    expect(within(perseusSection).getByRole('radio', { name: 'separate' })).toBeDisabled()
    expect(within(perseusSection).getByRole('radio', { name: 'pooled' })).toBeDisabled()
    expect(within(tritonSection).getByRole('radio', { name: 'separate' })).toBeDisabled()
    expect(within(tritonSection).getByRole('radio', { name: 'pooled' })).toBeDisabled()

    const assignedModelDrag = createMockDataTransfer()
    const sameNodeDestination = within(carrackSection).getByRole('region', { name: /rtx 5090 capacity/i })
    const sourceContainer = within(carrackSection).getByRole('region', { name: /rtx 3080 capacity/i })

    fireEvent.dragStart(screen.getByRole('button', { name: /qwen3-4b-q4_k_m, 2\.6 gb weights/i }), {
      dataTransfer: assignedModelDrag
    })
    expect(assignedModelDrag.setData).toHaveBeenCalledWith('text/assign-id', 'a4')
    expect(assignedModelDrag.setData).toHaveBeenCalledWith('text/source-node', 'node-a')
    expect(assignedModelDrag.setData).toHaveBeenCalledWith('text/source-container', '7')
    expect(assignedModelDrag.setData).toHaveBeenCalledWith('application/x-mesh-source-container-node-a-7', 'node-a-7')

    fireEvent.dragEnter(sourceContainer, { dataTransfer: assignedModelDrag })
    fireEvent.dragOver(sourceContainer, { dataTransfer: assignedModelDrag })
    expect(within(sourceContainer).queryByText('Drop to assign')).not.toBeInTheDocument()
    expect(assignedModelDrag.dropEffect).toBe('none')

    fireEvent.drop(sourceContainer, { dataTransfer: assignedModelDrag })
    expect(within(sourceContainer).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()

    fireEvent.dragEnter(sameNodeDestination, { dataTransfer: assignedModelDrag })
    fireEvent.dragOver(sameNodeDestination, { dataTransfer: assignedModelDrag })
    expect(within(sameNodeDestination).getByText('Drop to assign')).toBeInTheDocument()
    expect(assignedModelDrag.dropEffect).toBe('move')

    fireEvent.drop(sameNodeDestination, { dataTransfer: assignedModelDrag })
    await waitFor(() =>
      expect(within(sameNodeDestination).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
    )
    expect(within(sourceContainer).queryByRole('button', { name: /qwen3-4b-q4_k_m/i })).not.toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('radio', { name: 'pooled' }))
    expect(screen.getByRole('button', { name: /save config/i })).toBeEnabled()
    expect(within(carrackSection).getByRole('radio', { name: 'pooled' })).toBeChecked()
    carrackKeyboardTarget.focus()
    await user.keyboard('s')
    expect(within(carrackSection).getByRole('radio', { name: 'separate' })).toBeChecked()
    await user.keyboard('p')
    expect(within(carrackSection).getByRole('radio', { name: 'pooled' })).toBeChecked()

    const initialToml = buildTOML(CFG_NODES, INITIAL_ASSIGNS)
    expect(initialToml).toContain('hostname = "carrack"')
    expect(initialToml).not.toContain('perseus.local')
    expect(initialToml).not.toContain('triton.lab')
    expect(initialToml).toContain('gpu_index = 0')
    expect(initialToml).not.toContain('gpu = ')

    carrackKeyboardTarget.focus()
    await user.keyboard('a')
    expect(screen.getByRole('dialog', { name: 'Model catalog' })).toBeInTheDocument()
    expect(screen.getAllByText('Fits').length).toBeGreaterThan(0)
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())
    expect(screen.getByRole('button', { name: /remove phi-4-mini/i })).toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'llava')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.click(screen.getByRole('button', { name: /llava-next-34b, 22 gb/i }))
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())
    expect(screen.getByRole('button', { name: /remove llava-next-34b/i })).toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    const dragTransfer = createMockDataTransfer()
    fireEvent.dragStart(screen.getByRole('button', { name: /qwen3-4b-q4_k_m, 2.6 gb/i }), {
      clientX: 12,
      clientY: 12,
      dataTransfer: dragTransfer
    })
    expect(dragTransfer.setData).toHaveBeenCalledWith('text/model', 'qwen4')
    expect(dragTransfer.setDragImage).toHaveBeenCalledWith(
      expect.any(HTMLElement),
      expect.any(Number),
      expect.any(Number)
    )
    await user.click(screen.getByRole('button', { name: 'Close' }))
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.click(screen.getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument()
  })

  it('highlights the selected GPU container and targets it for catalog Enter adds', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)

    const carrackSection = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
    if (!carrackSection) throw new Error('Expected carrack section')

    const qwen4Button = within(carrackSection).getByRole('button', { name: /qwen3-4b-q4_k_m, 2\.6 gb weights/i })
    await user.click(qwen4Button)

    let rtx3080Capacity = within(carrackSection).getByRole('region', { name: /rtx 3080 capacity/i })
    const selected3080Container = rtx3080Capacity.closest('[data-config-container-selected="true"]')
    if (!(selected3080Container instanceof HTMLElement)) throw new Error('Expected selected RTX 3080 container')
    expect(selected3080Container).toContainElement(qwen4Button)

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 7/i })).toBeInTheDocument()
    rtx3080Capacity = within(carrackSection).getByRole('region', { name: /rtx 3080 capacity/i })
    expect(within(rtx3080Capacity).getByRole('button', { name: /phi-4-mini/i })).toBeInTheDocument()
    expect(rtx3080Capacity.closest('[data-config-container-selected="true"]')).toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'llava')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /remove llava-next-34b from gpu 2/i })).toBeInTheDocument()
    expect(within(rtx3080Capacity).queryByRole('button', { name: /llava-next-34b/i })).not.toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'mixtral')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')

    expect(screen.getByRole('textbox', { name: 'Command bar search' })).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(/mixtral-8x22b does not fit on any gpu in carrack/i)
    expect(screen.queryByRole('button', { name: /remove mixtral-8x22b/i })).not.toBeInTheDocument()
  })

  it('uses the clicked GPU container as the catalog add target', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)

    const carrackSection = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
    if (!carrackSection) throw new Error('Expected carrack section')

    const rtx6000CapacityRegions = within(carrackSection).getAllByRole('region', { name: /rtx 6000 pro capacity/i })
    const gpu3Capacity = rtx6000CapacityRegions[2]
    if (!gpu3Capacity) throw new Error('Expected carrack GPU 3 capacity region')

    await user.click(gpu3Capacity)

    const selectedGpu3Container = gpu3Capacity.closest('[data-config-container-selected="true"]')
    if (!(selectedGpu3Container instanceof HTMLElement))
      throw new Error('Expected clicked GPU 3 container to be selected')

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 3/i })).toBeInTheDocument()
    expect(within(gpu3Capacity).getByRole('button', { name: /phi-4-mini/i })).toBeInTheDocument()
  })

  it('restores separate GPU assignments after previewing pooled placement', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)

    const carrackSection = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
    if (!carrackSection) throw new Error('Expected carrack section')

    const gpu2Capacity = within(carrackSection).getAllByRole('region', { name: /rtx 6000 pro capacity/i })[1]
    const rtx3080Capacity = within(carrackSection).getByRole('region', { name: /rtx 3080 capacity/i })
    if (!gpu2Capacity) throw new Error('Expected carrack GPU 2 capacity region')

    expect(within(gpu2Capacity).getByRole('button', { name: /qwen3\.5-27b-q4_k_m/i })).toBeInTheDocument()
    expect(within(rtx3080Capacity).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('radio', { name: 'pooled' }))
    await user.click(within(carrackSection).getByRole('radio', { name: 'separate' }))

    const restoredGpu2Capacity = within(carrackSection).getAllByRole('region', { name: /rtx 6000 pro capacity/i })[1]
    const restoredRtx3080Capacity = within(carrackSection).getByRole('region', { name: /rtx 3080 capacity/i })
    if (!restoredGpu2Capacity) throw new Error('Expected restored carrack GPU 2 capacity region')

    expect(within(restoredGpu2Capacity).getByRole('button', { name: /qwen3\.5-27b-q4_k_m/i })).toBeInTheDocument()
    expect(within(restoredRtx3080Capacity).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
  })

  it('keeps separate placement snapshots aligned with undo history', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)
    const dispatchUndo = async () => {
      const event = new KeyboardEvent('keydown', { key: 'z', ctrlKey: true, bubbles: true, cancelable: true })

      await act(async () => {
        window.dispatchEvent(event)
      })
      expect(event.defaultPrevented).toBe(true)
    }
    const getCarrackSection = () => {
      const section = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
      if (!section) throw new Error('Expected carrack section')
      return section
    }
    const getRtx3080Capacity = () => within(getCarrackSection()).getByRole('region', { name: /rtx 3080 capacity/i })

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'separate' }))
    await user.click(within(getRtx3080Capacity()).getByRole('button', { name: /qwen3-4b-q4_k_m/i }))
    await user.keyboard('{Shift>}{ArrowUp}{/Shift}')
    expect(screen.getByRole('button', { name: /remove qwen3-4b-q4_k_m from gpu 6/i })).toBeInTheDocument()

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    await dispatchUndo()
    await dispatchUndo()
    await dispatchUndo()
    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'separate' }))

    expect(within(getRtx3080Capacity()).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
  })

  it('enables save only for dirty configuration changes and supports the save shortcut', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)
    const saveButton = screen.getByRole('button', { name: /save config/i })
    const revertButton = screen.getByRole('button', { name: /revert/i })
    expect(saveButton).toHaveAttribute('aria-keyshortcuts', 'Control+S')
    expect(revertButton).toHaveAttribute('aria-keyshortcuts', 'Control+X')
    expect(saveButton).toBeDisabled()
    expect(saveButton).toHaveAttribute('title', 'No changes to save')

    const carrackSection = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
    if (!carrackSection) throw new Error('Expected carrack section')
    await user.click(within(carrackSection).getByRole('radio', { name: 'pooled' }))
    expect(saveButton).toBeEnabled()

    const saveEvent = new KeyboardEvent('keydown', { key: 's', ctrlKey: true, bubbles: true, cancelable: true })
    await act(async () => {
      window.dispatchEvent(saveEvent)
    })
    expect(saveEvent.defaultPrevented).toBe(true)
    expect(saveButton).toBeDisabled()
    expect(saveButton).toHaveAttribute('title', 'No changes to save')
  })

  it('reverts dirty configuration changes with the revert shortcut', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)
    const saveButton = screen.getByRole('button', { name: /save config/i })
    const getCarrackSection = () => {
      const section = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
      if (!section) throw new Error('Expected carrack section')
      return section
    }

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    expect(saveButton).toBeEnabled()
    const revertEvent = new KeyboardEvent('keydown', { key: 'x', ctrlKey: true, bubbles: true, cancelable: true })
    await act(async () => {
      window.dispatchEvent(revertEvent)
    })
    expect(revertEvent.defaultPrevented).toBe(true)
    expect(saveButton).toBeDisabled()
    await expectTomlOccurrences(user, 'placement = "pooled"', 0)

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    await expectTomlOccurrences(user, 'placement = "pooled"', 1)
    const saveEvent = new KeyboardEvent('keydown', { key: 's', ctrlKey: true, bubbles: true, cancelable: true })
    await act(async () => {
      window.dispatchEvent(saveEvent)
    })
    expect(saveEvent.defaultPrevented).toBe(true)
    expect(saveButton).toBeDisabled()

    await user.click(within(getCarrackSection()).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())
    expect(screen.getByRole('button', { name: /phi-4-mini, .* weights/i })).toBeInTheDocument()
    expect(saveButton).toBeEnabled()
    await expectTomlOccurrences(user, 'placement = "pooled"', 1)
    const revertToSavedEvent = new KeyboardEvent('keydown', {
      key: 'x',
      ctrlKey: true,
      bubbles: true,
      cancelable: true
    })
    await act(async () => {
      window.dispatchEvent(revertToSavedEvent)
    })
    expect(revertToSavedEvent.defaultPrevented).toBe(true)
    expect(saveButton).toBeDisabled()
    await expectTomlOccurrences(user, 'placement = "pooled"', 1)
    expect(screen.queryByRole('button', { name: /phi-4-mini, .* weights/i })).not.toBeInTheDocument()
  })

  it('tracks full configuration history with Ctrl+Z and Ctrl+R', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)
    const undoButton = screen.getByRole('button', { name: /undo/i })
    const redoButton = screen.getByRole('button', { name: /redo/i })
    const dispatchShortcut = async (key: string) => {
      const event = new KeyboardEvent('keydown', { key, ctrlKey: true, bubbles: true, cancelable: true })

      await act(async () => {
        window.dispatchEvent(event)
      })
      expect(event.defaultPrevented).toBe(true)
    }

    expect(undoButton).toHaveAttribute('aria-keyshortcuts', 'Control+Z')
    expect(redoButton).toHaveAttribute('aria-keyshortcuts', 'Control+R')

    await user.keyboard('{ArrowDown}')
    await user.keyboard('{Alt>}{ArrowRight}{/Alt}')
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '17,408 ctx'
    )
    expect(undoButton).toBeEnabled()

    await dispatchShortcut('z')
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '16,384 ctx'
    )
    expect(redoButton).toBeEnabled()

    await dispatchShortcut('r')
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '17,408 ctx'
    )

    await user.keyboard('{Shift>}{ArrowDown}{/Shift}')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 3/i })).toBeInTheDocument()
    await dispatchShortcut('z')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()
    await dispatchShortcut('r')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 3/i })).toBeInTheDocument()

    const getCarrackSection = () => {
      const section = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
      if (!section) throw new Error('Expected carrack section')
      return section
    }

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    await expectTomlOccurrences(user, 'placement = "pooled"', 1)
    await dispatchShortcut('z')
    await expectTomlOccurrences(user, 'placement = "pooled"', 0)
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toBeInTheDocument()
    await dispatchShortcut('r')
    await expectTomlOccurrences(user, 'placement = "pooled"', 1)

    await user.click(within(getCarrackSection()).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())
    expect(screen.getByRole('button', { name: /phi-4-mini, .* weights/i })).toBeInTheDocument()
    await dispatchShortcut('z')
    expect(screen.queryByRole('button', { name: /phi-4-mini, .* weights/i })).not.toBeInTheDocument()
    await dispatchShortcut('r')
    expect(screen.getByRole('button', { name: /phi-4-mini, .* weights/i })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /remove phi-4-mini/i }))
    expect(screen.queryByRole('button', { name: /phi-4-mini, .* weights/i })).not.toBeInTheDocument()
    await dispatchShortcut('z')
    expect(screen.getByRole('button', { name: /phi-4-mini, .* weights/i })).toBeInTheDocument()
    await dispatchShortcut('r')
    expect(screen.queryByRole('button', { name: /phi-4-mini, .* weights/i })).not.toBeInTheDocument()
  })

  it('tracks drag and drop configuration history with Ctrl+Z and Ctrl+R', async () => {
    const user = userEvent.setup()
    render(<ConfigurationTab initialTab="local-deployment" enableNavigationBlocker={false} />)
    const dispatchShortcut = async (key: string) => {
      const event = new KeyboardEvent('keydown', { key, ctrlKey: true, bubbles: true, cancelable: true })

      await act(async () => {
        window.dispatchEvent(event)
      })
      expect(event.defaultPrevented).toBe(true)
    }

    const carrackSection = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
    if (!carrackSection) throw new Error('Expected configuration section')

    const sourceContainer = within(carrackSection).getByRole('region', { name: /rtx 3080 capacity/i })
    const destinationContainer = within(carrackSection).getByRole('region', { name: /rtx 5090 capacity/i })
    const assignedModelDrag = createMockDataTransfer()

    fireEvent.dragStart(within(sourceContainer).getByRole('button', { name: /qwen3-4b-q4_k_m/i }), {
      dataTransfer: assignedModelDrag
    })
    fireEvent.dragEnter(destinationContainer, { dataTransfer: assignedModelDrag })
    fireEvent.dragOver(destinationContainer, { dataTransfer: assignedModelDrag })
    fireEvent.drop(destinationContainer, { dataTransfer: assignedModelDrag })
    await waitFor(() =>
      expect(within(destinationContainer).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
    )
    expect(within(sourceContainer).queryByRole('button', { name: /qwen3-4b-q4_k_m/i })).not.toBeInTheDocument()

    await dispatchShortcut('z')
    await waitFor(() =>
      expect(within(sourceContainer).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
    )
    expect(within(destinationContainer).queryByRole('button', { name: /qwen3-4b-q4_k_m/i })).not.toBeInTheDocument()
    await dispatchShortcut('r')
    await waitFor(() =>
      expect(within(destinationContainer).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
    )
    expect(within(sourceContainer).queryByRole('button', { name: /qwen3-4b-q4_k_m/i })).not.toBeInTheDocument()

    await user.click(within(carrackSection).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    const catalogDrag = createMockDataTransfer()
    fireEvent.dragStart(screen.getByRole('button', { name: /phi-4-mini, .* gb, .* context, fits/i }), {
      clientX: 12,
      clientY: 12,
      dataTransfer: catalogDrag
    })
    fireEvent.dragEnter(sourceContainer, { dataTransfer: catalogDrag })
    fireEvent.dragOver(sourceContainer, { dataTransfer: catalogDrag })
    fireEvent.drop(sourceContainer, { dataTransfer: catalogDrag })
    await waitFor(() =>
      expect(within(sourceContainer).getByRole('button', { name: /phi-4-mini/i })).toBeInTheDocument()
    )

    await dispatchShortcut('z')
    await waitFor(() =>
      expect(within(sourceContainer).queryByRole('button', { name: /phi-4-mini/i })).not.toBeInTheDocument()
    )
    await dispatchShortcut('r')
    await waitFor(() =>
      expect(within(sourceContainer).getByRole('button', { name: /phi-4-mini/i })).toBeInTheDocument()
    )
  })
})
