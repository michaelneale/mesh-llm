import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { VRAMBar } from '@/features/configuration/components/VRAMBar'
import { contextGB, findModel } from '@/features/configuration/lib/config-math'
import { reservedVramSelectionId } from '@/features/configuration/lib/selection'
import type { ConfigAssign, ConfigNode } from '@/features/app-tabs/types'

function createMockDataTransfer() {
  const data = new Map<string, string>()

  return {
    dropEffect: 'none',
    effectAllowed: 'all',
    get types() { return Array.from(data.keys()) },
    getData: vi.fn((type: string) => data.get(type) ?? ''),
    setData: vi.fn((type: string, value: string) => { data.set(type, value) }),
    setDragImage: vi.fn(),
  }
}

const testNode: ConfigNode = {
  id: 'node-test',
  hostname: 'test-node',
  region: 'lab',
  status: 'online',
  cpu: 'test cpu',
  ramGB: 64,
  placement: 'separate',
  gpus: [{ idx: 0, name: 'test gpu', totalGB: 4, reservedGB: 1 }],
}

describe('VRAMBar', () => {
  it('sizes model usage against usable VRAM instead of raw total capacity', () => {
    const assigns: ConfigAssign[] = [{ id: 'assign-qwen', modelId: 'qwen4', nodeId: testNode.id, containerIdx: 0, ctx: 4096 }]
    const model = findModel('qwen4')

    if (!model) throw new Error('Expected qwen4 test model')

    render(
      <VRAMBar
        node={testNode}
        label={{ prefix: 'GPU 0', main: 'test gpu' }}
        totalGB={4}
        reservedGB={1}
        containerIdx={0}
        assigns={assigns}
        onPick={vi.fn()}
        onSelectContainer={vi.fn()}
        setAssigns={vi.fn()}
        dragOver={null}
        setDragOver={vi.fn()}
      />,
    )

    const bar = screen.getByRole('region', { name: /test gpu capacity/i })
    const reservedLane = within(bar).getByTitle('Reserved 1 GB')
    expect(reservedLane).toBeInTheDocument()
    expect(bar.firstElementChild).toBe(reservedLane)
    const modelSegment = within(bar).getByRole('button', { name: /qwen3-4b-q4_k_m/i })
    const widthPercent = Number.parseFloat(modelSegment.style.width)
    const expectedUsableWidth = ((model.sizeGB + contextGB(model, assigns[0].ctx)) / 3) * 100
    const rawTotalWidth = ((model.sizeGB + contextGB(model, assigns[0].ctx)) / 4) * 100

    expect(widthPercent).toBeCloseTo(expectedUsableWidth, 3)
    expect(widthPercent).toBeGreaterThan(rawTotalWidth)
  })

  it('selects and highlights reserved VRAM without changing placement', () => {
    const onPick = vi.fn()
    const selectedReservedId = reservedVramSelectionId(testNode.id, 0)

    render(
      <VRAMBar
        node={testNode}
        label={{ prefix: 'GPU 0', main: 'test gpu' }}
        totalGB={4}
        reservedGB={1}
        containerIdx={0}
        assigns={[]}
        selectedId={selectedReservedId}
        onPick={onPick}
        onSelectContainer={vi.fn()}
        setAssigns={vi.fn()}
        dragOver={null}
        setDragOver={vi.fn()}
      />,
    )

    const reservedLane = screen.getByRole('button', { name: /system reserved space, 1 GB reserved on test gpu/i })
    expect(reservedLane).toHaveAttribute('aria-pressed', 'true')

    fireEvent.click(reservedLane)
    expect(onPick).toHaveBeenCalledWith(selectedReservedId)
  })

  it('keeps size and context metadata visible on dense model bars', () => {
    const assigns: ConfigAssign[] = [{ id: 'assign-qwen-ud', modelId: 'qwenud', nodeId: testNode.id, containerIdx: 0, ctx: 262144 }]

    render(
      <VRAMBar
        node={testNode}
        label={{ prefix: 'GPU 0', main: 'test gpu' }}
        totalGB={48}
        reservedGB={1}
        containerIdx={0}
        assigns={assigns}
        onPick={vi.fn()}
        onSelectContainer={vi.fn()}
        setAssigns={vi.fn()}
        dragOver={null}
        setDragOver={vi.fn()}
        dense
      />,
    )

    expect(screen.getByRole('button', { name: /17\.8 GB weights, 6\.4 GB context cache/i })).toBeInTheDocument()
    expect(screen.getByText('17.8 GB · 262,144 ctx (6.4 GB)')).toBeInTheDocument()
  })

  it('does not allow models to be dropped onto reserved VRAM', () => {
    const setAssigns = vi.fn()
    const dataTransfer = createMockDataTransfer()

    dataTransfer.setData('text/model', 'qwen4')

    render(
      <VRAMBar
        node={testNode}
        label={{ prefix: 'GPU 0', main: 'test gpu' }}
        totalGB={4}
        reservedGB={1}
        containerIdx={0}
        assigns={[]}
        onPick={vi.fn()}
        onSelectContainer={vi.fn()}
        setAssigns={setAssigns}
        dragOver={null}
        setDragOver={vi.fn()}
      />,
    )

    const reservedLane = screen.getByTitle('Reserved 1 GB')

    fireEvent.dragEnter(reservedLane, { dataTransfer })
    fireEvent.dragOver(reservedLane, { dataTransfer })
    fireEvent.drop(reservedLane, { dataTransfer })

    expect(dataTransfer.dropEffect).toBe('none')
    expect(setAssigns).not.toHaveBeenCalled()
  })
})
