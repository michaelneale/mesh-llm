import { describe, expect, it } from 'vitest'
import type { MeshNode } from '@/features/app-tabs/types'
import { chooseClusteredMeshNodePosition } from './mesh-placement'

const CANVAS_WIDTH = 800
const CANVAS_HEIGHT = 420
const MIN_VISUAL_NODE_GAP_PX = 64

function meshNode(id: string, x: number, y: number): MeshNode {
  return {
    id,
    label: id,
    x,
    y,
    status: 'online',
    renderKind: 'worker'
  }
}

function nearestScreenDistance(position: Pick<MeshNode, 'x' | 'y'>, nodes: MeshNode[]) {
  return Math.min(
    ...nodes.map((node) =>
      Math.hypot(((position.x - node.x) / 100) * CANVAS_WIDTH, ((position.y - node.y) / 100) * CANVAS_HEIGHT)
    )
  )
}

describe('chooseClusteredMeshNodePosition', () => {
  it('keeps generated nodes out of the visual halo danger zone during mesh growth', () => {
    const nodes = [
      meshNode('self', 50, 50),
      meshNode('northwest', 38, 36),
      meshNode('northeast', 64, 39),
      meshNode('southwest', 35, 68),
      meshNode('southeast', 66, 70)
    ]

    for (let index = 0; index < 40; index += 1) {
      const position = chooseClusteredMeshNodePosition(
        'mesh-overlap-regression',
        index + 1,
        { renderKind: 'worker' },
        nodes
      )
      const distance = nearestScreenDistance(position, nodes)

      expect(distance).toBeGreaterThanOrEqual(MIN_VISUAL_NODE_GAP_PX)

      nodes.push(meshNode(`generated-${index}`, position.x, position.y))
    }
  })
})
