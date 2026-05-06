import { describe, expect, it } from 'vitest'
import { createAssignmentId } from '@/features/configuration/lib/assignment-ids'
import type { ConfigAssign } from '@/features/app-tabs/types'

describe('createAssignmentId', () => {
  it('returns the next deterministic assignment id', () => {
    const assigns: ConfigAssign[] = [
      { id: 'a-1', modelId: 'qwen4', nodeId: 'carrack', containerIdx: 0, ctx: 4096 },
      { id: 'a-2', modelId: 'phi4', nodeId: 'carrack', containerIdx: 1, ctx: 4096 }
    ]

    expect(createAssignmentId(assigns)).toBe('a-3')
  })

  it('skips occupied ids instead of relying on timestamps', () => {
    const assigns: ConfigAssign[] = [
      { id: 'a-1', modelId: 'qwen4', nodeId: 'carrack', containerIdx: 0, ctx: 4096 },
      { id: 'a-3', modelId: 'phi4', nodeId: 'carrack', containerIdx: 1, ctx: 4096 }
    ]

    expect(createAssignmentId(assigns)).toBe('a-4')
  })
})
