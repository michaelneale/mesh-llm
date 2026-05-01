import { render } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ShouldBlockFn } from '@tanstack/react-router'

const mockUseBlocker = vi.hoisted(() => vi.fn())

vi.mock('@tanstack/react-router', () => ({
  useBlocker: mockUseBlocker,
}))

import { UnsavedConfigurationNavigationBlocker } from '@/features/configuration/pages/ConfigurationPageNavigationBlocker'

function blockerArgs(currentPathname: string, nextPathname: string): Parameters<ShouldBlockFn>[0] {
  return {
    action: 'PUSH',
    current: { routeId: '/configuration/$configurationTab', fullPath: '/configuration/$configurationTab', pathname: currentPathname, params: { configurationTab: 'defaults' }, search: {} },
    next: { routeId: '/configuration/$configurationTab', fullPath: '/configuration/$configurationTab', pathname: nextPathname, params: { configurationTab: 'toml-review' }, search: {} },
  }
}

describe('UnsavedConfigurationNavigationBlocker', () => {
  let shouldBlockFn: ShouldBlockFn | undefined

  beforeEach(() => {
    shouldBlockFn = undefined
    mockUseBlocker.mockImplementation((options: { shouldBlockFn: ShouldBlockFn }) => {
      shouldBlockFn = options.shouldBlockFn
      return { status: 'idle' }
    })
  })

  it('allows tab-to-tab configuration route changes while dirty', () => {
    render(<UnsavedConfigurationNavigationBlocker hasUnsavedChanges />)

    expect(shouldBlockFn?.(blockerArgs('/configuration/defaults', '/configuration/toml-review'))).toBe(false)
  })

  it('blocks leaving configuration while dirty', () => {
    render(<UnsavedConfigurationNavigationBlocker hasUnsavedChanges />)

    expect(shouldBlockFn?.(blockerArgs('/configuration/toml-review', '/chat'))).toBe(true)
  })

  it('does not block when configuration is clean', () => {
    render(<UnsavedConfigurationNavigationBlocker hasUnsavedChanges={false} />)

    expect(shouldBlockFn?.(blockerArgs('/configuration/toml-review', '/chat'))).toBe(false)
  })
})
