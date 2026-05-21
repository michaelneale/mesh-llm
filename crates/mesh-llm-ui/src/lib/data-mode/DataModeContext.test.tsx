import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { DATA_MODE_STORAGE_KEY, DataModeProvider } from '@/lib/data-mode/DataModeContext'
import { useDataMode } from '@/lib/data-mode/useDataMode'
import { env } from '@/lib/env'

function providerWrapper(props: { initialMode?: 'live' | 'harness'; persist?: boolean; storageKey?: string } = {}) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <DataModeProvider {...props}>{children}</DataModeProvider>
  }
}

describe('DataModeProvider', () => {
  const originalIsDevelopment = env.isDevelopment

  beforeEach(() => {
    window.localStorage.clear()
  })

  afterEach(() => {
    env.isDevelopment = originalIsDevelopment
  })

  it('defaults to live mode in production builds and persists under the preview namespace', async () => {
    // Regression: the production console (local mesh-llm, fly app) should
    // show live mesh data by default, not fixture providers. Harness mode
    // is an explicit opt-in for the developer playground / mockups via the
    // in-app data-mode toggle.
    env.isDevelopment = false

    const { result } = renderHook(() => useDataMode(), { wrapper: providerWrapper() })

    expect(result.current.mode).toBe('live')

    await waitFor(() => {
      expect(window.localStorage.getItem(DATA_MODE_STORAGE_KEY)).toBe('live')
    })
  })

  it('defaults to harness mode in dev builds (npm run dev) for designer iteration', async () => {
    env.isDevelopment = true

    const { result } = renderHook(() => useDataMode(), { wrapper: providerWrapper() })

    expect(result.current.mode).toBe('harness')

    await waitFor(() => {
      expect(window.localStorage.getItem(DATA_MODE_STORAGE_KEY)).toBe('harness')
    })
  })

  it('still honours an explicit harness initialMode (developer playground)', async () => {
    const { result } = renderHook(() => useDataMode(), {
      wrapper: providerWrapper({ initialMode: 'harness' })
    })

    expect(result.current.mode).toBe('harness')

    await waitFor(() => {
      expect(window.localStorage.getItem(DATA_MODE_STORAGE_KEY)).toBe('harness')
    })
  })

  it('hydrates from a valid stored data mode', () => {
    window.localStorage.setItem(DATA_MODE_STORAGE_KEY, 'live')

    const { result } = renderHook(() => useDataMode(), { wrapper: providerWrapper() })

    expect(result.current.mode).toBe('live')
  })

  it('persists data mode updates', async () => {
    const { result } = renderHook(() => useDataMode(), { wrapper: providerWrapper() })

    act(() => {
      result.current.setMode('live')
    })

    await waitFor(() => {
      expect(result.current.mode).toBe('live')
      expect(window.localStorage.getItem(DATA_MODE_STORAGE_KEY)).toBe('live')
    })
  })

  it('can opt out of persistence for embedded hosts', () => {
    const storageKey = 'host-owned:data-mode'

    const { result } = renderHook(() => useDataMode(), {
      wrapper: providerWrapper({ initialMode: 'live', persist: false, storageKey })
    })

    expect(result.current.mode).toBe('live')
    expect(window.localStorage.getItem(storageKey)).toBeNull()
  })
})
