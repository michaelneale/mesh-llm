import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it } from 'vitest'
import { DATA_MODE_STORAGE_KEY, DataModeProvider } from '@/lib/data-mode/DataModeContext'
import { useDataMode } from '@/lib/data-mode/useDataMode'

function providerWrapper(props: { initialMode?: 'live' | 'harness'; persist?: boolean; storageKey?: string } = {}) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <DataModeProvider {...props}>{children}</DataModeProvider>
  }
}

describe('DataModeProvider', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('defaults to harness mode and persists under the preview namespace', async () => {
    const { result } = renderHook(() => useDataMode(), { wrapper: providerWrapper() })

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
