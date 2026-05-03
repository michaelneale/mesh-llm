import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useClipboardCopy } from '@/lib/useClipboardCopy'

function installClipboard(writeText: (text: string) => Promise<void>) {
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText },
  })
}

describe('useClipboardCopy', () => {
  afterEach(() => {
    vi.useRealTimers()
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined })
  })

  it('copies text and resets copied state', async () => {
    vi.useFakeTimers()
    const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)
    installClipboard(writeText)

    const { result } = renderHook(() => useClipboardCopy({ resetDelayMs: 25 }))

    await act(async () => {
      await expect(result.current.copyText('mesh command')).resolves.toBe(true)
    })

    expect(writeText).toHaveBeenCalledWith('mesh command')
    expect(result.current.copyState).toBe('copied')

    act(() => {
      vi.advanceTimersByTime(25)
    })

    expect(result.current.copyState).toBe('idle')
  })

  it('reports failed state when clipboard write is unavailable or rejected', async () => {
    vi.useFakeTimers()
    const writeText = vi.fn<(text: string) => Promise<void>>().mockRejectedValue(new DOMException('Blocked', 'NotAllowedError'))
    installClipboard(writeText)

    const { result } = renderHook(() => useClipboardCopy({ resetDelayMs: 25 }))

    await act(async () => {
      await expect(result.current.copyText('mesh command')).resolves.toBe(false)
    })

    expect(result.current.copyState).toBe('failed')

    act(() => {
      vi.advanceTimersByTime(25)
    })

    expect(result.current.copyState).toBe('idle')
  })
})
