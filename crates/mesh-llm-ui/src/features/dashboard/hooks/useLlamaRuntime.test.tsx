// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useLlamaRuntime } from '@/features/dashboard/hooks/useLlamaRuntime'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('useLlamaRuntime', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  })

  it('does not surface expected aborts from overlapping polls', async () => {
    const requests: DeferredFetch[] = []
    const fetchMock = vi.fn((_: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
      const deferred = createDeferredFetch(init?.signal)
      requests.push(deferred)
      return deferred.promise
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<RuntimeProbe />)
    expect(screen.getByTestId('loading')).toHaveTextContent('loading')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_500)
    })

    expect(requests).toHaveLength(2)
    await act(async () => {
      requests[1].resolve(jsonResponse({ metrics: { status: 'ready' }, slots: { status: 'ready' } }))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(screen.getByTestId('loading')).toHaveTextContent('idle')
    expect(screen.getByTestId('error')).toHaveTextContent('none')
  })
})

function RuntimeProbe() {
  const runtime = useLlamaRuntime(true)
  return (
    <div>
      <div data-testid="loading">{runtime.loading ? 'loading' : 'idle'}</div>
      <div data-testid="error">{runtime.error ?? 'none'}</div>
      <div data-testid="metrics-status">{runtime.data?.metrics.status ?? 'none'}</div>
    </div>
  )
}

type DeferredFetch = {
  promise: Promise<Response>
  resolve: (response: Response) => void
  reject: (error: Error) => void
}

function createDeferredFetch(signal: AbortSignal | null | undefined): DeferredFetch {
  let resolve: (response: Response) => void = () => undefined
  let reject: (error: Error) => void = () => undefined
  const promise = new Promise<Response>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })
  signal?.addEventListener(
    'abort',
    () => {
      reject(new DOMException('The operation was aborted.', 'AbortError'))
    },
    { once: true }
  )
  return { promise, resolve, reject }
}

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    headers: { 'content-type': 'application/json' }
  })
}
