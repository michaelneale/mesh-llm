import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { CommandBarModal, type CommandBarModalProps } from '@/features/app-shell/components/command-bar/CommandBarModal'
import { CommandBarProvider } from '@/features/app-shell/components/command-bar/CommandBarProvider'
import type { CommandBarMode } from '@/features/app-shell/components/command-bar/command-bar-types'
import { useCommandBar } from '@/features/app-shell/components/command-bar/useCommandBar'

type TestItem = {
  id: string
  label: string
  keywords?: readonly string[]
}

function TestIcon(props: React.ComponentProps<'svg'>) {
  return <svg {...props} />
}

function OpenCommandBarButton() {
  const { openCommandBar } = useCommandBar()

  return (
    <button type="button" onClick={() => openCommandBar()}>
      Open command bar
    </button>
  )
}

function createMode({
  id,
  label,
  source,
  onSelect = vi.fn()
}: {
  id: string
  label: string
  source: CommandBarMode<TestItem>['source']
  onSelect?: CommandBarMode<TestItem>['onSelect']
}): CommandBarMode<TestItem> {
  return {
    id,
    label,
    leadingIcon: TestIcon,
    source,
    getItemKey: (item) => item.id,
    getSearchText: (item) => item.label,
    getKeywords: (item) => item.keywords ?? [],
    onSelect
  }
}

function renderCommandBar(
  props: Partial<CommandBarModalProps<TestItem>> & { modes: readonly CommandBarMode<TestItem>[] }
) {
  render(
    <CommandBarProvider>
      <OpenCommandBarButton />
      <CommandBarModal<TestItem>
        behavior="distinct"
        title="Command bar"
        description="Search available results."
        {...props}
      />
    </CommandBarProvider>
  )

  return {
    openButton: screen.getByRole('button', { name: 'Open command bar' }) as HTMLButtonElement
  }
}

async function openCommandBar(user: ReturnType<typeof userEvent.setup>) {
  const openButton = screen.getByRole('button', { name: 'Open command bar' }) as HTMLButtonElement
  await user.click(openButton)
  await screen.findByRole('textbox', { name: 'Command bar search' })
  return openButton
}

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })

  return { promise, resolve, reject }
}

afterEach(() => {
  vi.useRealTimers()
})

describe('CommandBarModal', () => {
  it('opens with the expected shell and distinct-mode defaults', async () => {
    const user = userEvent.setup()

    renderCommandBar({
      behavior: 'distinct',
      defaultModeId: 'actions',
      interstitial: <div>Helpful tip</div>,
      modes: [
        createMode({
          id: 'models',
          label: 'Models',
          source: [{ id: 'phi', label: 'Phi-4 Mini' }]
        }),
        createMode({
          id: 'actions',
          label: 'Actions',
          source: [{ id: 'reload', label: 'Reload settings' }]
        })
      ],
      title: 'Model catalog'
    })

    await openCommandBar(user)

    expect(screen.getByRole('dialog', { name: 'Model catalog' })).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Search actions')).toBeInTheDocument()
    expect(screen.getByTestId('command-bar-interstitial')).toHaveTextContent('Helpful tip')
    expect(screen.getByRole('button', { name: /^Models/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Actions/ })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('option', { name: 'Reload settings' })).toBeInTheDocument()
  })

  it('filters and sorts results using prefix and keyword matches', async () => {
    const user = userEvent.setup()

    renderCommandBar({
      behavior: 'distinct',
      modes: [
        createMode({
          id: 'models',
          label: 'Models',
          source: [
            { id: 'alpha', label: 'Alpha' },
            { id: 'beta', label: 'Beta', keywords: ['alpine'] },
            { id: 'zalpha', label: 'Zalpha' }
          ]
        })
      ]
    })

    await openCommandBar(user)
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'alp')

    await waitFor(() => {
      expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual(['Alpha', 'Beta', 'Zalpha'])
    })
  })

  it('switches modes with the shortcut and resets selection to the first result', async () => {
    const user = userEvent.setup()

    renderCommandBar({
      behavior: 'distinct',
      modes: [
        createMode({
          id: 'models',
          label: 'Models',
          source: [
            { id: 'phi', label: 'Phi-4 Mini' },
            { id: 'qwen', label: 'Qwen 3' }
          ]
        }),
        createMode({
          id: 'actions',
          label: 'Actions',
          source: [
            { id: 'reload', label: 'Reload settings' },
            { id: 'save', label: 'Save configuration' }
          ]
        })
      ]
    })

    await openCommandBar(user)
    await user.keyboard('{ArrowDown}')
    expect(screen.getByRole('option', { name: 'Qwen 3' })).toHaveAttribute('aria-selected', 'true')

    fireEvent.keyDown(screen.getByRole('textbox', { name: 'Command bar search' }), {
      key: '2',
      ctrlKey: true,
      bubbles: true
    })

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search actions')).toBeInTheDocument()
      expect(screen.getByRole('option', { name: 'Reload settings' })).toHaveAttribute('aria-selected', 'true')
      expect(screen.getByRole('button', { name: /^Actions/ })).toHaveAttribute('aria-pressed', 'true')
    })
  })

  it('supports arrow navigation and Enter selection from the search input', async () => {
    const user = userEvent.setup()
    const onSelect = vi.fn(() => false)

    renderCommandBar({
      behavior: 'distinct',
      modes: [
        createMode({
          id: 'models',
          label: 'Models',
          onSelect,
          source: [
            { id: 'phi', label: 'Phi-4 Mini' },
            { id: 'qwen', label: 'Qwen 3' }
          ]
        })
      ]
    })

    await openCommandBar(user)
    await user.keyboard('{ArrowDown}{Enter}')

    expect(screen.getByRole('option', { name: 'Qwen 3' })).toHaveAttribute('aria-selected', 'true')
    expect(onSelect).toHaveBeenCalledWith({ id: 'qwen', label: 'Qwen 3' })
    expect(screen.getByRole('dialog', { name: 'Command bar' })).toBeInTheDocument()
  })

  it('shows async loading and then renders loaded results', async () => {
    const user = userEvent.setup()
    const deferred = createDeferred<readonly TestItem[]>()

    renderCommandBar({
      behavior: 'distinct',
      modes: [
        createMode({
          id: 'models',
          label: 'Models',
          source: vi.fn(() => deferred.promise)
        })
      ]
    })

    await openCommandBar(user)

    expect(screen.getByText('Loading results')).toBeInTheDocument()

    await act(async () => {
      deferred.resolve([{ id: 'phi', label: 'Phi-4 Mini' }])
    })

    expect(await screen.findByRole('option', { name: 'Phi-4 Mini' })).toBeInTheDocument()
    expect(screen.queryByText('Loading results')).not.toBeInTheDocument()
  })

  it('shows async load errors when a mode request fails', async () => {
    const user = userEvent.setup()

    renderCommandBar({
      behavior: 'distinct',
      modes: [
        createMode({
          id: 'models',
          label: 'Models',
          source: vi.fn(async () => {
            throw new Error('Network unavailable')
          })
        })
      ]
    })

    await openCommandBar(user)

    expect(await screen.findByText('Could not load results')).toBeInTheDocument()
    expect(screen.getByText('Network unavailable')).toBeInTheDocument()
  })

  it('ignores stale async results after a newer debounced request starts', async () => {
    vi.useFakeTimers()
    const deferredByQuery = new Map<string, ReturnType<typeof createDeferred<readonly TestItem[]>>>()
    const signalByQuery = new Map<string, AbortSignal>()

    const { openButton } = renderCommandBar({
      behavior: 'distinct',
      modes: [
        createMode({
          id: 'models',
          label: 'Models',
          source: vi.fn((query, signal) => {
            const deferred = createDeferred<readonly TestItem[]>()
            deferredByQuery.set(query, deferred)
            signalByQuery.set(query, signal)
            return deferred.promise
          })
        })
      ]
    })

    fireEvent.click(openButton)
    const input = screen.getByRole('textbox', { name: 'Command bar search' })

    fireEvent.change(input, { target: { value: 'a' } })
    await act(async () => {
      vi.advanceTimersByTime(150)
    })
    expect(deferredByQuery.has('a')).toBe(true)
    expect(signalByQuery.get('a')?.aborted).toBe(false)

    fireEvent.change(input, { target: { value: 'ab' } })
    expect(signalByQuery.get('a')?.aborted).toBe(true)

    await act(async () => {
      vi.advanceTimersByTime(150)
    })
    expect(deferredByQuery.has('ab')).toBe(true)

    await act(async () => {
      deferredByQuery.get('a')?.resolve([{ id: 'alpha', label: 'Alpha result' }])
      deferredByQuery.get('ab')?.resolve([{ id: 'abacus', label: 'Abacus result' }])
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(screen.getByRole('option', { name: 'Abacus result' })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: 'Alpha result' })).not.toBeInTheDocument()
  })

  it('renders selection errors when onSelect fails', async () => {
    const user = userEvent.setup()

    renderCommandBar({
      behavior: 'distinct',
      modes: [
        createMode({
          id: 'models',
          label: 'Models',
          onSelect: () => {
            throw new Error('Selection failed')
          },
          source: [{ id: 'phi', label: 'Phi-4 Mini' }]
        })
      ]
    })

    await openCommandBar(user)
    await user.keyboard('{Enter}')

    expect(await screen.findByRole('alert')).toHaveTextContent('Selection failed')
    expect(screen.getByRole('dialog', { name: 'Command bar' })).toBeInTheDocument()
  })

  it('closes and restores focus to the opener on Escape', async () => {
    const user = userEvent.setup()

    const { openButton } = renderCommandBar({
      behavior: 'distinct',
      modes: [
        createMode({
          id: 'models',
          label: 'Models',
          source: [{ id: 'phi', label: 'Phi-4 Mini' }]
        })
      ]
    })

    await openCommandBar(user)
    await user.keyboard('{Escape}')

    await waitFor(() => {
      expect(screen.queryByRole('dialog', { name: 'Command bar' })).not.toBeInTheDocument()
      expect(openButton).toHaveFocus()
    })
  })
})
