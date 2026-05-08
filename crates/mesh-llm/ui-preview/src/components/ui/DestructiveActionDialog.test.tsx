import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useRef, useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { DestructiveActionDialog } from '@/components/ui/DestructiveActionDialog'

function DestructiveActionDialogHarness({ onConfirm = vi.fn() }: { onConfirm?: () => void }) {
  const [open, setOpen] = useState(false)
  const openButtonRef = useRef<HTMLButtonElement | null>(null)

  return (
    <>
      <button ref={openButtonRef} type="button" onClick={() => setOpen(true)}>
        Request delete
      </button>
      <DestructiveActionDialog
        open={open}
        onOpenChange={setOpen}
        title="Delete this chat?"
        description="This permanently removes the selected chat and its message history."
        destructiveLabel="Delete chat"
        onConfirm={onConfirm}
        returnFocusRef={openButtonRef}
      />
    </>
  )
}

describe('DestructiveActionDialog', () => {
  it('opens as an alert dialog with initial focus on the destructive action', async () => {
    const user = userEvent.setup()

    render(<DestructiveActionDialogHarness />)
    const openButton = screen.getByRole('button', { name: 'Request delete' })
    await user.click(openButton)

    expect(screen.getByRole('alertdialog', { name: 'Delete this chat?' })).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Delete chat' })).toHaveFocus()
    })
  })

  it('supports keyboard tab order, Escape close, and focus restoration', async () => {
    const user = userEvent.setup()

    render(<DestructiveActionDialogHarness />)
    const openButton = screen.getByRole('button', { name: 'Request delete' })
    await user.click(openButton)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Delete chat' })).toHaveFocus())

    await user.tab()
    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus()

    await user.tab()
    expect(screen.getByRole('button', { name: 'Delete chat' })).toHaveFocus()

    await user.tab()
    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus()

    await user.tab({ shift: true })
    expect(screen.getByRole('button', { name: 'Delete chat' })).toHaveFocus()

    await user.keyboard('{Escape}')
    await waitFor(() => {
      expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
      expect(openButton).toHaveFocus()
    })
  })

  it('runs the destructive action from the keyboard', async () => {
    const user = userEvent.setup()
    const onConfirm = vi.fn()
    const onWindowKeyDown = vi.fn((event: KeyboardEvent) => event.preventDefault())
    window.addEventListener('keydown', onWindowKeyDown)

    try {
      render(<DestructiveActionDialogHarness onConfirm={onConfirm} />)
      await user.click(screen.getByRole('button', { name: 'Request delete' }))
      await waitFor(() => expect(screen.getByRole('button', { name: 'Delete chat' })).toHaveFocus())
      onWindowKeyDown.mockClear()
      await user.keyboard('{Enter}')

      expect(onWindowKeyDown).not.toHaveBeenCalled()
      expect(onConfirm).toHaveBeenCalledTimes(1)
      await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument())
    } finally {
      window.removeEventListener('keydown', onWindowKeyDown)
    }
  })
})
