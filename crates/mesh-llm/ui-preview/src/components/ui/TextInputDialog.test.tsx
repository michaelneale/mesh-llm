import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useRef, useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { TextInputDialog } from '@/components/ui/TextInputDialog'

function TextInputDialogHarness({
  initialValue = '',
  onSave = vi.fn()
}: {
  initialValue?: string
  onSave?: (value: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [value, setValue] = useState(initialValue)
  const openButtonRef = useRef<HTMLButtonElement | null>(null)

  return (
    <>
      <button ref={openButtonRef} type="button" onClick={() => setOpen(true)}>
        Open system prompt
      </button>
      <TextInputDialog
        open={open}
        onOpenChange={setOpen}
        title="Set system prompt"
        description="Saved instructions are sent before each chat message."
        label="System prompt"
        value={value}
        onValueChange={setValue}
        onSave={onSave}
        placeholder="You are a careful mesh-llm operator."
        saveLabel="Save prompt"
        returnFocusRef={openButtonRef}
      />
    </>
  )
}

describe('TextInputDialog', () => {
  it('opens with a large textarea, focused input, and live character counter', async () => {
    const user = userEvent.setup()

    render(<TextInputDialogHarness initialValue="Use short answers." />)
    await user.click(screen.getByRole('button', { name: 'Open system prompt' }))

    expect(screen.getByRole('dialog', { name: 'Set system prompt' })).toBeInTheDocument()
    expect(screen.getByText('Saved instructions are sent before each chat message.')).toHaveClass(
      'max-w-none',
      'text-wrap'
    )
    const textarea = screen.getByLabelText('System prompt')
    await waitFor(() => expect(textarea).toHaveFocus())
    expect(textarea).toHaveValue('Use short answers.')
    expect(textarea).toHaveStyle({ minHeight: '180px' })
    expect(screen.getByText('18 characters')).toBeInTheDocument()

    await user.type(textarea, ' Keep examples concrete.')

    expect(screen.getByText('42 characters')).toBeInTheDocument()
  })

  it('saves the edited value and restores focus to the opener', async () => {
    const user = userEvent.setup()
    const onSave = vi.fn()
    const onWindowKeyDown = vi.fn((event: KeyboardEvent) => event.preventDefault())
    window.addEventListener('keydown', onWindowKeyDown)

    try {
      render(<TextInputDialogHarness onSave={onSave} />)
      const openButton = screen.getByRole('button', { name: 'Open system prompt' })
      await user.click(openButton)

      onWindowKeyDown.mockClear()
      await user.type(screen.getByLabelText('System prompt'), 'Prefer terse routing explanations.')
      expect(onWindowKeyDown).not.toHaveBeenCalled()

      await user.click(screen.getByRole('button', { name: 'Save prompt' }))

      expect(onSave).toHaveBeenCalledWith('Prefer terse routing explanations.')
      await waitFor(() => {
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
        expect(openButton).toHaveFocus()
      })
    } finally {
      window.removeEventListener('keydown', onWindowKeyDown)
    }
  })

  it('cancels without saving and restores focus', async () => {
    const user = userEvent.setup()
    const onSave = vi.fn()

    render(<TextInputDialogHarness onSave={onSave} />)
    const openButton = screen.getByRole('button', { name: 'Open system prompt' })
    await user.click(openButton)

    await user.type(screen.getByLabelText('System prompt'), 'Draft only')
    await user.click(screen.getByRole('button', { name: 'Cancel' }))

    expect(onSave).not.toHaveBeenCalled()
    await waitFor(() => expect(openButton).toHaveFocus())
  })
})
