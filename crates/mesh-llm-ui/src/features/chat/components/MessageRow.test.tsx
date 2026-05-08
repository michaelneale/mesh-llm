import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { MessageRow } from '@/features/chat/components/MessageRow'

describe('MessageRow', () => {
  it('renders non-inspectable messages as static articles with row visuals', () => {
    const { container } = render(
      <MessageRow
        messageRole="assistant"
        body="Static response from the mesh"
        timestamp="12:09"
        model="llama-local-q4"
        route="carrack.mesh"
        tokens="256 tokens"
      />
    )

    const article = container.querySelector('article')

    expect(article).toBeInTheDocument()
    expect(article).toHaveClass('relative', '-mx-2', 'mb-5', 'block', 'w-[calc(100%+16px)]', 'select-none')
    expect(article).not.toHaveClass('focus-visible:outline-accent')
    expect(article).toHaveTextContent('Static response from the mesh')
    expect(container.querySelector('button')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /static response from the mesh/i })).not.toBeInTheDocument()
  })

  it('keeps assistant model in the header and renders the complete response stats bar', () => {
    render(
      <MessageRow
        messageRole="assistant"
        body="Response metadata should read like an operations ledger"
        timestamp="2026-05-06 14:52:31 UTC"
        model="llama-local-q4"
        tokens="27 tok"
        tokPerSec="15.3 tok/s"
        ttft="1116ms"
      />
    )

    const header = screen.getByText('Assistant').parentElement
    const stats = screen.getByText('27').closest('.select-none')

    expect(header).toHaveTextContent('Assistant')
    expect(header).toHaveTextContent('llama-local-q4')
    expect(header).not.toHaveTextContent('14:52')
    expect(stats).toHaveTextContent('27 tok')
    expect(stats).toHaveTextContent('15.3 tok/s')
    expect(stats).toHaveTextContent('TTFT 1116ms')
  })

  it('shows the submitted model in user headers without rendering timestamps or response stats', () => {
    render(
      <MessageRow
        messageRole="user"
        body="Route this prompt through the local node"
        timestamp="2026-05-06T20:17:00.212Z"
        model="GLM-4.7-Flash-Q4_K_M"
      />
    )

    const header = screen.getByText('You').parentElement

    expect(header).toHaveTextContent('GLM-4.7-Flash-Q4_K_M')
    expect(header).not.toHaveTextContent('2026-05-06T20:17:00.212Z')
    expect(screen.queryByText('0.0 tok/s')).not.toBeInTheDocument()
  })

  it('renders inspectable messages as labeled buttons and preserves the inspect action', async () => {
    const user = userEvent.setup()
    const inspect = vi.fn()

    render(
      <MessageRow
        messageRole="assistant"
        body="Inspectable response from the mesh"
        timestamp="12:10"
        inspect={inspect}
        inspectLabel="Inspect mesh route"
      />
    )

    const button = screen.getByRole('button', { name: 'Inspect mesh route' })

    expect(button).toHaveClass(
      'relative',
      '-mx-2',
      'mb-5',
      'block',
      'w-[calc(100%+16px)]',
      'select-none',
      'focus-visible:outline-accent'
    )
    expect(button).toHaveTextContent('Inspectable response from the mesh')

    await user.click(button)

    expect(inspect).toHaveBeenCalledTimes(1)
  })

  it('keeps inspectable rows accessible when no explicit inspect label is provided', () => {
    render(
      <MessageRow
        messageRole="user"
        body="Route this prompt through the local node"
        timestamp="12:11"
        inspect={() => {}}
      />
    )

    expect(screen.getByRole('button', { name: 'Inspect user message from 12:11' })).toBeInTheDocument()
  })

  it('renders submitted attachment actions beside user messages without nesting row inspection', async () => {
    const user = userEvent.setup()
    const inspect = vi.fn()
    const openAttachment = vi.fn()
    const { container } = render(
      <MessageRow
        messageRole="user"
        body="Describe this image to me"
        timestamp="12:11"
        inspect={inspect}
        inspectLabel="Inspect submitted image prompt"
        attachments={[
          {
            id: 'image-1',
            label: 'Image 1',
            kind: 'image',
            fileName: 'cat.png',
            onOpen: openAttachment
          }
        ]}
      />
    )

    expect(container.querySelector('article')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Inspect submitted image prompt' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Open cat.png' }))

    expect(openAttachment).toHaveBeenCalledTimes(1)
    expect(inspect).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Inspect submitted image prompt' }))

    expect(inspect).toHaveBeenCalledTimes(1)
  })

  it('renders user messages on a subtly dimmed surface', () => {
    render(<MessageRow messageRole="user" body="Route this prompt through the local node" timestamp="12:11" />)

    const userSurface = screen
      .getByText('Route this prompt through the local node')
      .closest('[style*="chat-user-message-background"]')

    expect(userSurface).toHaveAttribute(
      'style',
      expect.stringContaining('background: var(--chat-user-message-background)')
    )
    expect(userSurface).toHaveAttribute('style', expect.stringContaining('border-left: 1px solid var(--color-accent)'))
    expect(userSurface).not.toHaveAttribute('style', expect.stringContaining('border-radius'))
  })

  it('marks queued user messages with a queued label and muted leading rule', () => {
    render(<MessageRow messageRole="user" body="Queued follow-up" timestamp="12:11" state="queued" />)

    const queuedSurface = screen.getByText('Queued follow-up').closest('[style*="border-left"]')

    expect(screen.getByText('Queued')).toBeInTheDocument()
    expect(queuedSurface).toHaveAttribute(
      'style',
      expect.stringContaining(
        'border-left: 1px solid color-mix(in oklab, var(--color-fg-faint) 42%, var(--color-border))'
      )
    )
  })

  it('centers a queued-message remove control vertically in the full user message container', async () => {
    const user = userEvent.setup()
    const removeQueued = vi.fn()

    render(
      <MessageRow
        messageRole="user"
        body="Queued follow-up"
        timestamp="12:11"
        state="queued"
        onRemoveQueued={removeQueued}
      />
    )

    const removeButton = screen.getByRole('button', { name: 'Remove queued message' })
    const queuedSurface = removeButton.closest('[style*="chat-user-message-background"]')

    expect(queuedSurface).toHaveClass('relative')
    expect(removeButton).toHaveClass('absolute', 'right-3', 'top-1/2', '-translate-y-1/2')
    expect(removeButton).not.toHaveClass('left-1/2', '-translate-x-1/2')

    await user.click(removeButton)

    expect(removeQueued).toHaveBeenCalledTimes(1)
  })

  it('renders a spinner row while an assistant response is waiting for first text', () => {
    render(<MessageRow messageRole="assistant" body="" timestamp="12:11" model="Qwen3-8B" state="streaming" />)

    expect(screen.getByText('Assistant').parentElement).toHaveTextContent('Qwen3-8B')
    expect(screen.getByText('Streaming response...')).toBeInTheDocument()
  })

  it('places the streaming control on its own row below streamed text', () => {
    render(
      <MessageRow
        messageRole="assistant"
        body="Partial streamed response text"
        timestamp="12:11"
        model="Qwen3-8B"
        state="streaming"
      />
    )

    const streamingControlRow = screen.getByRole('button', { name: 'Stop streaming' }).parentElement

    expect(streamingControlRow).toHaveClass('mt-3', 'block')
  })

  it('separates closed thinking traces from final assistant response text', () => {
    render(
      <MessageRow
        messageRole="assistant"
        body="<think>Check geography facts before answering.</think> The capital of France is Paris."
        timestamp="12:11"
        model="Qwen3-8B"
      />
    )

    expect(screen.getByText('Thinking trace').closest('[data-thinking-state="complete"]')).toHaveTextContent(
      'Thinking trace'
    )
    expect(screen.getByText('Check geography facts before answering.')).toHaveClass('select-text')
    expect(screen.getByText('The capital of France is Paris.').closest('.select-text')).toBeInTheDocument()
  })

  it('treats leading streamed text before a closing think tag as model thinking', () => {
    render(
      <MessageRow
        messageRole="assistant"
        body="The user asked a simple factual question.</think> The capital of France is Paris."
        timestamp="12:11"
        model="Qwen3-8B"
      />
    )

    expect(screen.getByText('Thinking trace').closest('[data-thinking-state="complete"]')).toHaveTextContent(
      'The user asked a simple factual question.'
    )
    expect(screen.getByText('The capital of France is Paris.')).toBeInTheDocument()
  })

  it('formats final assistant response text as markdown', () => {
    render(
      <MessageRow
        messageRole="assistant"
        body={'<think>Check geography facts.</think> The capital of France is **Paris**.\n\n- *Verified*\n- `Concise`'}
        timestamp="12:11"
        model="Qwen3-8B"
      />
    )

    expect(screen.getByText('Paris').tagName.toLowerCase()).toBe('strong')
    expect(screen.getByText('Verified').tagName.toLowerCase()).toBe('em')
    expect(screen.getByText('Concise').tagName.toLowerCase()).toBe('code')
    expect(screen.getByText('Paris').closest('.select-text')).toBeInTheDocument()
  })

  it('marks an open thinking segment as in progress while streaming', () => {
    render(
      <MessageRow
        messageRole="assistant"
        body="<think>Checking the current capital from memory"
        timestamp="12:11"
        model="Qwen3-8B"
        state="streaming"
      />
    )

    expect(screen.getByText('Thinking').closest('[data-thinking-state="active"]')).toHaveTextContent('Thinking')
    expect(screen.getByText('Checking the current capital from memory')).toHaveClass('select-text')
    expect(screen.getByText('Streaming response...')).toBeInTheDocument()
  })

  it('keeps streamed reasoning without an opening tag inside the active thinking container', () => {
    render(
      <MessageRow
        messageRole="assistant"
        body="The user asked for a long story, so plan the structure first."
        timestamp="12:11"
        model="Qwen3-8B"
        state="streaming"
      />
    )

    const thinkingContainer = screen.getByText('Thinking').closest('[data-thinking-state="active"]')

    expect(thinkingContainer).toHaveTextContent('The user asked for a long story, so plan the structure first.')
    expect(screen.getByText('The user asked for a long story, so plan the structure first.')).toHaveClass('select-text')
    expect(screen.getByText('Streaming response...')).toBeInTheDocument()
  })

  it('shows the active thinking container as soon as the think tag streams in', () => {
    render(<MessageRow messageRole="assistant" body="<think>" timestamp="12:11" model="Qwen3-8B" state="streaming" />)

    const thinkingContainer = screen.getByText('Thinking').closest('[data-thinking-state="active"]')

    expect(thinkingContainer).toBeInTheDocument()
    expect(thinkingContainer).toHaveTextContent('Thinking')
    expect(screen.getByText('Streaming response...')).toBeInTheDocument()
  })

  it('only opts the message body back into text selection', () => {
    const { container } = render(
      <MessageRow
        messageRole="assistant"
        body="Only this response text should be selectable"
        timestamp="12:12"
        model="llama-local-q4"
        route="carrack.mesh"
        tokens="256 tokens"
      />
    )

    expect(container.querySelector('article')).toHaveClass('select-none')
    expect(screen.getByText('Only this response text should be selectable').closest('.select-text')).toBeInTheDocument()
    expect(screen.getByText('Assistant').parentElement).toHaveClass('select-none')
    expect(screen.getByText('256').closest('.select-none')).toHaveClass('select-none')
  })

  it('does not draw a rectangular border around inspected assistant message bodies', () => {
    render(<MessageRow messageRole="assistant" body="Selected assistant response" timestamp="12:12" inspected />)

    expect(screen.getByText('Selected assistant response').closest('[style*="border"]')).toHaveAttribute(
      'style',
      expect.stringContaining('border: 1px solid transparent')
    )
  })

  it('hides route metadata when disabled', () => {
    render(
      <>
        <MessageRow
          messageRole="user"
          body="Route this prompt through the local node"
          timestamp="12:13"
          routeNode="carrack"
          showRouteMetadata={false}
        />
        <MessageRow
          messageRole="assistant"
          body="Selected assistant response"
          timestamp="12:14"
          route="carrack"
          routeNode="carrack"
          showRouteMetadata={false}
        />
      </>
    )

    expect(screen.queryByText(/sent to/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/routed via/i)).not.toBeInTheDocument()
  })
})
