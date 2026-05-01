import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { MessageRow } from './MessageRow'

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
      />,
    )

    const article = container.querySelector('article')

    expect(article).toBeInTheDocument()
    expect(article).toHaveClass('relative', '-mx-2', 'mb-5', 'block', 'w-[calc(100%+16px)]', 'select-none')
    expect(article).not.toHaveClass('focus-visible:outline-accent')
    expect(article).toHaveTextContent('Static response from the mesh')
    expect(container.querySelector('button')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /static response from the mesh/i })).not.toBeInTheDocument()
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
      />,
    )

    const button = screen.getByRole('button', { name: 'Inspect mesh route' })

    expect(button).toHaveClass('relative', '-mx-2', 'mb-5', 'block', 'w-[calc(100%+16px)]', 'select-none', 'focus-visible:outline-accent')
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
      />,
    )

    expect(screen.getByRole('button', { name: 'Inspect user message from 12:11' })).toBeInTheDocument()
  })

  it('renders user messages on a subtly dimmed surface', () => {
    render(
      <MessageRow
        messageRole="user"
        body="Route this prompt through the local node"
        timestamp="12:11"
      />,
    )

    const userSurface = screen.getByText('Route this prompt through the local node').parentElement

    expect(userSurface).toHaveAttribute('style', expect.stringContaining('background: var(--chat-user-message-background)'))
    expect(userSurface).toHaveAttribute('style', expect.stringContaining('border-left: 1px solid var(--color-accent)'))
    expect(userSurface).not.toHaveAttribute('style', expect.stringContaining('border-radius'))
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
      />,
    )

    expect(container.querySelector('article')).toHaveClass('select-none')
    expect(screen.getByText('Only this response text should be selectable')).toHaveClass('select-text')
    expect(screen.getByText('Assistant').parentElement).toHaveClass('select-none')
    expect(screen.getByText('256 tokens').parentElement).toHaveClass('select-none')
  })

  it('does not draw a rectangular border around inspected assistant message bodies', () => {
    render(
      <MessageRow
        messageRole="assistant"
        body="Selected assistant response"
        timestamp="12:12"
        inspected
      />,
    )

    expect(screen.getByText('Selected assistant response')).toHaveAttribute('style', expect.stringContaining('border: 1px solid transparent'))
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
      </>,
    )

    expect(screen.queryByText(/sent to/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/routed via/i)).not.toBeInTheDocument()
  })
})
