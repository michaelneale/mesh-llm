import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { FeatureErrorBoundary } from '@/app/error-boundaries/FeatureErrorBoundary'
import { ErrorBoundaryPanel } from '@/lib/error/ErrorBoundaryPanel'
import { formatErrorDiagnostics } from '@/lib/error/format-error-diagnostics'

describe('ErrorBoundaryPanel', () => {
  it('formats stack traces with the React component stack when available', () => {
    const error = new Error('boom')
    error.stack = 'Error: boom\n    at FaultyComponent'

    expect(formatErrorDiagnostics(error, '\n    at RoutePanel')).toBe(
      'Error: boom\n    at FaultyComponent\n\nReact component stack:\nat RoutePanel',
    )
  })

  it('renders a larger diagnostic surface with stack trace text', () => {
    const error = new Error('section exploded')
    error.stack = 'Error: section exploded\n    at ChatPanel'

    render(
      <ErrorBoundaryPanel
        title="Something went wrong"
        description="This section failed to render."
        error={error}
        scopeLabel="Route section"
      />,
    )

    expect(screen.getByRole('alert')).toHaveClass('max-w-6xl')
    expect(screen.getByText('This section failed to render.')).toHaveClass('max-w-[96ch]')
    expect(screen.getByText('Error message')).toBeInTheDocument()
    expect(screen.getByText('section exploded')).toBeInTheDocument()
    expect(screen.getByText('Development diagnostics')).toBeInTheDocument()
    expect(screen.getByText(/at ChatPanel/)).toBeInTheDocument()
    expect(screen.getByText(/at ChatPanel/)).toHaveClass('text-[length:var(--density-type-label)]')
    expect(screen.getByText('Route section')).toBeInTheDocument()
  })
})

describe('FeatureErrorBoundary', () => {
  it('surfaces route errors with a refresh action', () => {
    const error = new Error('conversation route crashed')
    error.stack = 'Error: conversation route crashed\n    at ConversationRoute'

    render(<FeatureErrorBoundary error={error} />)

    expect(screen.getByRole('heading', { name: 'Something went wrong' })).toBeInTheDocument()
    expect(screen.getByText('conversation route crashed')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Refresh route' })).toBeInTheDocument()
    expect(screen.getByText(/ConversationRoute/)).toBeInTheDocument()
  })
})
