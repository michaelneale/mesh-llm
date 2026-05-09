import { Component, type ErrorInfo, type ReactNode } from 'react'
import { ErrorBoundaryPanel } from '@/lib/error/ErrorBoundaryPanel'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
  errorInfo?: ErrorInfo
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    this.setState({ error, errorInfo: info })
    console.error('ErrorBoundary caught:', error, info)
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback
      return (
        <div className="flex min-h-[360px] w-full items-center justify-center p-6 md:p-10">
          <ErrorBoundaryPanel
            title="Something went wrong"
            description="A component failed to render. Refresh the app to reset this part of the interface."
            error={this.state.error}
            componentStack={this.state.errorInfo?.componentStack ?? undefined}
            scopeLabel="React component tree"
          />
        </div>
      )
    }
    return this.props.children
  }
}
