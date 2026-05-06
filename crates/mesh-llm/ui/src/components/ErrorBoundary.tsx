import { Component, ErrorInfo, ReactNode } from "react";

export class ErrorBoundary extends Component<{
  children: ReactNode;
  fallback?: ReactNode;
}, {
  hasError: boolean;
  error: Error | null;
}> {
  constructor(props: { children: ReactNode; fallback?: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): { hasError: true; error: Error } {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("❌ ERROR BOUNDARY CAUGHT:", error);
    console.error("Error info:", errorInfo);
    console.error("Component stack:", errorInfo.componentStack);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-destructive/10 p-8 text-destructive">
          <div className="mx-auto max-w-2xl rounded-lg border border-destructive/50 bg-card p-6">
            <h1 className="mb-4 text-2xl font-bold">Application Error</h1>
            <pre className="whitespace-pre-wrap rounded bg-black/50 p-4 text-sm text-foreground">
              {this.state.error?.message}
            </pre>
            {this.state.error?.stack && (
              <details className="mt-4">
                <summary className="cursor-pointer text-sm underline">Stack trace</summary>
                <pre className="mt-2 overflow-auto whitespace-pre-wrap rounded bg-black/50 p-4 text-xs">
                  {this.state.error.stack}
                </pre>
              </details>
            )}
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="mt-4 rounded bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
