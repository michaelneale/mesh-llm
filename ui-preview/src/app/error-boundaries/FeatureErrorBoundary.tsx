import { ErrorBoundaryPanel } from '@/lib/error/ErrorBoundaryPanel'

type FeatureErrorBoundaryProps = { error?: Error }

export function FeatureErrorBoundary({ error }: FeatureErrorBoundaryProps) {
  return (
    <div className="flex min-h-full w-full items-center justify-center p-6 md:p-10">
      <ErrorBoundaryPanel
        title="Something went wrong"
        description="This section failed to render, but the rest of the app can stay available. Refresh the page to retry the route."
        error={error}
        scopeLabel="Route section"
        recoveryActionLabel="Refresh route"
      />
    </div>
  )
}
