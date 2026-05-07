import { MeshVizPerfHarness } from '@/features/network/components/MeshVizPerfHarness'

export function MeshVizPerfPage() {
  return (
    <section className="space-y-4" data-testid="meshviz-perf-page">
      <div className="space-y-1">
        <p className="font-mono text-[length:var(--density-type-label)] uppercase tracking-[0.14em] text-accent">
          Performance harness
        </p>
        <h1 className="type-hero-title">MeshViz 200-node benchmark</h1>
        <p className="max-w-3xl text-[length:var(--density-type-body)] text-muted-foreground">
          Deterministic 200-node MeshViz scene for browser frame-pacing measurements.
        </p>
      </div>
      <MeshVizPerfHarness />
    </section>
  )
}
