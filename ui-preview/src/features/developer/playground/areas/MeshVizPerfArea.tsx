import { MeshVizPerfHarness } from '@/features/network/components/MeshVizPerfHarness'
import { PlaygroundPanel } from '../primitives'

export function MeshVizPerfArea() {
  return (
    <PlaygroundPanel
      title="MeshViz 200-node benchmark"
      description="Full-width deterministic topology scene for stress-testing pan, zoom, packet alignment, and frame pacing without drawer or sidebar chrome."
    >
      <MeshVizPerfHarness height={680} />
    </PlaygroundPanel>
  )
}
