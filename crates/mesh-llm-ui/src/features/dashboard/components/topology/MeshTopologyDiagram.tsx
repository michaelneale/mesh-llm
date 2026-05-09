import { EmptyPanel } from '@/features/dashboard/components/details'

import type { MeshTopologyDiagramProps } from '@/features/dashboard/components/topology/types'
import { MeshRadarField } from '@/features/dashboard/components/topology/ui/MeshRadarField'

export function MeshTopologyDiagram({
  status,
  nodes,
  selectedModel,
  themeMode,
  onOpenNode,
  highlightedNodeId,
  fullscreen,
  heightClass,
  containerStyle
}: MeshTopologyDiagramProps) {
  if (!status) {
    return <EmptyPanel text="No topology data yet." />
  }
  if (!nodes.length) {
    return <EmptyPanel text="No host or worker nodes visible yet." />
  }

  return (
    <MeshRadarField
      status={status}
      nodes={nodes}
      selectedModel={selectedModel}
      themeMode={themeMode}
      onOpenNode={onOpenNode}
      highlightedNodeId={highlightedNodeId}
      fullscreen={fullscreen ?? false}
      heightClass={heightClass}
      containerStyle={containerStyle}
    />
  )
}
