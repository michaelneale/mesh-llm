import { useCallback, useEffect, useMemo, useState } from 'react'
import { ConnectBlock, type ConnectApiTargetLiveness } from '@/features/network/components/ConnectBlock'
import { DashboardLiveLoadingGhost } from '@/features/network/components/DashboardLiveLoadingGhost'
import { MeshViz } from '@/features/network/components/MeshViz'
import { MeshVizTopologyGhost } from '@/features/network/components/MeshVizTopologyGhost'
import { ModelCatalog } from '@/features/network/components/ModelCatalog'
import { NetworkHeroBanner } from '@/features/network/components/NetworkHeroBanner'
import { PeersTable } from '@/features/network/components/PeersTable'
import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { DashboardLayout } from '@/features/network/layouts/DashboardLayout'
import {
  DASHBOARD_MESH_ID,
  meshNodeForPeer,
  reconcileDashboardMeshNodes
} from '@/features/network/lib/dashboard-mesh-nodes'
import { ModelDrawer } from '@/features/drawers/components/ModelDrawer'
import { NodeDrawer } from '@/features/drawers/components/NodeDrawer'
import { StatusStrip } from '@/features/status/components/StatusStrip'
import { DASHBOARD_HARNESS } from '@/features/app-tabs/data'
import type { DashboardHarnessData, MeshNode, ModelSummary, Peer } from '@/features/app-tabs/types'
import { env } from '@/lib/env'
import { useDataMode } from '@/lib/data-mode'
import { useClipboardCopy } from '@/lib/useClipboardCopy'
import { useStatusQuery } from '@/features/network/api/use-status-query'
import { useModelsQuery } from '@/features/network/api/use-models-query'
import { adaptStatusToDashboard } from '@/features/network/api/status-adapter'
import { adaptModelsToSummary } from '@/features/network/api/models-adapter'

const DASHBOARD_LIVE_ERROR_OVERLAY_DELAY_MS = 10_000

type DashboardDrawer = 'model' | 'node' | null
type DashboardPageProps = { data?: DashboardHarnessData }
type DashboardPageContentProps = DashboardPageProps & {
  liveFetchPaused: boolean
  setLiveFetchPaused: (paused: boolean) => void
}
type MeshNodeSnapshot = {
  source: DashboardHarnessData
  meshId: string | undefined
  nodes: MeshNode[]
}

function peerForNode(peers: Peer[], node: MeshNode | undefined) {
  if (!node) return undefined
  return peers.find((peer) => peer.id === node.peerId || peer.id === node.id)
}

function resolveConnectApiTargetLiveness(
  statusQuery: ReturnType<typeof useStatusQuery>,
  liveMode: boolean
): ConnectApiTargetLiveness {
  if (!liveMode) return 'configured'
  if (statusQuery.isError) return 'unavailable'
  if (statusQuery.data) return 'live'
  return 'checking'
}

function buildMeshNodeSnapshot(data: DashboardHarnessData, previous: MeshNodeSnapshot | undefined): MeshNodeSnapshot {
  const previousNodes = previous?.meshId === data.meshId ? previous.nodes : []

  return {
    source: data,
    meshId: data.meshId,
    nodes: reconcileDashboardMeshNodes(previousNodes, data.peers, data.meshId, data.meshNodeSeeds)
  }
}

function DashboardPageContent({
  data = DASHBOARD_HARNESS,
  liveFetchPaused,
  setLiveFetchPaused
}: DashboardPageContentProps) {
  const { mode, setMode } = useDataMode()
  const liveMode = mode === 'live'
  const liveQueriesEnabled = liveMode && !liveFetchPaused
  const statusQuery = useStatusQuery({ enabled: liveQueriesEnabled })
  const modelsQuery = useModelsQuery({ enabled: liveQueriesEnabled })
  const connectApiTargetLiveness = resolveConnectApiTargetLiveness(statusQuery, liveMode)

  const liveModels = useMemo(
    () => (modelsQuery.data ? adaptModelsToSummary(modelsQuery.data.mesh_models) : undefined),
    [modelsQuery.data]
  )
  const liveData = useMemo(
    () => (statusQuery.data ? adaptStatusToDashboard(statusQuery.data, liveModels ?? []) : undefined),
    [liveModels, statusQuery.data]
  )
  const resolvedData = liveMode ? liveData : data
  const displayData = resolvedData ?? data
  const hasLiveData = Boolean(liveData)
  const meshVizReadinessKey = liveMode && !hasLiveData ? undefined : (displayData.meshId ?? DASHBOARD_MESH_ID)
  const [meshVizReadyKey, setMeshVizReadyKey] = useState<string | undefined>()
  const [meshNodeSnapshot, setMeshNodeSnapshot] = useState<MeshNodeSnapshot>(() =>
    buildMeshNodeSnapshot(displayData, undefined)
  )
  const meshVizReady = meshVizReadinessKey !== undefined && meshVizReadyKey === meshVizReadinessKey
  useEffect(() => {
    if (!liveMode) {
      setLiveFetchPaused(false)
      return undefined
    }

    if (liveFetchPaused) return undefined

    if (hasLiveData) return undefined

    const timer = window.setTimeout(() => {
      setLiveFetchPaused(true)
    }, DASHBOARD_LIVE_ERROR_OVERLAY_DELAY_MS)

    return () => window.clearTimeout(timer)
  }, [hasLiveData, liveFetchPaused, liveMode, setLiveFetchPaused])
  const showLiveError = liveMode && liveFetchPaused
  const showLiveLoading = liveMode && !liveData && !showLiveError
  let currentMeshNodeSnapshot = meshNodeSnapshot
  if (currentMeshNodeSnapshot.source !== displayData) {
    currentMeshNodeSnapshot = buildMeshNodeSnapshot(displayData, currentMeshNodeSnapshot)
    setMeshNodeSnapshot(currentMeshNodeSnapshot)
  }
  const meshNodes = currentMeshNodeSnapshot.nodes
  const selfId = meshNodes.find((node) => node.role === 'self')?.id ?? 'self'
  const peerById = useMemo(() => new Map(displayData.peers.map((peer) => [peer.id, peer])), [displayData.peers])
  const getNodePeer = useCallback(
    (node: MeshNode) => (node.peerId ? peerById.get(node.peerId) : peerById.get(node.id)),
    [peerById]
  )
  const [selectedModel, setSelectedModel] = useState<ModelSummary | undefined>(undefined)
  const [selectedNode, setSelectedNode] = useState<MeshNode | undefined>()
  const [hoveredPeerId, setHoveredPeerId] = useState<string | undefined>()
  const [filteredPeerIds, setFilteredPeerIds] = useState<string[]>([])
  const [drawer, setDrawer] = useState<DashboardDrawer>(null)
  const { copyState, copyText } = useClipboardCopy()
  const selectedModelView =
    selectedModel && displayData.models.some((model) => model.name === selectedModel.name) ? selectedModel : undefined
  const selectedNodeView =
    selectedNode && displayData.peers.some((peer) => peer.id === selectedNode.peerId || peer.id === selectedNode.id)
      ? selectedNode
      : undefined
  const selectedPeer = peerForNode(displayData.peers, selectedNodeView)
  const hoveredMeshNodeId = useMemo(() => {
    if (!hoveredPeerId) return undefined
    const hoveredPeer = peerById.get(hoveredPeerId)
    if (!hoveredPeer) return undefined
    return meshNodeForPeer(hoveredPeer, meshNodes).id
  }, [hoveredPeerId, meshNodes, peerById])
  const dimmedMeshNodeIds = useMemo(() => {
    if (filteredPeerIds.length === 0 || filteredPeerIds.length === displayData.peers.length) {
      return undefined
    }

    const filteredPeerIdSet = new Set(filteredPeerIds)

    return new Set(meshNodes.filter((node) => !filteredPeerIdSet.has(node.peerId ?? node.id)).map((node) => node.id))
  }, [displayData.peers.length, filteredPeerIds, meshNodes])

  const selectModel = (model: ModelSummary) => {
    setSelectedModel(model)
    setDrawer('model')
  }
  const selectNode = (node: MeshNode) => {
    setSelectedNode(node)
    setDrawer('node')
  }
  const selectPeer = (peer: Peer) => selectNode(meshNodeForPeer(peer, meshNodes))
  const closeDrawer = () => setDrawer(null)
  const closeModelDrawer = () => {
    setDrawer(null)
    setSelectedModel(undefined)
  }
  const copyConnectCommand = useCallback(() => {
    void copyText(displayData.connect.runCommand)
  }, [copyText, displayData.connect.runCommand])
  const retryLiveData = useCallback(() => {
    setLiveFetchPaused(false)
    void Promise.all([statusQuery.refetch(), modelsQuery.refetch()])
  }, [modelsQuery, setLiveFetchPaused, statusQuery])
  const switchToTestData = useCallback(() => setMode('harness'), [setMode])

  if (showLiveError) {
    return (
      <LiveDataUnavailableOverlay
        debugTitle="Could not reach the local mesh backend"
        title="Live mesh data is unavailable"
        debugDescription="The dashboard could not fetch the initial status and model catalog from the configured API target. Start the backend, verify the endpoint, or switch Data source back to Harness in Tweaks while debugging."
        productionDescription="The dashboard is waiting for live mesh status and model data. Keep the page open while the service recovers."
        onRetry={retryLiveData}
        onSwitchToTestData={switchToTestData}
      >
        <DashboardLiveLoadingGhost />
      </LiveDataUnavailableOverlay>
    )
  }

  if (showLiveLoading) {
    return <DashboardLiveLoadingGhost />
  }

  return (
    <DashboardLayout
      hero={
        <NetworkHeroBanner
          title={displayData.hero.title}
          description={displayData.hero.description}
          actions={displayData.hero.actions}
        />
      }
      status={<StatusStrip metrics={displayData.statusMetrics} />}
      topology={
        <div className="relative h-full min-h-0">
          <MeshViz
            key={meshVizReadinessKey}
            nodes={meshNodes}
            selfId={selfId}
            meshId={displayData.meshId ?? DASHBOARD_MESH_ID}
            selectedNodeId={selectedNodeView?.id}
            hoveredNodeId={hoveredMeshNodeId}
            dimmedNodeIds={dimmedMeshNodeIds}
            onPick={selectNode}
            onReady={() => setMeshVizReadyKey(meshVizReadinessKey)}
            getNodePeer={getNodePeer}
          />
          {!meshVizReady && (
            <div className="pointer-events-none absolute inset-0">
              <MeshVizTopologyGhost />
            </div>
          )}
        </div>
      }
      catalog={
        <ModelCatalog models={displayData.models} selectedModelName={selectedModelView?.name} onSelect={selectModel} />
      }
      peers={
        <PeersTable
          peers={displayData.peers}
          models={displayData.models}
          summary={displayData.peerSummary}
          selectedPeerId={selectedPeer?.id}
          onSelect={selectPeer}
          onHoverPeerIdChange={setHoveredPeerId}
          onFilteredPeerIdsChange={setFilteredPeerIds}
        />
      }
      connect={
        <ConnectBlock
          installHref={displayData.connect.installHref}
          apiUrl={env.apiUrl}
          apiStatus={displayData.connect.apiStatus}
          apiTargetLiveness={connectApiTargetLiveness}
          runCommand={displayData.connect.runCommand}
          description={displayData.connect.description}
          copyState={copyState}
          onCopy={copyConnectCommand}
        />
      }
      drawers={
        <>
          <ModelDrawer
            open={drawer === 'model'}
            model={selectedModelView}
            peers={displayData.peers}
            onClose={closeModelDrawer}
          />
          <NodeDrawer
            open={drawer === 'node'}
            node={selectedNodeView}
            peer={selectedPeer}
            models={displayData.models}
            onClose={closeDrawer}
          />
        </>
      }
    />
  )
}

export function DashboardPageSurface(props: DashboardPageProps = {}) {
  const [liveFetchPaused, setLiveFetchPaused] = useState(false)

  return (
    <>
      <DashboardPageContent {...props} liveFetchPaused={liveFetchPaused} setLiveFetchPaused={setLiveFetchPaused} />
    </>
  )
}

export function DashboardPage(props: DashboardPageProps = {}) {
  return <DashboardPageSurface {...props} />
}
