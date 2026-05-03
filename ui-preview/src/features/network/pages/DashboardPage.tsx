import { useCallback, useEffect, useMemo, useState } from 'react'
import { LiveRefreshPill } from '@/components/ui/LiveRefreshPill'
import { ConnectBlock } from '@/features/network/components/ConnectBlock'
import { DashboardLiveLoadingGhost } from '@/features/network/components/DashboardLiveLoadingGhost'
import { MeshViz } from '@/features/network/components/MeshViz'
import { ModelCatalog } from '@/features/network/components/ModelCatalog'
import { NetworkHeroBanner } from '@/features/network/components/NetworkHeroBanner'
import { PeersTable } from '@/features/network/components/PeersTable'
import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { DashboardLayout } from '@/features/network/layouts/DashboardLayout'
import { buildDashboardMeshNodes, DASHBOARD_MESH_ID, meshNodeForPeer } from '@/features/network/lib/dashboard-mesh-nodes'
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
import { useStatusStream } from '@/features/network/api/use-status-stream'
import { adaptStatusToDashboard } from '@/features/network/api/status-adapter'
import { adaptModelsToSummary } from '@/features/network/api/models-adapter'
import { QueryProvider } from '@/lib/query/QueryProvider'

const DASHBOARD_LIVE_ERROR_OVERLAY_DELAY_MS = 10_000

function StatusStreamConnector({ enabled }: { enabled: boolean }) {
  const { mode } = useDataMode()
  useStatusStream({ enabled: mode === 'live' && enabled })
  return null
}

type DashboardDrawer = 'model' | 'node' | null
type DashboardPageProps = { data?: DashboardHarnessData }
type DashboardPageContentProps = DashboardPageProps & {
  liveFetchPaused: boolean
  setLiveFetchPaused: (paused: boolean) => void
}

function peerForNode(peers: Peer[], node: MeshNode | undefined) {
  if (!node) return undefined
  return peers.find((peer) => peer.id === node.peerId || peer.id === node.id)
}

function DashboardPageContent({ data = DASHBOARD_HARNESS, liveFetchPaused, setLiveFetchPaused }: DashboardPageContentProps) {
  const { mode, setMode } = useDataMode()
  const liveMode = mode === 'live'
  const liveQueriesEnabled = liveMode && !liveFetchPaused
  const statusQuery = useStatusQuery({ enabled: liveQueriesEnabled })
  const modelsQuery = useModelsQuery({ enabled: liveQueriesEnabled })

  const liveModels = modelsQuery.data ? adaptModelsToSummary(modelsQuery.data.mesh_models) : undefined
  const liveData = statusQuery.data && liveModels ? adaptStatusToDashboard(statusQuery.data, liveModels) : undefined
  const resolvedData = liveMode ? liveData : data
  const displayData = resolvedData ?? data
  const liveDataFetching = statusQuery.isFetching || modelsQuery.isFetching
  const hasLiveData = Boolean(liveData)
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
  const showLiveRefresh = liveMode && hasLiveData && liveDataFetching && !liveFetchPaused

  const meshNodes = useMemo(
    () => buildDashboardMeshNodes(displayData.peers, displayData.meshId, displayData.meshNodeSeeds),
    [displayData.meshId, displayData.meshNodeSeeds, displayData.peers],
  )
  const selfId = meshNodes.find((node) => node.role === 'self')?.id ?? 'self'
  const [selectedModel, setSelectedModel] = useState<ModelSummary | undefined>(undefined)
  const [selectedNode, setSelectedNode] = useState<MeshNode | undefined>()
  const [drawer, setDrawer] = useState<DashboardDrawer>(null)
  const { copyState, copyText } = useClipboardCopy()
  const selectedModelView = selectedModel && displayData.models.some((model) => model.name === selectedModel.name)
    ? selectedModel
    : undefined
  const selectedNodeView = selectedNode && displayData.peers.some((peer) => peer.id === selectedNode.peerId || peer.id === selectedNode.id)
    ? selectedNode
    : undefined
  const selectedPeer = peerForNode(displayData.peers, selectedNodeView)

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
        productionDescription="The dashboard is waiting for live mesh status and model data. Keep the page open while the service recovers, or switch Data source back to Harness in Tweaks to inspect sample data."
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
    <>
      {showLiveRefresh ? <LiveRefreshPill className="mb-3">Refreshing live data</LiveRefreshPill> : null}
      <DashboardLayout
        hero={(
          <NetworkHeroBanner
            title={displayData.hero.title}
            description={displayData.hero.description}
            actions={displayData.hero.actions}
          />
        )}
        status={<StatusStrip metrics={displayData.statusMetrics} />}
        topology={(
          <MeshViz
            nodes={meshNodes}
            selfId={selfId}
            meshId={displayData.meshId ?? DASHBOARD_MESH_ID}
            height={420}
            selectedNodeId={selectedNodeView?.id}
            onPick={selectNode}
            getNodePeer={(node) => peerForNode(displayData.peers, node)}
          />
        )}
        catalog={<ModelCatalog models={displayData.models} filterLabel="All" selectedModelName={selectedModelView?.name} onSelect={selectModel} />}
        peers={<PeersTable peers={displayData.peers} summary={displayData.peerSummary} selectedPeerId={selectedPeer?.id} onSelect={selectPeer} />}
        connect={(
          <ConnectBlock
            installHref={displayData.connect.installHref}
            apiUrl={env.apiUrl}
            apiStatus={displayData.connect.apiStatus}
            runCommand={displayData.connect.runCommand}
            description={displayData.connect.description}
            copyState={copyState}
            onCopy={copyConnectCommand}
          />
        )}
        drawers={(
          <>
            <ModelDrawer open={drawer === 'model'} model={selectedModelView} peers={displayData.peers} onClose={closeModelDrawer} />
            <NodeDrawer open={drawer === 'node'} node={selectedNodeView} peer={selectedPeer} models={displayData.models} onClose={closeDrawer} />
          </>
        )}
      />
    </>
  )
}

export function DashboardPageSurface(props: DashboardPageProps = {}) {
  const [liveFetchPaused, setLiveFetchPaused] = useState(false)

  return (
    <>
      {typeof EventSource !== 'undefined' && <StatusStreamConnector enabled={!liveFetchPaused} />}
      <DashboardPageContent {...props} liveFetchPaused={liveFetchPaused} setLiveFetchPaused={setLiveFetchPaused} />
    </>
  )
}

export function DashboardPage(props: DashboardPageProps = {}) {
  return (
    <QueryProvider>
      <DashboardPageSurface {...props} />
    </QueryProvider>
  )
}
