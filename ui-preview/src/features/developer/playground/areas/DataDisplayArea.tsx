import { useMemo, useState } from 'react'
import { Network } from 'lucide-react'
import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { ModelDrawer } from '@/features/drawers/components/ModelDrawer'
import { NodeDrawer } from '@/features/drawers/components/NodeDrawer'
import { MeshViz } from '@/features/network/components/MeshViz'
import { ModelCatalog } from '@/features/network/components/ModelCatalog'
import { NetworkHeroBanner } from '@/features/network/components/NetworkHeroBanner'
import { PeersTable } from '@/features/network/components/PeersTable'
import { DashboardLiveLoadingGhost } from '@/features/network/components/DashboardLiveLoadingGhost'
import { buildDashboardMeshNodes, DASHBOARD_MESH_ID, meshNodeForPeer } from '@/features/network/lib/dashboard-mesh-nodes'
import { StatusStrip } from '@/features/status/components/StatusStrip'
import { DASHBOARD_HARNESS } from '@/features/app-tabs/data'
import type { MeshNode } from '@/features/app-tabs/types'
import { OptionGroup, PlaygroundPanel, SidebarTabs, ToggleChip } from '../primitives'
import { TAG_OPTIONS, type DeveloperPlaygroundState } from '../useDeveloperPlaygroundState'

export function DataDisplayArea({ state }: { state: DeveloperPlaygroundState }) {
  const meshNodes = useMemo(
    () => buildDashboardMeshNodes(DASHBOARD_HARNESS.peers, DASHBOARD_HARNESS.meshId ?? DASHBOARD_MESH_ID, DASHBOARD_HARNESS.meshNodeSeeds),
    [],
  )
  const selfId = meshNodes.find((node) => node.role === 'self')?.id ?? meshNodes[0]?.id ?? 'self'
  const [selectedNode, setSelectedNode] = useState<MeshNode | undefined>(meshNodes[0])
  const [openDrawer, setOpenDrawer] = useState<'model' | 'node' | null>(null)
  const [showTopologyLoading, setShowTopologyLoading] = useState(false)
  const selectedPeer = selectedNode
    ? DASHBOARD_HARNESS.peers.find((peer) => peer.id === selectedNode.peerId || peer.id === selectedNode.id)
    : undefined

  function selectNode(node: MeshNode) {
    setSelectedNode(node)
    setOpenDrawer('node')
  }

  function toggleTopologyLoading() {
    if (!showTopologyLoading) setOpenDrawer(null)
    setShowTopologyLoading((current) => !current)
  }

  return (
    <>
      <SidebarTabs
        ariaLabel="Data display previews"
        defaultValue="tables"
        tabs={[
        {
          value: 'tables',
          label: 'Tables and catalog',
          content: (
            <>
              <PlaygroundPanel
                title="Data display controls"
                description="Swap highlights and model tags while the table and catalog stay in their compact operational layout."
              >
                <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
                  <OptionGroup
                    label="Selected model"
                    value={state.selectedDashboardModelName}
                    options={state.dashboardModels.slice(0, 4).map((model) => ({ value: model.name, label: model.name }))}
                    onChange={state.setSelectedDashboardModelName}
                  />
                  <OptionGroup
                    label="Selected peer"
                    value={state.selectedPeerId ?? DASHBOARD_HARNESS.peers[0]?.id ?? ''}
                    options={DASHBOARD_HARNESS.peers.map((peer) => ({ value: peer.id, label: peer.hostname }))}
                    onChange={state.setSelectedPeerId}
                  />
                </div>
                <div className="mt-4 rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                  <div className="type-label text-fg-faint">Model tags</div>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {TAG_OPTIONS.map((tag) => (
                      <ToggleChip
                        key={tag}
                        label={`${tag} tag`}
                        onToggle={() => state.toggleSelectedModelTag(tag)}
                        pressed={Boolean(state.selectedDashboardModel?.tags.includes(tag))}
                      />
                    ))}
                  </div>
                  <section aria-label="Selected model tags" className="mt-3 flex flex-wrap gap-1.5">
                    {(state.selectedDashboardModel?.tags.length ? state.selectedDashboardModel.tags : ['No tags']).map((tag) => (
                      <span key={tag} className="inline-flex items-center rounded-full border border-border px-2 py-0.5 text-[length:var(--density-type-label)] font-medium text-fg-faint">
                        {tag}
                      </span>
                    ))}
                  </section>
                </div>
              </PlaygroundPanel>

              <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
                <PeersTable
                  peers={DASHBOARD_HARNESS.peers}
                  summary={DASHBOARD_HARNESS.peerSummary}
                  selectedPeerId={state.selectedPeerId}
                  onSelect={(peer) => state.setSelectedPeerId(peer.id)}
                />
                <ModelCatalog
                  models={state.dashboardModels}
                  filterLabel="All"
                  selectedModelName={state.selectedDashboardModelName}
                  onSelect={(model) => state.setSelectedDashboardModelName(model.name)}
                />
              </div>
            </>
          ),
        },
        {
          value: 'topology',
          label: 'Topology and drawers',
          content: (
            <>
              <PlaygroundPanel
                title="Topology loading state"
                description="Switch this section between the interactive topology preview and the production live loading ghost."
                actions={(
                  <ToggleChip
                    label="Show loading state"
                    onToggle={toggleTopologyLoading}
                    pressed={showTopologyLoading}
                  />
                )}
              >
                <p className="max-w-[68ch] text-[length:var(--density-type-caption-lg)] text-fg-dim">
                  Use this to inspect how operators see pending mesh data before topology controls and drawer launchers are ready.
                </p>
              </PlaygroundPanel>

              {showTopologyLoading ? (
                <DashboardLiveLoadingGhost />
              ) : (
                <>
                  <PlaygroundPanel
                    title="Network hero"
                    description="Preview the high-level network summary banner with its production action stack."
                  >
                    <NetworkHeroBanner
                      title={DASHBOARD_HARNESS.hero.title}
                      description={DASHBOARD_HARNESS.hero.description}
                      actions={DASHBOARD_HARNESS.hero.actions}
                      leadingIcon={<Network className="size-4" aria-hidden="true" strokeWidth={1.8} />}
                    />
                  </PlaygroundPanel>

                  <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
                    <PlaygroundPanel
                      title="Mesh topology"
                      description="Exercise pan, zoom, hover cards, node selection, and drawer handoff from the same harness peers used by the dashboard."
                    >
                      <MeshViz
                        compact
                        nodes={meshNodes}
                        selfId={selfId}
                        meshId={DASHBOARD_HARNESS.meshId ?? DASHBOARD_MESH_ID}
                        height={360}
                        selectedNodeId={selectedNode?.id}
                        onPick={selectNode}
                        getNodePeer={(node) => DASHBOARD_HARNESS.peers.find((peer) => peer.id === node.peerId || peer.id === node.id)}
                      />
                    </PlaygroundPanel>

                    <PlaygroundPanel
                      title="Drawer launchers"
                      description="Open node and model drawers without leaving the playground. The selected table/catalog state feeds both panels."
                    >
                      <div className="space-y-3">
                        <button
                          className="ui-control-primary inline-flex w-full items-center justify-center rounded-[var(--radius)] px-3 py-2 text-[length:var(--density-type-control)] font-medium"
                          onClick={() => setOpenDrawer('node')}
                          type="button"
                        >
                          Open node drawer
                        </button>
                        <button
                          className="ui-control inline-flex w-full items-center justify-center rounded-[var(--radius)] border px-3 py-2 text-[length:var(--density-type-control)] font-medium"
                          onClick={() => setOpenDrawer('model')}
                          type="button"
                        >
                          Open model drawer
                        </button>
                        <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                          <div className="type-label text-fg-faint">Selected node</div>
                          <div className="mt-1 font-mono text-[length:var(--density-type-caption-lg)] text-foreground">{selectedPeer?.hostname ?? selectedNode?.label ?? 'No node selected'}</div>
                        </div>
                        <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                          <div className="type-label text-fg-faint">Selected model</div>
                          <div className="mt-1 font-mono text-[length:var(--density-type-caption-lg)] text-foreground">{state.selectedDashboardModel?.name ?? 'No model selected'}</div>
                        </div>
                      </div>
                    </PlaygroundPanel>
                  </div>
                </>
              )}
            </>
          ),
        },
        {
          value: 'status',
          label: 'Status strip',
          content: (
            <PlaygroundPanel
              title="Metric strip"
              description="Keep status density high, labels explicit, and machine values easy to compare at a glance."
            >
              <StatusStrip metrics={DASHBOARD_HARNESS.statusMetrics} />
            </PlaygroundPanel>
          ),
        },
        {
          value: 'live-state',
          label: 'Live states',
          content: (
            <LiveDataUnavailableOverlay
              debugTitle="Could not reach the local mesh backend"
              title="Live mesh data is unavailable"
              debugDescription="The playground renders the dashboard fallback copy used when live status and model catalog queries cannot resolve."
              productionDescription="The dashboard is waiting for live mesh status and model data. Operators can retry or switch back to harness data."
              onRetry={() => undefined}
              onSwitchToTestData={() => undefined}
            >
              <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
                <PlaygroundPanel title="Fallback topology" description="Dimmed child content stays recognizable while the live-data alert takes focus.">
                  <MeshViz
                    compact
                    nodes={meshNodes}
                    selfId={selfId}
                    meshId={DASHBOARD_HARNESS.meshId ?? DASHBOARD_MESH_ID}
                    height={300}
                    selectedNodeId={selectedNode?.id}
                    getNodePeer={(node) => DASHBOARD_HARNESS.peers.find((peer) => peer.id === node.peerId || peer.id === node.id)}
                  />
                </PlaygroundPanel>
                <ModelCatalog
                  models={state.dashboardModels}
                  filterLabel="Harness"
                  selectedModelName={state.selectedDashboardModelName}
                  onSelect={(model) => state.setSelectedDashboardModelName(model.name)}
                />
              </div>
            </LiveDataUnavailableOverlay>
          ),
        },
        ]}
      />
      <ModelDrawer open={openDrawer === 'model'} model={state.selectedDashboardModel} peers={DASHBOARD_HARNESS.peers} onClose={() => setOpenDrawer(null)} />
      <NodeDrawer open={openDrawer === 'node'} node={selectedNode ?? meshNodeForPeer(DASHBOARD_HARNESS.peers[0], meshNodes)} peer={selectedPeer} models={state.dashboardModels} onClose={() => setOpenDrawer(null)} />
    </>
  )
}
