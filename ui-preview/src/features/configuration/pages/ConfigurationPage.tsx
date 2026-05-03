import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Binary, Blocks, Brackets, Computer, LockKeyhole, ShieldCheck } from 'lucide-react'
import { LoadingGhostBlock } from '@/components/ui/LoadingGhostBlock'
import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { LiveRefreshPill } from '@/components/ui/LiveRefreshPill'
import { CatalogPopover } from '@/features/configuration/components/CatalogPopover'
import { ConfigurationHeader } from '@/features/configuration/components/ConfigurationHeader'
import { ConfigurationTabs, type ConfigurationTabItem } from '@/features/configuration/components/ConfigurationTabs'
import type { ConfigurationTabId } from '@/features/configuration/components/configuration-tab-ids'
import { jumpCtxPower, stepCtx } from '@/features/configuration/components/ctx-slider-utils'
import { DefaultsTab } from '@/features/configuration/components/DefaultsTab'
import { NodeRail } from '@/features/configuration/components/NodeRail'
import { NodeSection } from '@/features/configuration/components/NodeSection'
import { TomlView } from '@/features/configuration/components/TomlView'
import { createDefaultsValues } from '@/features/configuration/hooks/useDefaultsSettingsState'
import { cloneConfigurationState, createConfigurationSnapshot, createInitialConfigurationState, hasInvalidAllocation, type ConfigurationState, useConfigurationHistory } from '@/features/configuration/hooks/useConfigurationHistory'
import { ConfigurationDeploymentLayout, ConfigurationLayout, ConfigurationPlaceholderPanel } from '@/features/configuration/layouts/ConfigurationLayout'
import { hasConfigurablePlacement } from '@/features/configuration/lib/config-math'
import { getNodeTargetContainerIdx, createSeparatePlacementSnapshot, restoreSeparatePlacement } from '@/features/configuration/pages/ConfigurationPage.helpers'
import { KeyboardLegend } from '@/features/configuration/pages/ConfigurationPageKeyboardLegend'
import { UnsavedConfigurationNavigationBlocker } from '@/features/configuration/pages/ConfigurationPageNavigationBlocker'
import { useConfigurationPageSelection } from '@/features/configuration/pages/useConfigurationPageSelection'
import { useConfigurationPageKeyboardShortcuts } from '@/features/configuration/pages/useConfigurationPageKeyboardShortcuts'
import { CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import type { ConfigurationHarnessData, Placement } from '@/features/app-tabs/types'
import { useConfigQuery } from '@/features/configuration/api/use-config-query'
import { useDataMode } from '@/lib/data-mode'
import { useBooleanFeatureFlag } from '@/lib/feature-flags'
import { QueryProvider } from '@/lib/query/QueryProvider'

type ConfigurationPageProps = {
  activeTab?: ConfigurationTabId
  data?: ConfigurationHarnessData
  enableNavigationBlocker?: boolean
  initialTab?: ConfigurationTabId
  onTabChange?: (tab: ConfigurationTabId) => void
}

function ReadOnlyNodesDivider() {
  return (
    <div className="mt-4 mb-2 flex items-center gap-2.5">
      <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.16em] text-fg-faint">Peers</span>
      <span aria-hidden="true" className="h-px flex-1 bg-border-soft" />
      <span className="inline-flex shrink-0 items-center gap-1.5 font-mono text-[11px] font-medium uppercase leading-none tracking-[0.14em] text-fg-faint">
        read-only
        <LockKeyhole aria-hidden="true" className="size-3.5" strokeWidth={1.7} />
      </span>
    </div>
  )
}

function ConfigurationLiveLoadingGhost() {
  const railRows = ['node-a', 'node-b', 'node-c', 'node-d']
  const nodeCards = ['gpu-a', 'gpu-b', 'gpu-c']
  const settingsRows = ['setting-a', 'setting-b', 'setting-c']

  return (
    <ConfigurationLayout
      header={(
        <header className="sticky top-0 z-20 bg-transparent">
          <div className="flex min-h-[76px] flex-wrap items-center justify-between gap-3 px-5 py-3">
            <div className="min-w-0">
              <div className="type-label text-fg-faint">Live API</div>
              <LoadingGhostBlock className="mt-2 h-6 w-56" />
              <LoadingGhostBlock className="mt-2 h-3 w-80" />
            </div>
            <div className="flex items-center gap-1.5">
              <LoadingGhostBlock className="h-[30px] w-[30px]" />
              <LoadingGhostBlock className="h-[30px] w-[30px]" />
              <LoadingGhostBlock className="h-[30px] w-24" />
              <LoadingGhostBlock className="h-[30px] w-28" />
            </div>
          </div>
        </header>
      )}
    >
      <div className="grid gap-3.5 px-5" style={{ gridTemplateColumns: '220px minmax(0, 1fr)' }}>
        <aside className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-3">
          <LoadingGhostBlock className="h-4 w-28" />
          <div className="mt-3 space-y-2">
            {railRows.map((row) => <LoadingGhostBlock key={row} className="h-12" />)}
          </div>
        </aside>
        <section className="space-y-3">
          <div className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-4">
            <div className="flex items-center justify-between">
              <LoadingGhostBlock className="h-5 w-44" />
              <LoadingGhostBlock className="h-7 w-32 rounded-full" />
            </div>
            <div className="mt-4 grid gap-3">
              {nodeCards.map((card) => (
                <div key={card} className="rounded-[var(--radius)] border border-border-soft bg-background p-3">
                  <div className="flex items-center justify-between">
                    <LoadingGhostBlock className="h-4 w-36" />
                    <LoadingGhostBlock className="h-4 w-20" />
                  </div>
                  <LoadingGhostBlock className="mt-3 h-2 w-full rounded-full" />
                  <LoadingGhostBlock className="mt-2 h-2 w-2/3 rounded-full" />
                </div>
              ))}
            </div>
          </div>
          <div className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-4">
            <LoadingGhostBlock className="h-4 w-40" />
            <div className="mt-3 space-y-2">
              {settingsRows.map((row) => <LoadingGhostBlock key={row} className="h-10" />)}
            </div>
          </div>
        </section>
      </div>
    </ConfigurationLayout>
  )
}

export function ConfigurationPageContent({ activeTab: controlledActiveTab, data = CONFIGURATION_HARNESS, enableNavigationBlocker = true, initialTab = 'defaults', onTabChange }: ConfigurationPageProps = {}) {
  const { mode, setMode } = useDataMode()
  const liveMode = mode === 'live'
  const signingAttestationEnabled = useBooleanFeatureFlag('configuration/signingAttestation')
  const integrationsEnabled = useBooleanFeatureFlag('configuration/integrations')
  const { data: liveData, isFetching, isError, modelsQuery, statusQuery } = useConfigQuery({ enabled: liveMode })
  const resolvedData = liveMode ? liveData : data
  const displayData = resolvedData ?? data
  const showLiveError = liveMode && !liveData && !isFetching && isError
  const showLiveLoading = liveMode && !liveData && !showLiveError
  const showLiveRefresh = liveMode && Boolean(liveData) && isFetching

  const initialDefaultsValues = useMemo(() => createDefaultsValues(displayData.defaults), [displayData.defaults])
  const initialConfiguration = useMemo(() => createInitialConfigurationState(displayData.nodes, displayData.assigns, initialDefaultsValues), [displayData.assigns, displayData.nodes, initialDefaultsValues])
  const configurationSourceKey = useMemo(() => createConfigurationSnapshot(displayData.nodes, displayData.assigns, initialDefaultsValues), [displayData.assigns, displayData.nodes, initialDefaultsValues])
  const latestInitialConfigurationRef = useRef(initialConfiguration)
  useEffect(() => {
    latestInitialConfigurationRef.current = initialConfiguration
  }, [initialConfiguration])
  const { configuration, setAssigns, updateConfiguration, resetConfiguration, canUndo, canRedo, undoConfigurationChange, redoConfigurationChange } = useConfigurationHistory(initialConfiguration)
  const nodes = configuration.nodes
  const assigns = configuration.assigns
  const defaultsValues = configuration.defaultsValues
  const localNodeId = nodes[0]?.id ?? displayData.nodes[0]?.id ?? null
  const localNodes = useMemo(() => (localNodeId ? nodes.filter((node) => node.id === localNodeId) : []), [localNodeId, nodes])
  const remoteNodes = useMemo(() => (localNodeId ? nodes.filter((node) => node.id !== localNodeId) : []), [localNodeId, nodes])
  const localAssigns = useMemo(() => (localNodeId ? assigns.filter((assign) => assign.nodeId === localNodeId) : []), [assigns, localNodeId])
  const localInitialConfiguration = useMemo(
    () => createInitialConfigurationState(
      initialConfiguration.nodes.filter((node) => node.id === localNodeId),
      initialConfiguration.assigns.filter((assign) => assign.nodeId === localNodeId),
      initialConfiguration.defaultsValues,
    ),
    [initialConfiguration, localNodeId],
  )
  const [activeTabState, setActiveTabState] = useState<ConfigurationTabId>(initialTab)
  const activeTab = controlledActiveTab ?? activeTabState
  const [collapsedMap, setCollapsedMap] = useState<Record<string, boolean>>({})
  const [savedConfiguration, setSavedConfiguration] = useState<ConfigurationState>(() => cloneConfigurationState(initialConfiguration))
  const appliedConfigurationSourceKeyRef = useRef(configurationSourceKey)
  useEffect(() => {
    if (appliedConfigurationSourceKeyRef.current === configurationSourceKey) return

    const nextConfiguration = latestInitialConfigurationRef.current
    resetConfiguration(nextConfiguration)
    setSavedConfiguration(cloneConfigurationState(nextConfiguration))
    appliedConfigurationSourceKeyRef.current = configurationSourceKey
  }, [configurationSourceKey, resetConfiguration])
  const {
    selectedId,
    selectedNodeId,
    selectedContainerTarget,
    selectedAssignId,
    selectedAssign,
    catalogFor,
    catalogError,
    selectedCatalogNode,
    setNodeRef,
    setSelectedNodeId,
    closeCatalog,
    restorePreferredSelection,
    selectContainerTarget,
    openCatalogForNode,
    selectNodeByOffset,
    selectGpuSlotByOffset,
    selectModelInCurrentGpu,
    moveSelectedAssignByGpuOffset,
    removeAssignById,
    pickNodeAssignment,
    selectCatalogModel,
  } = useConfigurationPageSelection({ nodes: localNodes, assigns: localAssigns, models: displayData.catalog, initialConfiguration: localInitialConfiguration, preferredAssignId: displayData.preferredAssignId, setAssigns })

  const setNodePlacement = useCallback((nodeId: string, placement: Placement) => {
    updateConfiguration((current) => {
      if (nodeId !== localNodeId) return current
      const node = current.nodes.find((item) => item.id === nodeId)
      if (!node || !hasConfigurablePlacement(node) || node.placement === placement) return current

      const separatePlacementSnapshot = current.separatePlacementSnapshots[nodeId] ?? {}
      const nextSeparatePlacementSnapshots = placement === 'pooled' && node.placement === 'separate'
        ? { ...current.separatePlacementSnapshots, [nodeId]: createSeparatePlacementSnapshot(current.assigns, nodeId) }
        : current.separatePlacementSnapshots
      const nextNodes = current.nodes.map((item) => (item.id === nodeId ? { ...item, placement } : item))
      const nextNode = nextNodes.find((item) => item.id === nodeId) ?? node
      const nextAssigns = placement === 'pooled'
        ? current.assigns.map((assign) => (assign.nodeId === nodeId ? { ...assign, containerIdx: 0 } : assign))
        : restoreSeparatePlacement(current.assigns, nextNode, separatePlacementSnapshot, displayData.catalog)

      return {
        nodes: nextNodes,
        assigns: nextAssigns,
        defaultsValues: current.defaultsValues,
        separatePlacementSnapshots: nextSeparatePlacementSnapshots,
      }
    })
  }, [displayData.catalog, localNodeId, updateConfiguration])

  const hasInvalidNode = useMemo(() => hasInvalidAllocation(localNodes, localAssigns, displayData.catalog), [displayData.catalog, localAssigns, localNodes])
  const currentSnapshot = useMemo(() => createConfigurationSnapshot(nodes, assigns, defaultsValues), [assigns, defaultsValues, nodes])
  const savedSnapshot = useMemo(() => createConfigurationSnapshot(savedConfiguration.nodes, savedConfiguration.assigns, savedConfiguration.defaultsValues), [savedConfiguration])
  const hasUnsavedChanges = currentSnapshot !== savedSnapshot
  const defaultsDirty = useMemo(() => JSON.stringify(defaultsValues) !== JSON.stringify(savedConfiguration.defaultsValues), [defaultsValues, savedConfiguration.defaultsValues])
  const localDeploymentDirty = useMemo(
    () => createConfigurationSnapshot(nodes, assigns, savedConfiguration.defaultsValues) !== createConfigurationSnapshot(savedConfiguration.nodes, savedConfiguration.assigns, savedConfiguration.defaultsValues),
    [assigns, nodes, savedConfiguration],
  )

  const updateDefaultSetting = useCallback((settingId: string, value: string) => {
    updateConfiguration((current) => ({ ...current, defaultsValues: { ...current.defaultsValues, [settingId]: value } }))
  }, [updateConfiguration])

  const resetDefaultSettings = useCallback(() => {
    updateConfiguration((current) => ({ ...current, defaultsValues: initialDefaultsValues }))
  }, [initialDefaultsValues, updateConfiguration])

  const stepSelectedContext = useCallback((direction: -1 | 1, jumpToPower = false) => {
    if (!selectedAssign) return

    setAssigns((items) => items.map((assign) => (assign.id === selectedAssign.id ? { ...assign, ctx: jumpToPower ? jumpCtxPower(assign.ctx, direction) : stepCtx(assign.ctx, direction) } : assign)))
  }, [selectedAssign, setAssigns])

  const revertConfiguration = useCallback(() => {
    const restoredConfiguration = cloneConfigurationState(savedConfiguration)

    resetConfiguration(restoredConfiguration)
    restorePreferredSelection(restoredConfiguration, displayData.preferredAssignId)
  }, [displayData.preferredAssignId, resetConfiguration, restorePreferredSelection, savedConfiguration])

  const saveConfiguration = useCallback(() => {
    if (hasInvalidNode || !hasUnsavedChanges) return
    setSavedConfiguration(cloneConfigurationState(configuration))
  }, [configuration, hasInvalidNode, hasUnsavedChanges])
  const retryLiveData = useCallback(() => {
    void Promise.all([statusQuery.refetch(), modelsQuery.refetch()])
  }, [modelsQuery, statusQuery])
  const switchToTestData = useCallback(() => setMode('harness'), [setMode])

  const currentKeyboardNode = useMemo(
    () => localNodes.find((item) => item.id === (selectedNodeId ?? selectedAssign?.nodeId)) ?? null,
    [localNodes, selectedAssign, selectedNodeId],
  )

  const openCatalogForCurrentNode = useCallback(() => {
    if (currentKeyboardNode) openCatalogForNode(currentKeyboardNode)
  }, [currentKeyboardNode, openCatalogForNode])

  const setCurrentNodePlacement = useCallback((placement: Placement) => {
    if (!currentKeyboardNode || !hasConfigurablePlacement(currentKeyboardNode) || currentKeyboardNode.placement === placement) return false

    setNodePlacement(currentKeyboardNode.id, placement)
    return true
  }, [currentKeyboardNode, setNodePlacement])
  const ignoreReadOnlyAction = useCallback(() => undefined, [])

  useConfigurationPageKeyboardShortcuts({
    canUndo,
    canRedo,
    selectedAssignId,
    saveConfiguration,
    revertConfiguration,
    undoConfigurationChange,
    redoConfigurationChange,
    selectNodeByOffset,
    selectGpuSlotByOffset,
    selectModelInCurrentGpu,
    moveSelectedAssignByGpuOffset,
    stepSelectedContext,
    openCatalogForCurrentNode,
    setCurrentNodePlacement,
    removeSelectedAssign: removeAssignById,
  })

  const jump = (nodeId: string) => document.getElementById(`node-${nodeId}`)?.scrollIntoView({ block: 'start', behavior: 'smooth' })

  const keyboardHint: ReactNode = <KeyboardLegend />

  const rail = <NodeRail nodes={nodes} assigns={assigns} models={displayData.catalog} collapsedMap={collapsedMap} setCollapsedMap={setCollapsedMap} onJump={jump} keyboardHint={keyboardHint} />

  const localDeployment = (
    <ConfigurationDeploymentLayout rail={rail}>
      {localNodes.map((node) => (
        <div key={node.id} ref={(element) => { setNodeRef(node.id, element) }}>
          <NodeSection
            node={node}
            assigns={assigns}
            models={displayData.catalog}
            setAssigns={setAssigns}
            selectedId={selectedId}
            selectedContainerIdx={selectedContainerTarget?.nodeId === node.id ? selectedContainerTarget.containerIdx : null}
            selectedNode={selectedNodeId === node.id}
            onFocusNode={() => setSelectedNodeId(node.id)}
            onPick={(id) => pickNodeAssignment(node, id)}
            onSelectContainer={(containerIdx) => selectContainerTarget(node.id, getNodeTargetContainerIdx(node, containerIdx))}
            collapsed={Boolean(collapsedMap[node.id])}
            setCollapsed={(collapsed) => setCollapsedMap((map) => ({ ...map, [node.id]: collapsed }))}
            onOpenCatalog={openCatalogForNode}
            onPlacementChange={setNodePlacement}
          />
        </div>
      ))}
      {remoteNodes.length > 0 ? <ReadOnlyNodesDivider /> : null}
      {remoteNodes.map((node) => (
        <NodeSection
          key={node.id}
          node={node}
          assigns={assigns}
          models={displayData.catalog}
          setAssigns={setAssigns}
          selectedId={null}
          selectedContainerIdx={null}
          selectedNode={false}
          onFocusNode={ignoreReadOnlyAction}
          onPick={ignoreReadOnlyAction}
          onSelectContainer={ignoreReadOnlyAction}
          collapsed={Boolean(collapsedMap[node.id])}
          setCollapsed={(collapsed) => setCollapsedMap((map) => ({ ...map, [node.id]: collapsed }))}
          onOpenCatalog={ignoreReadOnlyAction}
          onPlacementChange={setNodePlacement}
          readOnly
        />
      ))}
    </ConfigurationDeploymentLayout>
  )

  const tabs: ConfigurationTabItem[] = [
    { id: 'defaults', label: 'Defaults', icon: Binary, dirty: defaultsDirty, content: <DefaultsTab data={displayData.defaults} values={defaultsValues} onResetAll={resetDefaultSettings} onSettingValueChange={updateDefaultSetting} configFilePath={displayData.configFilePath} /> },
    { id: 'local-deployment', label: 'Model Deployment', icon: Computer, dirty: localDeploymentDirty, content: localDeployment },
    ...(signingAttestationEnabled ? [{ id: 'signing', label: 'Signing / Attestation', icon: ShieldCheck, dirty: hasUnsavedChanges, content: <ConfigurationPlaceholderPanel title="Signing / Attestation" icon={ShieldCheck}>Unsigned local configuration. This pass reserves the review surface for key binding, attestation receipts, and dirty-state signing checks.</ConfigurationPlaceholderPanel> } satisfies ConfigurationTabItem] : []),
    ...(integrationsEnabled ? [{ id: 'integrations', label: 'Integrations', icon: Blocks, content: <ConfigurationPlaceholderPanel title="Integrations" icon={Blocks}>Plugin and external endpoint defaults will live here once the underlying TOML fields are available.</ConfigurationPlaceholderPanel> } satisfies ConfigurationTabItem] : []),
    { id: 'toml-review', label: 'TOML Output', icon: Brackets, dirty: hasUnsavedChanges, content: <TomlView nodes={localNodes} assigns={localAssigns} models={displayData.catalog} defaults={displayData.defaults} defaultsValues={defaultsValues} reviewMode configPath={displayData.configFilePath} validationWarnings={displayData.validationWarnings} launchSummaryConfig={displayData.launchSummaryConfig} /> },
  ]
  const renderedActiveTab = tabs.some((tab) => tab.id === activeTab) ? activeTab : 'defaults'

  const setActiveTab = useCallback((tab: ConfigurationTabId) => {
    if (controlledActiveTab === undefined) setActiveTabState(tab)
    onTabChange?.(tab)
  }, [controlledActiveTab, onTabChange])

  if (showLiveError) {
    return (
      <LiveDataUnavailableOverlay
        debugTitle="Could not reach live configuration sources"
        title="Live configuration is unavailable"
        debugDescription="Configuration could not fetch the initial status and model catalog from the configured API target. Start the backend, verify the endpoint, or switch Data source back to Harness in Tweaks while debugging."
        productionDescription="Configuration is waiting for live node and model data before rendering editable controls. Keep the page open while the service recovers, or switch Data source back to Harness in Tweaks to inspect sample configuration."
        onRetry={retryLiveData}
        onSwitchToTestData={switchToTestData}
      >
        <ConfigurationLiveLoadingGhost />
      </LiveDataUnavailableOverlay>
    )
  }

  if (showLiveLoading) {
    return <ConfigurationLiveLoadingGhost />
  }

  return (
    <>
      {showLiveRefresh ? <LiveRefreshPill className="mx-5 mb-2">Refreshing live configuration</LiveRefreshPill> : null}
      <ConfigurationLayout
        header={(
          <ConfigurationHeader
            title={displayData.title}
            description={displayData.description}
            nodes={nodes}
            canUndo={canUndo}
            canRedo={canRedo}
            hasUnsavedChanges={hasUnsavedChanges}
            hasInvalidNode={hasInvalidNode}
            onUndo={undoConfigurationChange}
            onRedo={redoConfigurationChange}
            onRevert={revertConfiguration}
            onSave={saveConfiguration}
          />
        )}
      >
        <ConfigurationTabs value={renderedActiveTab} onValueChange={setActiveTab} tabs={tabs} />
      </ConfigurationLayout>
      {enableNavigationBlocker ? <UnsavedConfigurationNavigationBlocker hasUnsavedChanges={hasUnsavedChanges} /> : null}
      {catalogFor && selectedCatalogNode ? <CatalogPopover open={Boolean(catalogFor)} onClose={closeCatalog} selectedNode={selectedCatalogNode} assigns={assigns} models={displayData.catalog} errorMessage={catalogError} onSelectModel={selectCatalogModel} /> : null}
    </>
  )
}

export function ConfigurationPage(props: ConfigurationPageProps = {}) {
  return (
    <QueryProvider>
      <ConfigurationPageContent {...props} />
    </QueryProvider>
  )
}
