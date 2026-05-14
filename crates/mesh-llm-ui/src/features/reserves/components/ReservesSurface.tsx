import { animate, createScope, stagger } from 'animejs'
import { Activity } from 'lucide-react'
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'
import { Badge } from '@/components/ui/badge'
import { InfoBanner } from '@/components/ui/InfoBanner'
import { DEFAULT_RESERVE_WAKE_POLICY_SETTINGS } from '@/features/reserves/lib/reserve-policy'
import { AddReserveProviderDialog } from '@/features/reserves/components/AddReserveProviderDialog'
import { ReserveActionDialog } from '@/features/reserves/components/ReserveActionDialog'
import { ReserveFleetPanel } from '@/features/reserves/components/ReserveFleetPanel'
import { ReservePolicyPanel } from '@/features/reserves/components/ReservePolicyPanel'
import { getMeshProvider } from '@/features/reserves/lib/mesh-providers'
import { DEFAULT_PROVIDER_DRAFT, type ProviderDraft } from '@/features/reserves/lib/provider-draft'
import type { ReserveNode, ReserveProvider, ReserveWakePolicySettings } from '@/features/reserves/lib/reserve-types'

type ReservesSurfaceProps = {
  configurationHref?: string
  liveMeshVramGB?: number
  providers: ReserveProvider[]
}

type ReserveSurfaceAction =
  | { kind: 'wake-provider'; providerId: string }
  | { kind: 'retry-all'; providerId: string }
  | { kind: 'retry-node'; providerId: string; nodeId: string }
  | { kind: 'logs'; providerId: string; nodeId: string }
  | { kind: 'dismiss'; providerId: string; nodeId: string }

function cloneProviders(providers: ReserveProvider[]) {
  return providers.map((provider) => ({
    ...provider,
    tags: provider.tags ? [...provider.tags] : undefined,
    nodes: provider.nodes.map((node) => ({ ...node, models: [...node.models] }))
  }))
}

function slugify(value: string) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function updateProviderNodes(
  providers: ReserveProvider[],
  providerId: string,
  updater: (nodes: ReserveNode[]) => ReserveNode[]
) {
  return providers.map((provider) =>
    provider.id === providerId
      ? {
          ...provider,
          nodes: updater(provider.nodes)
        }
      : provider
  )
}

function promoteNodeToWake(node: ReserveNode): ReserveNode {
  return {
    ...node,
    state: 'waking',
    progress: node.progress ?? 12,
    eta: node.eta ?? 240,
    error: undefined,
    failedAt: undefined,
    lastSeen: undefined,
    note: 'Preview wake requested. No backend request was sent.'
  }
}

export function ReservesSurface({ configurationHref, liveMeshVramGB, providers }: ReservesSurfaceProps) {
  const navigate = useNavigate()
  const rootRef = useRef<HTMLDivElement | null>(null)
  const [sourceProviders, setSourceProviders] = useState(providers)
  const [surfaceProviders, setSurfaceProviders] = useState<ReserveProvider[]>(() => cloneProviders(providers))
  const [wakePolicySettings, setWakePolicySettings] = useState<ReserveWakePolicySettings>(
    DEFAULT_RESERVE_WAKE_POLICY_SETTINGS
  )
  const [action, setAction] = useState<ReserveSurfaceAction | null>(null)
  const [addProviderOpen, setAddProviderOpen] = useState(false)
  const [providerDraft, setProviderDraft] = useState<ProviderDraft>(DEFAULT_PROVIDER_DRAFT)

  if (sourceProviders !== providers) {
    setSourceProviders(providers)
    setSurfaceProviders(cloneProviders(providers))
  }

  useLayoutEffect(() => {
    if (!rootRef.current) return undefined

    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches

    if (prefersReducedMotion) return undefined

    const scope = createScope({ root: rootRef }).add(() => {
      animate('[data-reserve-entrance]', {
        opacity: [0, 1],
        y: [10, 0],
        duration: 260,
        delay: stagger(45),
        ease: 'out(4)'
      })
    })

    return () => scope.revert()
  }, [])

  const selectedProvider = useMemo(
    () => (action ? (surfaceProviders.find((provider) => provider.id === action.providerId) ?? null) : null),
    [action, surfaceProviders]
  )
  const selectedNode = useMemo(
    () =>
      action && 'nodeId' in action ? (selectedProvider?.nodes.find((node) => node.id === action.nodeId) ?? null) : null,
    [action, selectedProvider]
  )

  function resetProviderDraft() {
    setProviderDraft(DEFAULT_PROVIDER_DRAFT)
  }

  function handleAddProvider() {
    const selectedMeshProvider = getMeshProvider(providerDraft.providerId)
    if (selectedMeshProvider.availability !== 'supported') return

    const normalizedName = providerDraft.name.trim() || selectedMeshProvider.defaultName
    const normalizedRegion = providerDraft.region.trim() || selectedMeshProvider.defaultRegion
    const idBase = slugify(normalizedName) || 'preview-provider'

    setSurfaceProviders((currentProviders) => [
      {
        id: `${idBase}-${currentProviders.length + 1}`,
        name: normalizedName,
        kind: selectedMeshProvider.kind,
        icon: selectedMeshProvider.icon,
        region: normalizedRegion,
        tags: [normalizedRegion],
        billing: selectedMeshProvider.billing,
        summary: selectedMeshProvider.summary,
        nodes: [
          {
            id: `${idBase}-01`,
            hw: 'RTX 4090',
            vram: 24,
            location: `${normalizedRegion} · preview reserve`,
            note: 'This node exists only inside the UI preview and does not wake a real machine.',
            models: ['Qwen3-14B'],
            state: 'standby',
            since: 'just added'
          }
        ]
      },
      ...currentProviders
    ])
    resetProviderDraft()
  }

  function handleActionConfirm() {
    if (!action) return

    setSurfaceProviders((currentProviders) => {
      switch (action.kind) {
        case 'wake-provider':
          return updateProviderNodes(currentProviders, action.providerId, (nodes) => {
            const standbyIndex = nodes.findIndex((node) => node.state === 'standby')
            if (standbyIndex === -1) return nodes

            return nodes.map((node, index) => (index === standbyIndex ? promoteNodeToWake(node) : node))
          })
        case 'retry-all':
          return updateProviderNodes(currentProviders, action.providerId, (nodes) =>
            nodes.map((node) =>
              node.state === 'failed' || node.state === 'unreachable' ? promoteNodeToWake(node) : node
            )
          )
        case 'retry-node':
          return updateProviderNodes(currentProviders, action.providerId, (nodes) =>
            nodes.map((node) => (node.id === action.nodeId ? promoteNodeToWake(node) : node))
          )
        case 'dismiss':
          return updateProviderNodes(currentProviders, action.providerId, (nodes) =>
            nodes.filter((node) => node.id !== action.nodeId)
          )
        case 'logs':
          return currentProviders
      }
    })
  }

  const openConfigurationTab = useCallback(() => {
    if (!configurationHref) return

    void navigate({ to: '/configuration/$configurationTab', params: { configurationTab: 'wake-policy' } })
  }, [configurationHref, navigate])

  function actionDialogCopy() {
    if (!action || !selectedProvider) return null

    if (action.kind === 'wake-provider') {
      return {
        title: `Wake ${selectedProvider.name}`,
        description: `Queue the next standby node from ${selectedProvider.name}. This preview updates the panel locally and skips backend provisioning calls.`,
        confirmLabel: 'Queue wake',
        confirmTone: 'default' as const
      }
    }

    if (action.kind === 'retry-all') {
      return {
        title: `Retry all wake failures for ${selectedProvider.name}`,
        description:
          'Re-queue every failed or unreachable node in this provider group. This preview only flips the visible state chips and progress rows.',
        confirmLabel: 'Retry all',
        confirmTone: 'default' as const
      }
    }

    if (!selectedNode) return null

    if (action.kind === 'retry-node') {
      return {
        title: `Retry ${selectedNode.id}`,
        description:
          'Re-run the wake attempt for this node without contacting a provider API. The mockup moves the row back into active wake state.',
        confirmLabel: 'Retry node',
        confirmTone: 'default' as const
      }
    }

    if (action.kind === 'logs') {
      return {
        title: `Logs for ${selectedNode.id}`,
        description:
          'Inspect the last visible wake failure details. This preview keeps log text inline until backend log streaming exists.',
        confirmLabel: 'Close',
        confirmTone: 'default' as const
      }
    }

    return {
      title: `Dismiss ${selectedNode.id}`,
      description:
        'Hide this failed reserve row from the current UI view. This is a preview-only dismissal and does not change backend state.',
      confirmLabel: 'Dismiss node',
      confirmTone: 'destructive' as const
    }
  }

  const dialogCopy = actionDialogCopy()

  return (
    <div className="flex min-w-0 flex-col gap-[14px]" ref={rootRef}>
      <div data-reserve-entrance data-testid="reserves-hero">
        <InfoBanner
          action={
            <div className="flex flex-wrap items-center gap-3 text-[12px] leading-none text-fg-dim sm:text-[12.5px]">
              <a className="ui-link font-medium" href="#reserve-policy">
                Reserve policy →
              </a>
              <span aria-hidden="true" className="text-fg-faint">
                ·
              </span>
              <button
                className="ui-link-muted font-normal text-fg-dim"
                onClick={() => setAddProviderOpen(true)}
                type="button"
              >
                Add provider →
              </button>
            </div>
          }
          actionClassName="basis-full justify-start pl-[46px] pt-1 sm:basis-auto sm:justify-end sm:pl-0 sm:pt-0"
          className="min-h-[68px] flex-wrap items-start gap-3 rounded-[var(--radius)] px-[18px] py-[14px] sm:flex-nowrap sm:items-center"
          description="Off-mesh nodes you can wake on demand. Cloud VMs, colocated hosts, and office workstations join the mesh when demand rises, then step back when queues clear."
          descriptionClassName="mt-0.5 text-[12px] leading-[1.45]"
          leadingIcon={<Activity className="size-4" aria-hidden="true" />}
          title="Reserves"
          titleClassName="text-[13.5px] font-semibold leading-tight tracking-normal"
          titleLevel="h1"
        />
      </div>

      <div data-reserve-entrance id="reserve-fleet">
        <ReserveFleetPanel
          liveMeshVramGB={liveMeshVramGB}
          onDismissNode={(provider, node) => setAction({ kind: 'dismiss', providerId: provider.id, nodeId: node.id })}
          onOpenLogs={(provider, node) => setAction({ kind: 'logs', providerId: provider.id, nodeId: node.id })}
          onRetryAll={(provider) => setAction({ kind: 'retry-all', providerId: provider.id })}
          onRetryNode={(provider, node) => setAction({ kind: 'retry-node', providerId: provider.id, nodeId: node.id })}
          onWakeProvider={(provider) => setAction({ kind: 'wake-provider', providerId: provider.id })}
          providers={surfaceProviders}
          wakePolicySettings={wakePolicySettings}
        />
      </div>

      <div data-reserve-entrance id="reserve-policy">
        <ReservePolicyPanel
          mode="reserves"
          onOpenConfigurationTab={configurationHref ? openConfigurationTab : undefined}
          onSettingsChange={setWakePolicySettings}
          providers={surfaceProviders.map((p) => p.name)}
          settings={wakePolicySettings}
        />
      </div>

      <AddReserveProviderDialog
        confirmLabel="Add preview provider"
        description="Stage a reserve provider row in the local preview. No infrastructure is provisioned."
        onDraftChange={setProviderDraft}
        onConfirm={handleAddProvider}
        onOpenChange={(open) => {
          setAddProviderOpen(open)
          if (!open) resetProviderDraft()
        }}
        open={addProviderOpen}
        providerDraft={providerDraft}
      />

      {dialogCopy ? (
        <ReserveActionDialog
          confirmLabel={dialogCopy.confirmLabel}
          confirmTone={dialogCopy.confirmTone}
          description={dialogCopy.description}
          onConfirm={handleActionConfirm}
          onOpenChange={(open) => {
            if (!open) setAction(null)
          }}
          open={action !== null}
          showCancel={action?.kind !== 'logs'}
          title={dialogCopy.title}
        >
          {selectedProvider ? (
            <div className="space-y-3 rounded-[var(--radius)] border border-border bg-background px-3.5 py-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge className="rounded-full px-2 py-0.5 text-[10px] uppercase tracking-[0.08em] text-fg-dim">
                  {selectedProvider.name}
                </Badge>
                {selectedNode ? (
                  <Badge className="rounded-full px-2 py-0.5 text-[10px] uppercase tracking-[0.08em] text-fg-dim">
                    {selectedNode.id}
                  </Badge>
                ) : null}
              </div>
              {selectedNode ? (
                <div className="space-y-1">
                  <div className="text-[length:var(--density-type-control-lg)] font-medium text-foreground">
                    {selectedNode.hw} · {selectedNode.vram} GB
                  </div>
                  {selectedNode.location ? (
                    <div className="type-caption text-fg-faint">{selectedNode.location}</div>
                  ) : null}
                  {selectedNode.error ? <div className="type-caption text-fg-dim">{selectedNode.error}</div> : null}
                </div>
              ) : (
                <div className="type-caption text-fg-dim">
                  The action will target the next standby reserve in this provider group.
                </div>
              )}
              {action?.kind === 'logs' && selectedNode ? (
                <div className="rounded-[var(--radius)] border border-border bg-panel px-3 py-2 font-mono text-[length:var(--density-type-caption)] text-fg-dim">
                  [{selectedNode.id}] wake attempt timed out while waiting for provider health checks. retryable=
                  {String(selectedNode.retryable !== false)}
                </div>
              ) : null}
            </div>
          ) : null}
        </ReserveActionDialog>
      ) : null}
    </div>
  )
}
