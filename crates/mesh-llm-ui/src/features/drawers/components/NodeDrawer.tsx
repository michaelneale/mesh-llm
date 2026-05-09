import { useId } from 'react'
import { Activity, Cpu, HardDrive, Hash, Network, UserRound } from 'lucide-react'
import { StatusBadge } from '@/components/ui/StatusBadge'
import type { StatusBadgeTone } from '@/components/ui/StatusBadge'
import { ConfigNodeDrawerPanel } from '@/features/drawers/components/ConfigNodeDrawerPanel'
import { Drawer } from '@/features/drawers/components/Drawer'
import { drawerIcon } from '@/features/drawers/lib/badge-styles'
import { modelStatusBadge } from '@/features/drawers/lib/model-status'
import { DrawerHeader } from '@/features/drawers/components/DrawerHeader'
import { KV } from '@/features/drawers/components/KV'
import { SectionHead } from '@/features/drawers/components/SectionHead'
import { meshNodeStatusSource, meshStatusLabel, meshStatusTone } from '@/features/network/lib/mesh-status'
import { useLlamaRuntime } from '@/features/network/api/use-llama-runtime'
import { formatLatency } from '@/lib/format-latency'
import type { ConfigNode, MeshNode, ModelSummary, Peer } from '@/features/app-tabs/types'
import type { LlamaRuntimeMetricSample, LlamaRuntimePayload, LlamaRuntimeSlotItem } from '@/lib/api/types'

type DrawerNode = ConfigNode | MeshNode
type NodeDrawerProps = { open: boolean; node?: DrawerNode; peer?: Peer; models?: ModelSummary[]; onClose: () => void }

function isConfigNode(node: DrawerNode): node is ConfigNode {
  return 'gpus' in node && Array.isArray(node.gpus)
}

function isMeshNode(node: DrawerNode): node is MeshNode {
  return 'role' in node
}

function hardwareLabel(peer: Peer | undefined, node: MeshNode) {
  if (!peer) return node.subLabel ?? 'Mesh node'
  return peer.hardwareLabel ?? node.subLabel ?? 'Mesh node'
}

function modelForName(models: ModelSummary[], modelName: string) {
  return models.find((model) => model.name === modelName)
}

function runtimeBadgeLabel(status: string | undefined, loading: boolean, runtimeError: string | null) {
  if (loading && !status) return 'Loading'
  if (!status && runtimeError) return 'Unavailable'
  if (status === 'ready') return 'Live'
  if (status === 'error') return 'Error'
  if (status === 'unavailable') return 'Unavailable'
  return status ?? 'Unknown'
}

function runtimeBadgeTone(status: string | undefined, loading: boolean, runtimeError: string | null): StatusBadgeTone {
  if (loading && !status) return 'muted'
  if (!status && runtimeError) return 'bad'
  if (status === 'ready') return 'good'
  if (status === 'error' || status === 'unavailable') return 'bad'
  return 'muted'
}

function runtimeMetrics(runtime: LlamaRuntimePayload | null | undefined) {
  return runtime?.items?.metrics ?? runtime?.metrics.samples ?? []
}

function runtimeSlots(runtime: LlamaRuntimePayload | null | undefined): LlamaRuntimeSlotItem[] {
  if (runtime?.items?.slots) return runtime.items.slots
  const rawSlots = runtime?.slots.slots ?? []
  return rawSlots.map((slot, index) => ({
    index: slot.index ?? index,
    id: slot.id,
    id_task: slot.id_task,
    n_ctx: slot.n_ctx,
    is_processing: slot.is_processing ?? false
  }))
}

function runtimeSlotCounts(runtime: LlamaRuntimePayload | null | undefined) {
  const slots = runtimeSlots(runtime)
  return {
    total: runtime?.items?.slots_total ?? slots.length,
    busy: runtime?.items?.slots_busy ?? slots.filter((slot) => slot.is_processing).length
  }
}

function formatMetricName(item: LlamaRuntimeMetricSample) {
  const suffix = Object.values(item.labels ?? {})
    .filter(Boolean)
    .join(' · ')
  const name = item.name
    .replace(/^llamacpp:/, '')
    .replace(/^llama_/, '')
    .replace(/_/g, ' ')
  return suffix ? `${name} · ${suffix}` : name
}

function formatMetricValue(value: number) {
  if (!Number.isFinite(value)) return `${value}`
  if (Math.abs(value) >= 100) return value.toFixed(0)
  if (Math.abs(value) >= 10) return value.toFixed(1)
  return value.toFixed(2)
}

function slotContextWeight(slot: LlamaRuntimeSlotItem) {
  return typeof slot.n_ctx === 'number' && Number.isFinite(slot.n_ctx) && slot.n_ctx > 0 ? slot.n_ctx : 1
}

function formatSlotContext(slot: LlamaRuntimeSlotItem) {
  if (slot.n_ctx == null) return 'n/a'
  return slot.id_task != null ? `${slot.n_ctx} · task ${slot.id_task}` : `${slot.n_ctx}`
}

function formatSlotIndex(slot: LlamaRuntimeSlotItem) {
  return slot.id != null && slot.id !== slot.index ? `#${slot.index} · id ${slot.id}` : `#${slot.index}`
}

function runtimeMetricError(runtime: LlamaRuntimePayload | null | undefined, fallbackError: string | null) {
  return runtime?.metrics.error ?? fallbackError
}

function runtimeSlotsError(runtime: LlamaRuntimePayload | null | undefined, fallbackError: string | null) {
  return runtime?.slots.error ?? fallbackError
}

function meshRoleLabel(peer: Peer | undefined, node: MeshNode) {
  if (peer?.role === 'you' || node.role === 'self') return 'You'
  if (peer?.role === 'host' || node.host) return 'Host'
  if (peer?.role === 'client' || node.client || node.renderKind === 'client') return 'Client'
  if (peer?.role === 'worker' || node.renderKind === 'worker') return 'Worker'
  return 'Peer'
}

function meshRoleTone(role: string): StatusBadgeTone {
  if (role === 'You') return 'accent'
  if (role === 'Client' || role === 'Peer') return 'muted'
  if (role === 'Worker') return 'warn'
  return 'good'
}

export function NodeDrawer({ open, node, peer, models = [], onClose }: NodeDrawerProps) {
  const titleId = useId()
  const runtime = useLlamaRuntime(Boolean(open && node && isMeshNode(node) && node.role === 'self'))

  return (
    <Drawer ariaLabel="Node details" labelledBy={node ? titleId : undefined} open={open} onClose={onClose}>
      {node ? (
        <NodeDrawerContent
          node={node}
          peer={peer}
          models={models}
          onClose={onClose}
          runtime={runtime}
          titleId={titleId}
        />
      ) : (
        <div className="px-[18px] py-4 text-[length:var(--density-type-control)] text-fg-faint">No node selected.</div>
      )}
    </Drawer>
  )
}

function RuntimeMetricsTable({ metrics, loading }: { metrics: LlamaRuntimeMetricSample[]; loading: boolean }) {
  if (metrics.length === 0) {
    return (
      <p className="text-[length:var(--density-type-label)] text-fg-faint">
        {loading ? 'Loading live metrics…' : 'No metric samples reported yet.'}
      </p>
    )
  }
  return (
    <div className="overflow-hidden rounded-[var(--radius)] border border-border-soft bg-background">
      <div className="grid grid-cols-[1fr_auto] bg-panel-strong px-3 py-2 text-[length:var(--density-type-annotation)] font-medium uppercase tracking-[0.5px] text-fg-faint">
        <div>Metric</div>
        <div>Value</div>
      </div>
      {metrics.map((item) => {
        const labelParts = Object.entries(item.labels ?? {})
          .map(([k, v]) => `${k}=${v}`)
          .join(', ')
        return (
          <div
            className="grid grid-cols-[1fr_auto] items-center border-t border-border-soft px-3 py-[7px]"
            key={`${item.name}:${JSON.stringify(item.labels ?? {})}`}
            title={labelParts ? `${item.name} (${labelParts})` : item.name}
          >
            <span className="truncate text-[length:var(--density-type-label)] text-fg-dim">
              {formatMetricName(item)}
            </span>
            <span className="ml-4 font-mono text-[length:var(--density-type-label)] tabular-nums text-foreground">
              {formatMetricValue(item.value)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function LlamaSlotContextMap({
  slots,
  slotsBusy,
  slotsTotal
}: {
  slots: LlamaRuntimeSlotItem[]
  slotsBusy: number
  slotsTotal: number
}) {
  return (
    <div className="space-y-2 rounded-[var(--radius)] border border-border-soft bg-panel p-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[length:var(--density-type-label)] font-medium text-foreground">Slot context map</span>
        <div className="flex flex-wrap items-center gap-3 text-[length:var(--density-type-label)] text-fg-faint">
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-full" style={{ background: 'var(--color-good)' }} />
            Available
          </span>
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-full" style={{ background: 'var(--color-warn)' }} />
            Active
          </span>
          <span className="font-mono text-[length:var(--density-type-annotation)] text-foreground">
            {slotsBusy}/{slotsTotal}
          </span>
        </div>
      </div>
      <ul
        aria-label={`Llama slot context map. ${slotsBusy} of ${slotsTotal} slots active.`}
        className="flex min-h-7 list-none gap-px overflow-hidden rounded-[var(--radius)] border border-border-soft bg-background"
      >
        {slots.map((slot) => {
          const stateLabel = slot.is_processing ? 'Active' : 'Available'
          const contextLabel = formatSlotContext(slot)
          const label = `${formatSlotIndex(slot)} · ${stateLabel} · context ${contextLabel}`
          return (
            <li
              className="min-w-[40px]"
              key={slot.id ?? slot.index}
              style={{ flexBasis: 0, flexGrow: slotContextWeight(slot) }}
            >
              <button
                aria-label={label}
                className="flex h-full min-h-6 w-full items-center justify-center overflow-hidden px-2 font-mono text-[length:var(--density-type-annotation)] font-semibold tabular-nums transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset"
                style={
                  slot.is_processing
                    ? {
                        background: 'color-mix(in oklab, var(--color-warn) 22%, var(--color-background))',
                        color: 'var(--color-warn)'
                      }
                    : {
                        background: 'color-mix(in oklab, var(--color-good) 22%, var(--color-background))',
                        color: 'var(--color-good)'
                      }
                }
                title={label}
                type="button"
              >
                <span className="truncate">
                  {formatSlotIndex(slot)} · {contextLabel}
                </span>
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function NodeDrawerContent({
  node,
  peer,
  models,
  runtime,
  onClose,
  titleId
}: {
  node: DrawerNode
  peer?: Peer
  models: ModelSummary[]
  runtime: ReturnType<typeof useLlamaRuntime>
  onClose: () => void
  titleId: string
}) {
  if (isConfigNode(node)) {
    return <ConfigNodeDrawerPanel node={node} onClose={onClose} titleId={titleId} />
  }

  const title = peer?.hostname ?? node.label
  const nodeId = peer?.shortId ?? node.id
  const role = meshRoleLabel(peer, node)
  const statusSource = meshNodeStatusSource(peer, node)

  return (
    <div>
      <DrawerHeader
        badges={
          <>
            <StatusBadge tone={meshRoleTone(role)}>{role}</StatusBadge>
            <StatusBadge dot tone={meshStatusTone(statusSource)}>
              {meshStatusLabel(statusSource, { online: 'Online', degraded: 'Degraded' })}
            </StatusBadge>
          </>
        }
        onClose={onClose}
        subtitle={nodeId}
        title={title}
        titleId={titleId}
      />

      <div className="pb-6 pt-3">
        <h3 className="sr-only">Node metadata</h3>
        <div className="flex gap-2 px-[18px]">
          <KV icon={drawerIcon(Activity)} label="Latency">
            {peer ? `${formatLatency(peer.latencyMs)} ms` : 'N/A'}
          </KV>
          <KV icon={drawerIcon(HardDrive)} label="Node VRAM">
            {peer?.vramGB != null ? `${peer.vramGB.toFixed(1)} GB` : 'N/A'}
          </KV>
          <KV icon={drawerIcon(Network)} label="Mesh share">
            {peer ? `${peer.sharePct}%` : 'N/A'}
          </KV>
          <KV icon={drawerIcon(Cpu)} label="Models">
            {peer?.hostedModels.length ?? 0}
          </KV>
        </div>

        {peer ? (
          <>
            <h3 className="sr-only">Hosted models</h3>
            <SectionHead icon={drawerIcon(Cpu)}>Models</SectionHead>
            <div className="mx-[18px] overflow-hidden rounded-[var(--radius)] border border-border-soft bg-background">
              <div className="grid grid-cols-[1.6fr_1fr_0.6fr] bg-panel-strong px-3 py-2 text-[length:var(--density-type-label)] font-medium uppercase tracking-[0.5px] text-fg-faint">
                <div>Model</div>
                <div>Role</div>
                <div>Mesh</div>
              </div>

              {peer.hostedModels.map((modelName, index) => {
                const hostedModel = modelForName(models, modelName)
                const hostedStatus = modelStatusBadge(hostedModel?.status)
                return (
                  <div
                    className="grid grid-cols-[1.6fr_1fr_0.6fr] items-center border-t border-border-soft px-3 py-[9px]"
                    key={modelName}
                  >
                    <span className="truncate font-mono text-[length:var(--density-type-control)]">{modelName}</span>
                    <div className="flex flex-wrap gap-1">
                      {index === 0 ? <StatusBadge tone="good">Serving</StatusBadge> : null}
                      <StatusBadge tone="accent">Hosted</StatusBadge>
                    </div>
                    <StatusBadge dot tone={hostedStatus.tone}>
                      {hostedStatus.label}
                    </StatusBadge>
                  </div>
                )
              })}
            </div>

            <SectionHead icon={drawerIcon(Cpu)}>Hardware</SectionHead>
            <div className="space-y-2 px-[18px]">
              <KV icon={drawerIcon(Hash)} label="Hostname">
                {peer.hostname}
              </KV>
              <KV icon={drawerIcon(Hash)} label="Version">
                {peer.version ? `v${peer.version}` : 'N/A'}
              </KV>
              <KV icon={drawerIcon(Cpu)} label="Device">
                {hardwareLabel(peer, node)}
              </KV>
            </div>
          </>
        ) : null}

        {node.role === 'self' ? (
          <>
            <SectionHead icon={drawerIcon(Activity)}>Runtime</SectionHead>
            <div className="space-y-2 px-[18px]">
              <div className="flex flex-wrap gap-2">
                <StatusBadge tone={runtimeBadgeTone(runtime.data?.metrics.status, runtime.loading, runtime.error)}>
                  Metrics • {runtimeBadgeLabel(runtime.data?.metrics.status, runtime.loading, runtime.error)}
                </StatusBadge>
                <StatusBadge tone={runtimeBadgeTone(runtime.data?.slots.status, runtime.loading, runtime.error)}>
                  Slots • {runtimeBadgeLabel(runtime.data?.slots.status, runtime.loading, runtime.error)}
                </StatusBadge>
                <StatusBadge tone={runtimeSlotCounts(runtime.data).busy > 0 ? 'warn' : 'good'}>
                  {runtimeSlotCounts(runtime.data).busy}/{runtimeSlotCounts(runtime.data).total} slots busy
                </StatusBadge>
              </div>

              {!runtime.data && runtime.error ? (
                <p className="text-[length:var(--density-type-label)] text-fg-faint">
                  Runtime unavailable: {runtime.error}
                </p>
              ) : null}

              {runtime.data && runtimeMetricError(runtime.data, runtime.error) ? (
                <p className="text-[length:var(--density-type-label)] text-fg-faint">
                  Metrics: {runtimeMetricError(runtime.data, runtime.error)}
                </p>
              ) : null}

              {runtime.data &&
              runtimeSlotsError(runtime.data, runtime.error) &&
              runtimeSlotsError(runtime.data, runtime.error) !== runtimeMetricError(runtime.data, runtime.error) ? (
                <p className="text-[length:var(--density-type-label)] text-fg-faint">
                  Slots: {runtimeSlotsError(runtime.data, runtime.error)}
                </p>
              ) : null}

              <RuntimeMetricsTable metrics={runtimeMetrics(runtime.data)} loading={runtime.loading} />
              {runtimeSlots(runtime.data).length > 0 ? (
                <LlamaSlotContextMap
                  slots={runtimeSlots(runtime.data)}
                  slotsBusy={runtimeSlotCounts(runtime.data).busy}
                  slotsTotal={runtimeSlotCounts(runtime.data).total}
                />
              ) : null}
            </div>
          </>
        ) : null}

        <SectionHead icon={drawerIcon(UserRound)}>Ownership</SectionHead>
        <div className="space-y-2 px-[18px]">
          <p className="text-[length:var(--density-type-caption)] leading-5 text-fg-faint">
            Whether this node&apos;s identity is cryptographically bound to a stable owner.
          </p>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            <KV label="Ownership">{peer?.ownership ?? 'Unknown'}</KV>
            <KV label="Owner">{peer?.owner ?? 'Unknown'}</KV>
            <KV label="Node ID">{nodeId}</KV>
          </div>
        </div>
      </div>
    </div>
  )
}
