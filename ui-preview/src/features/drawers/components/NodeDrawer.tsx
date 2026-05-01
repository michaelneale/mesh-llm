import { useId } from 'react'
import { Activity, Cpu, HardDrive, Hash, Network, UserRound } from 'lucide-react'
import { Drawer } from '@/features/drawers/components/Drawer'
import { DrawerBadge } from '@/features/drawers/components/DrawerBadge'
import { drawerIcon } from '@/features/drawers/lib/badge-styles'
import type { DrawerBadgeTone } from '@/features/drawers/lib/badge-styles'
import { DrawerHeader } from '@/features/drawers/components/DrawerHeader'
import { KV } from '@/features/drawers/components/KV'
import { SectionHead } from '@/features/drawers/components/SectionHead'
import { nodeTotalGB } from '@/features/configuration/lib/config-math'
import { formatLatency } from '@/lib/format-latency'
import type { ConfigNode, MeshNode, ModelSummary, Peer } from '@/features/app-tabs/types'

type DrawerNode = ConfigNode | MeshNode
type NodeDrawerProps = { open: boolean; node?: DrawerNode; peer?: Peer; models?: ModelSummary[]; onClose: () => void }

function isConfigNode(node: DrawerNode): node is ConfigNode {
  return 'gpus' in node && Array.isArray(node.gpus)
}

function statusTone(status: ConfigNode['status'] | MeshNode['status']): DrawerBadgeTone {
  if (status === 'online') return 'good'
  if (status === 'degraded') return 'warn'
  return 'bad'
}

function hardwareLabel(peer: Peer | undefined, node: MeshNode) {
  if (!peer) return node.subLabel ?? 'Mesh node'
  return peer.hardwareLabel ?? node.subLabel ?? 'Mesh node'
}

function modelForName(models: ModelSummary[], modelName: string) {
  return models.find((model) => model.name === modelName)
}

function modelStatusBadge(status: ModelSummary['status'] | undefined): { label: string; tone: DrawerBadgeTone } {
  if (status === 'offline') return { label: 'Offline', tone: 'bad' }
  if (status === 'warming') return { label: 'Warming', tone: 'warn' }
  if (status === 'ready') return { label: 'Ready', tone: 'good' }
  if (status === 'warm') return { label: 'Warm', tone: 'good' }
  return { label: 'Unknown', tone: 'muted' }
}

export function NodeDrawer({ open, node, peer, models = [], onClose }: NodeDrawerProps) {
  const titleId = useId()

  return (
    <Drawer ariaLabel="Node details" labelledBy={node ? titleId : undefined} open={open} onClose={onClose}>
      {node ? (
        <NodeDrawerContent node={node} peer={peer} models={models} onClose={onClose} titleId={titleId} />
      ) : (
        <div className="px-[18px] py-4 text-[length:var(--density-type-control)] text-fg-faint">No node selected.</div>
      )}
    </Drawer>
  )
}

function NodeDrawerContent({
  node,
  peer,
  models,
  onClose,
  titleId,
}: {
  node: DrawerNode
  peer?: Peer
  models: ModelSummary[]
  onClose: () => void
  titleId: string
}) {
  if (isConfigNode(node)) {
    return (
      <div>
        <DrawerHeader
          badges={
            <>
              <DrawerBadge tone={statusTone(node.status)}>{node.status}</DrawerBadge>
              <DrawerBadge tone="muted">{node.placement}</DrawerBadge>
            </>
          }
          onClose={onClose}
          subtitle={node.region}
          title={node.hostname}
          titleId={titleId}
        />

        <div className="pb-5 pt-3">
          <SectionHead icon={drawerIcon(Cpu)}>Node metadata</SectionHead>
          <div className="grid grid-cols-2 gap-2 px-[18px]">
            <KV icon={drawerIcon(Cpu)} label="CPU">{node.cpu}</KV>
            <KV icon={drawerIcon(HardDrive)} label="RAM">{node.ramGB} GB</KV>
            <KV icon={drawerIcon(HardDrive)} label="VRAM">{nodeTotalGB(node)} GB</KV>
            <KV icon={drawerIcon(Cpu)} label="GPUs">{node.gpus.length}</KV>
          </div>

          <SectionHead icon={drawerIcon(Cpu)}>Accelerators</SectionHead>
          <div className="space-y-1.5 px-[18px]">
            {node.gpus.map((gpu) => (
              <div
                className="rounded-[var(--radius)] border border-border-soft bg-background px-3 py-2 text-[length:var(--density-type-control)]"
                key={gpu.idx}
              >
                <div className="font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim">GPU {gpu.idx}</div>
                <div className="mt-0.5 text-[length:var(--density-type-control)] text-foreground">{gpu.name}</div>
                <div className="mt-1 font-mono text-[length:var(--density-type-label)] text-fg-faint">{gpu.totalGB} GB total</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const title = peer?.hostname ?? node.label
  const nodeId = peer?.shortId ?? node.id
  const role = peer?.role === 'you' || node.role === 'self' ? 'You' : 'Host'

  return (
    <div>
      <DrawerHeader
        badges={
          <>
            <DrawerBadge tone={role === 'You' ? 'accent' : 'muted'}>{role}</DrawerBadge>
            <DrawerBadge dot tone="good">Serving</DrawerBadge>
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
          <KV icon={drawerIcon(Activity)} label="Latency">{peer ? `${formatLatency(peer.latencyMs)} ms` : 'N/A'}</KV>
          <KV icon={drawerIcon(HardDrive)} label="Node VRAM">{peer?.vramGB != null ? `${peer.vramGB.toFixed(1)} GB` : 'N/A'}</KV>
          <KV icon={drawerIcon(Network)} label="Mesh share">{peer ? `${peer.sharePct}%` : 'N/A'}</KV>
          <KV icon={drawerIcon(Cpu)} label="Models">{peer?.hostedModels.length ?? 0}</KV>
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
                  <div className="grid grid-cols-[1.6fr_1fr_0.6fr] items-center border-t border-border-soft px-3 py-[9px]" key={modelName}>
                    <span className="truncate font-mono text-[length:var(--density-type-control)]">{modelName}</span>
                    <div className="flex flex-wrap gap-1">
                      {index === 0 ? <DrawerBadge tone="good">Serving</DrawerBadge> : null}
                      <DrawerBadge tone="accent">Hosted</DrawerBadge>
                    </div>
                    <DrawerBadge dot tone={hostedStatus.tone}>{hostedStatus.label}</DrawerBadge>
                  </div>
                )
              })}
            </div>

            <SectionHead icon={drawerIcon(Cpu)}>Hardware</SectionHead>
            <div className="space-y-2 px-[18px]">
              <KV icon={drawerIcon(Hash)} label="Hostname">{peer.hostname}</KV>
              <KV icon={drawerIcon(Hash)} label="Version">{peer.version ? `v${peer.version}` : 'N/A'}</KV>
              <KV icon={drawerIcon(Cpu)} label="Device">{hardwareLabel(peer, node)}</KV>
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
