import { useId } from 'react'
import { Cpu, HardDrive, Hash, Network } from 'lucide-react'
import { Drawer } from '@/features/drawers/components/Drawer'
import { DrawerBadge } from '@/features/drawers/components/DrawerBadge'
import { drawerIcon } from '@/features/drawers/lib/badge-styles'
import type { DrawerBadgeTone } from '@/features/drawers/lib/badge-styles'
import { DrawerHeader } from '@/features/drawers/components/DrawerHeader'
import { KV } from '@/features/drawers/components/KV'
import { SectionHead } from '@/features/drawers/components/SectionHead'
import { formatLatency } from '@/lib/format-latency'
import type { ConfigModel, ModelSummary, Peer } from '@/features/app-tabs/types'

type DrawerModel = ConfigModel | ModelSummary
type ModelDrawerProps = { open: boolean; model?: DrawerModel; peers?: Peer[]; onClose: () => void }

function isConfigModel(model: DrawerModel): model is ConfigModel {
  return 'id' in model
}

function modelSubtitle(model: DrawerModel) {
  return isConfigModel(model) ? model.family : model.fullId ?? model.family
}

function modelQuant(model: ModelSummary) {
  if (model.quant) return model.quant
  if (!model.fullId?.startsWith(`${model.name}-`)) return 'Q4_K_XL'
  return model.fullId.slice(model.name.length + 1)
}

function modelSummarySize(model: ModelSummary) {
  return model.sizeGB === undefined ? model.size : `${model.sizeGB} GB`
}

function modelSummaryContext(model: ModelSummary) {
  return model.ctxMaxK === undefined ? model.context : `${model.ctxMaxK}k`
}

function peersForModel(model: ModelSummary, peers: Peer[] = []) {
  return peers.filter((peer) => peer.hostedModels.includes(model.name))
}

function modelStatusBadge(status: ModelSummary['status']): { label: string; tone: DrawerBadgeTone } {
  if (status === 'offline') return { label: 'Offline', tone: 'bad' }
  if (status === 'warming') return { label: 'Warming', tone: 'warn' }
  if (status === 'ready') return { label: 'Ready', tone: 'good' }
  return { label: 'Warm', tone: 'good' }
}

export function ModelDrawer({ open, model, peers = [], onClose }: ModelDrawerProps) {
  const titleId = useId()

  return (
    <Drawer ariaLabel="Model details" labelledBy={model ? titleId : undefined} open={open} onClose={onClose}>
      {model ? (
        <ModelDrawerContent model={model} onClose={onClose} peers={peers} titleId={titleId} />
      ) : (
        <div className="px-[18px] py-4 text-[length:var(--density-type-control)] text-fg-faint">No model selected.</div>
      )}
    </Drawer>
  )
}

function ModelDrawerContent({
  model,
  peers,
  onClose,
  titleId,
}: {
  model: DrawerModel
  peers: Peer[]
  onClose: () => void
  titleId: string
}) {
  if (isConfigModel(model)) {
    return (
      <div>
        <DrawerHeader
          badges={
            <>
              <DrawerBadge tone="muted">{model.quant}</DrawerBadge>
              <DrawerBadge tone={model.vision ? 'accent' : 'muted'}>{model.vision ? 'vision' : 'text'}</DrawerBadge>
              <DrawerBadge tone={model.moe ? 'accent' : 'muted'}>{model.moe ? 'moe' : 'dense'}</DrawerBadge>
            </>
          }
          onClose={onClose}
          subtitle={modelSubtitle(model)}
          title={model.name}
          titleId={titleId}
        />

        <div className="pb-[20px] pt-[12px]">
          <SectionHead icon={drawerIcon(Cpu)}>Model metadata</SectionHead>
          <div className="grid grid-cols-2 gap-[8px] px-[18px]">
            <KV icon={drawerIcon(Cpu)} label="Params">{model.paramsB}B</KV>
            <KV icon={drawerIcon(HardDrive)} label="Size">{model.sizeGB} GB</KV>
            <KV icon={drawerIcon(HardDrive)} label="Disk">{model.diskGB} GB</KV>
            <KV icon={drawerIcon(Cpu)} label="Context">{model.ctxMaxK}k</KV>
          </div>

          <SectionHead icon={drawerIcon(Network)}>Capabilities</SectionHead>
          <div className="flex flex-wrap gap-[6px] px-[18px]">
            {model.tags.map((tag) => (
              <DrawerBadge key={tag} tone="accent">{tag}</DrawerBadge>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const activePeers = peersForModel(model, peers)
  const availability = model.nodeCount ?? (activePeers.length || 1)
  const status = modelStatusBadge(model.status)

  return (
    <div>
      <DrawerHeader
        badges={
          <>
            <DrawerBadge dot tone={status.tone}>{status.label}</DrawerBadge>
            <DrawerBadge tone="good">Fits</DrawerBadge>
          </>
        }
        onClose={onClose}
        subtitle={modelSubtitle(model)}
        title={model.name}
        titleId={titleId}
      />

      <div className="pb-[24px] pt-[12px]">
        <h3 className="sr-only">Model metadata</h3>
        <span className="sr-only">{model.context}</span>
        <div className="flex gap-[8px] px-[18px]">
          <KV icon={drawerIcon(Network)} label="Availability">{availability} node{availability === 1 ? '' : 's'}</KV>
          <KV icon={drawerIcon(HardDrive)} label="Mesh VRAM">{modelSummarySize(model)}</KV>
          <KV icon={drawerIcon(Cpu)} label="Context">{modelSummaryContext(model)}</KV>
          <KV icon={drawerIcon(Cpu)} label="Quant">{modelQuant(model)}</KV>
        </div>

        <SectionHead icon={drawerIcon(Network)}>Capabilities</SectionHead>
        <div className="flex flex-wrap gap-[6px] px-[18px]">
          {model.tags.map((tag) => (
            <DrawerBadge key={tag} tone="accent">{tag}</DrawerBadge>
          ))}
        </div>

        <SectionHead icon={drawerIcon(Hash)}>Files</SectionHead>
        <div className="flex flex-col gap-[8px] px-[18px]">
          <KV icon={drawerIcon(Hash)} label="Shorthand">{model.name}</KV>
          <KV icon={drawerIcon(Hash)} label="Full name">{model.fullId ?? model.name}.gguf</KV>
        </div>

        <SectionHead icon={drawerIcon(Network)}>Active peers</SectionHead>
        <div className="mx-[18px] overflow-hidden rounded-[var(--radius)] border border-border-soft bg-background">
          <div className="grid grid-cols-[1.5fr_0.7fr_0.7fr_0.6fr] bg-panel-strong px-[12px] py-[8px] text-[length:var(--density-type-label)] font-medium uppercase tracking-[0.5px] text-fg-faint">
            <div>Node</div>
            <div>Latency</div>
            <div>VRAM</div>
            <div>Share</div>
          </div>

          {activePeers.length ? (
            activePeers.map((peer) => (
              <div
                className="grid grid-cols-[1.5fr_0.7fr_0.7fr_0.6fr] items-center border-t border-border-soft px-[12px] py-[10px] font-mono text-[length:var(--density-type-caption-lg)]"
                key={peer.id}
              >
                <span>{peer.shortId ?? peer.hostname}</span>
                <span className="text-fg-faint">{formatLatency(peer.latencyMs)} ms</span>
                <span>{modelSummarySize(model)}</span>
                <DrawerBadge tone="accent">100%</DrawerBadge>
              </div>
            ))
          ) : (
            <div className="border-t border-border-soft px-[12px] py-[12px] text-[length:var(--density-type-caption-lg)] text-fg-faint">
              No active peers for this model.
            </div>
          )}
        </div>

        {model.activitySummary ? <div className="px-[18px] pt-[8px] text-[length:var(--density-type-caption)] leading-[16px] text-fg-faint">{model.activitySummary}</div> : null}
      </div>
    </div>
  )
}
