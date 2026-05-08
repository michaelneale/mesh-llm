import { Cpu, HardDrive } from 'lucide-react'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import { DrawerHeader } from '@/features/drawers/components/DrawerHeader'
import { KV } from '@/features/drawers/components/KV'
import { SectionHead } from '@/features/drawers/components/SectionHead'
import { drawerIcon } from '@/features/drawers/lib/badge-styles'
import { nodeTotalGB } from '@/features/configuration/lib/config-math'
import type { ConfigNode } from '@/features/app-tabs/types'

type ConfigNodeDrawerPanelProps = {
  node: ConfigNode
  onClose: () => void
  titleId: string
}

function configNodeStatusTone(status: ConfigNode['status']): StatusBadgeTone {
  if (status === 'online') return 'good'
  if (status === 'degraded') return 'warn'
  return 'bad'
}

export function ConfigNodeDrawerPanel({ node, onClose, titleId }: ConfigNodeDrawerPanelProps) {
  return (
    <div>
      <DrawerHeader
        badges={
          <>
            <StatusBadge tone={configNodeStatusTone(node.status)}>{node.status}</StatusBadge>
            <StatusBadge tone="muted">{node.placement}</StatusBadge>
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
          <KV icon={drawerIcon(Cpu)} label="CPU">
            {node.cpu}
          </KV>
          <KV icon={drawerIcon(HardDrive)} label="RAM">
            {node.ramGB} GB
          </KV>
          <KV icon={drawerIcon(HardDrive)} label="VRAM">
            {nodeTotalGB(node)} GB
          </KV>
          <KV icon={drawerIcon(Cpu)} label="GPUs">
            {node.gpus.length}
          </KV>
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
              <div className="mt-1 font-mono text-[length:var(--density-type-label)] text-fg-faint">
                {gpu.totalGB} GB total
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
