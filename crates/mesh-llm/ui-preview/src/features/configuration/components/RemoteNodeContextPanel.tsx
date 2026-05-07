import { LockKeyhole } from 'lucide-react'
import { findModel, nodeUsableGB } from '@/features/configuration/lib/config-math'
import { formatGB, nodeGpuCountLabel } from '@/features/configuration/lib/config-display'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'

type RemoteNodeContextPanelProps = {
  node: ConfigNode
  assigns: ConfigAssign[]
  models: ConfigModel[]
}

export function RemoteNodeContextPanel({ node, assigns, models }: RemoteNodeContextPanelProps) {
  const nodeAssigns = assigns.filter((assign) => assign.nodeId === node.id)

  return (
    <section
      id={`node-${node.id}`}
      aria-labelledby={`remote-node-${node.id}`}
      className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-3.5 shadow-surface-panel"
      data-panel-soft-elevation="none"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="type-label text-fg-faint">Remote context</p>
          <h2
            id={`remote-node-${node.id}`}
            className="mt-1 text-[length:var(--density-type-control-lg)] font-semibold text-foreground"
          >
            {node.hostname}
          </h2>
          <p className="mt-1 text-[length:var(--density-type-control)] leading-relaxed text-fg-dim">
            {node.region} · {nodeGpuCountLabel(node)} · {formatGB(nodeUsableGB(node), { fixedFractionDigits: 1 })} GB
            usable
          </p>
        </div>
        <span className="inline-flex items-center gap-1.5 rounded-[var(--radius)] border border-border-soft bg-surface px-2 py-1 font-mono text-[length:var(--density-type-annotation)] uppercase tracking-[0.06em] text-fg-faint">
          <LockKeyhole aria-hidden="true" className="size-3" />
          read-only
        </span>
      </div>
      <div className="mt-3 rounded-[var(--radius)] border border-border-soft bg-surface p-2.5">
        <p className="text-[length:var(--density-type-control)] text-fg-dim">
          Remote placements are visible for capacity planning, but this page only writes the local node config.
        </p>
        {nodeAssigns.length > 0 ? (
          <ul className="mt-2 space-y-1.5">
            {nodeAssigns.map((assign) => {
              const model = findModel(assign.modelId, models)

              return (
                <li
                  className="flex flex-wrap items-center justify-between gap-2 font-mono text-[length:var(--density-type-caption)] text-fg-dim"
                  key={assign.id}
                >
                  <span>{model?.name ?? assign.modelId}</span>
                  <span>{assign.ctx.toLocaleString()} ctx</span>
                </li>
              )
            })}
          </ul>
        ) : null}
      </div>
    </section>
  )
}
