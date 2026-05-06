import type { Dispatch, ReactNode, SetStateAction } from 'react'
import { ChevronDown } from 'lucide-react'
import { configurationNavigationIconClassName } from '@/features/configuration/components/configuration-navigation-class-names'
import { containerUsedGB, nodeUsableGB } from '@/features/configuration/lib/config-math'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'

type NodeRailProps = {
  nodes: ConfigNode[]
  assigns: ConfigAssign[]
  models?: ConfigModel[]
  collapsedMap: Record<string, boolean>
  setCollapsedMap: Dispatch<SetStateAction<Record<string, boolean>>>
  onJump: (nodeId: string) => void
  keyboardHint?: ReactNode
}
export function NodeRail({
  nodes,
  assigns,
  models,
  collapsedMap: _collapsedMap,
  setCollapsedMap,
  onJump,
  keyboardHint
}: NodeRailProps) {
  void _collapsedMap
  return (
    <nav
      className="panel-shell static self-start rounded-[var(--radius-lg)] border border-border bg-panel p-2.5 lg:sticky lg:top-[70px]"
      aria-label="Configuration nodes"
    >
      <div className="mb-2 px-0.5 text-[length:var(--density-type-annotation)] font-semibold uppercase tracking-[0.06em] text-fg-faint">
        Nodes · {nodes.length}
      </div>
      {nodes.map((node) => {
        const usable = nodeUsableGB(node)
        const used = node.gpus.reduce((sum, gpu) => sum + containerUsedGB(assigns, node.id, gpu.idx, models), 0)
        const pct = usable > 0 ? Math.min(100, (used / usable) * 100) : 0
        const deviceLabel = `${node.gpus.length} ${node.gpus.length === 1 ? 'device' : 'devices'}`
        return (
          <div key={node.id} className="mb-1.5">
            <div className="flex items-center gap-1.5 rounded-[var(--radius)] px-2 py-1.5">
              <button
                className="ui-control-ghost flex min-w-0 flex-1 items-center gap-1.5 rounded-[var(--radius)] px-1 py-0.5 text-left"
                onClick={() => onJump(node.id)}
                tabIndex={-1}
                type="button"
              >
                <span
                  className="size-1.5 shrink-0 rounded-full"
                  style={{
                    background: node.status === 'online' ? 'var(--color-good)' : 'var(--color-fg-faint)',
                    boxShadow: node.status === 'online' ? 'var(--shadow-status-good)' : 'none'
                  }}
                />
                <span className="flex-1 truncate font-mono text-[length:var(--density-type-caption-lg)] font-medium">
                  {node.hostname}
                </span>
              </button>
              <button
                className="ui-control-ghost rounded-[var(--radius)] p-0.5"
                onClick={() => setCollapsedMap((map) => ({ ...map, [node.id]: !map[node.id] }))}
                tabIndex={-1}
                type="button"
                aria-label={`Toggle ${node.hostname}`}
              >
                <ChevronDown className={configurationNavigationIconClassName} />
              </button>
            </div>
            <div className="mx-2 h-[3px] overflow-hidden rounded-sm bg-panel-strong">
              <div className="h-full bg-accent" style={{ width: `${pct}%`, opacity: 0.65 }} />
            </div>
            <div className="px-2 py-0.5 font-mono text-[length:var(--density-type-annotation)] text-fg-faint">
              {deviceLabel}
            </div>
          </div>
        )
      })}
      {keyboardHint ? (
        <div className="panel-divider mt-2 border-t border-border-soft px-0.5 pt-2 text-[length:var(--density-type-caption)] text-fg-faint">
          {keyboardHint}
        </div>
      ) : null}
    </nav>
  )
}
