import { Network, RotateCcw, Save, Undo2, Redo2, UserRound } from 'lucide-react'
import type { ReactNode } from 'react'
import type { ConfigNode } from '@/features/app-tabs/types'
import { cn } from '@/lib/cn'

type ConfigurationHeaderProps = {
  title: string
  description: ReactNode
  nodes: ConfigNode[]
  canUndo: boolean
  canRedo: boolean
  hasUnsavedChanges: boolean
  hasInvalidNode: boolean
  onUndo: () => void
  onRedo: () => void
  onRevert: () => void
  onSave: () => void
}

function formatLocalHostname(node: ConfigNode | undefined) {
  if (!node) return 'local node'

  return node.hostname.includes('.') ? node.hostname : `${node.hostname}.local`
}

function InlineMeta({ children, className, icon }: { children: ReactNode; className?: string; icon: ReactNode }) {
  return (
    <span className={cn('inline-flex items-center gap-1.5 whitespace-nowrap text-[12px] leading-none text-fg-dim', className)}>
      <span className="inline-flex size-3 items-center justify-center text-fg-faint">{icon}</span>
      {children}
    </span>
  )
}

const iconActionClass = 'ui-control inline-flex size-[30px] items-center justify-center rounded-[var(--radius)] border text-[length:var(--density-type-caption-lg)] font-semibold leading-none'
const textActionClass = 'ui-control inline-flex h-[30px] items-center gap-1.5 rounded-[var(--radius)] border px-4 text-[length:var(--density-type-control)] font-medium leading-none'

export function ConfigurationHistoryActions({
  canUndo,
  canRedo,
  hasUnsavedChanges,
  hasInvalidNode,
  onUndo,
  onRedo,
  onRevert,
  onSave,
}: Omit<ConfigurationHeaderProps, 'title' | 'description' | 'nodes'>) {
  return (
    <fieldset aria-label="Configuration history actions" className="mt-[11px] flex items-center gap-1.5 border-0 p-0">
      <button className={iconActionClass} disabled={!canUndo} onClick={onUndo} type="button" aria-label="Undo" aria-keyshortcuts="Control+Z"><Undo2 aria-hidden="true" className="size-3.5" /></button>
      <button className={iconActionClass} disabled={!canRedo} onClick={onRedo} type="button" aria-label="Redo" aria-keyshortcuts="Control+R"><Redo2 aria-hidden="true" className="size-3.5" /></button>
      <button
        className={textActionClass}
        onClick={onRevert}
        type="button"
        aria-keyshortcuts="Control+X"
      >
        <RotateCcw aria-hidden="true" className="size-3.5" />
        Revert
      </button>
      <button
        aria-invalid={hasInvalidNode}
        className={cn(
          'inline-flex h-[30px] items-center gap-1.5 rounded-[var(--radius)] border px-4 text-[length:var(--density-type-control)] font-semibold leading-none',
          hasInvalidNode ? 'ui-control-destructive' : 'ui-control-primary',
        )}
        disabled={hasInvalidNode || !hasUnsavedChanges}
        onClick={onSave}
        title={hasInvalidNode ? 'Resolve invalid model allocations before saving' : hasUnsavedChanges ? 'Save config' : 'No changes to save'}
        type="button"
        aria-keyshortcuts="Control+S"
      >
        <Save aria-hidden="true" className="size-3.5" />
        Save config
      </button>
    </fieldset>
  )
}

export function ConfigurationHeader(props: ConfigurationHeaderProps) {
  const localNode = props.nodes[0]
  const onlineNodes = props.nodes.length
  const localHostname = formatLocalHostname(localNode)

  return (
    <header className="relative z-20 bg-transparent">
      <div className="flex min-h-[76px] flex-wrap items-center justify-between gap-3 px-5 py-3">
        <div className="min-w-0">
          <h1 className="text-[20px] font-bold leading-none tracking-[-0.4px] text-foreground">{props.title}</h1>
          <div className="mt-2 flex flex-wrap items-center gap-x-2.5 gap-y-1.5 text-[12px] text-fg-dim">
            <InlineMeta icon={<UserRound aria-hidden="true" className="size-[11px]" strokeWidth={1.6} />}>
              Editing <strong className="font-mono text-[11.5px] font-semibold text-foreground">{localHostname}</strong>
              <span className="hidden text-fg-faint md:inline">· local node only</span>
            </InlineMeta>
            <InlineMeta className="hidden md:inline-flex" icon={<Network aria-hidden="true" className="size-[11px]" strokeWidth={1.6} />}>
              {onlineNodes} nodes connected · remote read-only
            </InlineMeta>
          </div>
        </div>
        <ConfigurationHistoryActions {...props} />
      </div>
    </header>
  )
}
