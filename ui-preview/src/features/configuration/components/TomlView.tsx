import { useState, type ReactNode } from 'react'
import { cn } from '@/lib/cn'
import { Copy } from 'lucide-react'
import { buildTOML } from '@/features/configuration/lib/build-toml'
import { HighlightedTomlLines } from '@/features/configuration/components/toml-highlight'
import type { ConfigAssign, ConfigModel, ConfigNode, ConfigurationDefaultsHarnessData, ConfigurationDefaultsValues, TomlValidationWarning } from '@/features/app-tabs/types'
import { copyStateLabel } from '@/lib/copyStateLabel'
import { useClipboardCopy } from '@/lib/useClipboardCopy'

type TomlViewProps = {
  nodes: ConfigNode[]
  assigns: ConfigAssign[]
  models?: ConfigModel[]
  defaults?: ConfigurationDefaultsHarnessData
  defaultsValues?: ConfigurationDefaultsValues
  reviewMode?: boolean
  configPath?: string
  validationWarnings?: TomlValidationWarning[]
  launchSummaryConfig?: { httpBind?: string; mmap?: string }
}

type TomlPanelProps = {
  toml: string
  lineCount: number
  highlighted: ReactNode
  scrollOffset: { left: number; top: number }
  setScrollOffset: (offset: { left: number; top: number }) => void
  sourceHeight?: string
  copyLabel: string
  onCopy: () => void
  reviewMode: boolean
  configPath?: string
}

function TomlEditorPanel({ toml, lineCount, highlighted, scrollOffset, setScrollOffset, sourceHeight, copyLabel, onCopy, reviewMode, configPath }: TomlPanelProps) {
  return (
    <section className={["panel-shell overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel", reviewMode ? 'flex h-full min-h-0 flex-col' : ''].filter(Boolean).join(' ')}>
      <header className="panel-divider flex min-h-[46px] items-center justify-between border-b border-border-soft px-[14px] py-[10px]">
        <div className="flex min-w-0 items-baseline gap-1">
          <h2 className="type-panel-title shrink-0">{reviewMode ? 'Generated TOML' : 'Configuration TOML'}</h2>
          {reviewMode && configPath ? <span className="truncate font-mono text-[11px] font-normal text-fg-faint">{configPath}</span> : null}
        </div>
        <div className="flex shrink-0 items-center gap-2.5">
          <span className="text-[11px] text-fg-faint">{lineCount} lines</span>
          {reviewMode ? <span className="rounded-full border border-border-soft px-2 py-0.5 text-[10.5px] text-fg-faint">edits this node only</span> : null}
          <button
            className="ui-control inline-flex size-7 shrink-0 items-center justify-center rounded-[var(--radius)] border p-1.5"
            onClick={onCopy}
            type="button"
            aria-label={copyLabel}
            title={copyLabel}
          >
            <Copy className="size-3.5" strokeWidth={1.75} />
          </button>
        </div>
      </header>
      <div
        className={["relative overflow-hidden bg-background font-mono text-[length:var(--density-type-caption-lg)] leading-[1.6]", reviewMode ? 'min-h-0 flex-1' : ''].filter(Boolean).join(' ')}
        style={{ height: sourceHeight }}
      >
        <pre
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 m-0 whitespace-pre px-[14px] py-3 text-fg-dim"
          style={{ transform: `translate(${-scrollOffset.left}px, ${-scrollOffset.top}px)` }}
        >
          {highlighted}
        </pre>
        <textarea
          aria-label="Configuration TOML source"
          className="absolute inset-0 block h-full w-full resize-none overflow-auto bg-transparent px-[14px] py-3 font-mono leading-[1.6] text-transparent caret-transparent outline-none focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent"
          onScroll={(event) => setScrollOffset({ left: event.currentTarget.scrollLeft, top: event.currentTarget.scrollTop })}
          readOnly
          value={toml}
          wrap="off"
        />
      </div>
    </section>
  )
}

function ReviewPanel({ title, children, className }: { title: string; children: ReactNode; className?: string }) {
  return (
    <section className={cn('panel-shell overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel', className)}>
      <header className="border-b border-border-soft px-[14px] py-[10px]">
        <h3 className="type-panel-title">{title}</h3>
      </header>
      {children}
    </section>
  )
}

function warningDotClass(kind: TomlValidationWarning['kind']): string {
  if (kind === 'ok') return 'mt-1.5 size-[7px] shrink-0 rounded-full bg-[var(--color-good)] shadow-[0_0_7px_var(--color-good)]'
  if (kind === 'warn') return 'mt-1.5 size-[7px] shrink-0 rounded-full bg-[var(--color-warn)]'
  return 'mt-1.5 size-[7px] shrink-0 rounded-full bg-fg-faint'
}

function WarningItem({ kind, text }: TomlValidationWarning) {
  return (
    <div className="flex items-start gap-2 border-t border-border-soft px-[14px] py-2 first:border-t-0">
      <span aria-hidden="true" className={warningDotClass(kind)} />
      <span className="text-[11.5px] leading-[1.55] text-fg-dim">{text}</span>
    </div>
  )
}

const DEFAULT_VALIDATION_WARNINGS: TomlValidationWarning[] = [
  { kind: 'ok', text: 'All pinned models have valid gpu_id targets.' },
  { kind: 'warn', text: 'carrack · GPU 0 · GLM-4.7-Flash will exceed 80% VRAM at 16K context. Consider 8K or moving to GPU 1.' },
  { kind: 'ok', text: 'Plugin endpoint http://localhost:8000/v1 is reachable.' },
  { kind: 'info', text: 'Flash attention is on by default, no per-model override emitted.' },
]

function ValidationPanel({ warnings, className }: { warnings?: TomlValidationWarning[]; className?: string }) {
  const resolvedWarnings = warnings ?? DEFAULT_VALIDATION_WARNINGS

  return (
    <ReviewPanel title="Validation" className={className}>
      <div className="py-2">
        {resolvedWarnings.map((warning) => (
          <WarningItem key={`${warning.kind}-${warning.text}`} kind={warning.kind} text={warning.text} />
        ))}
      </div>
    </ReviewPanel>
  )
}

function LaunchSummaryPanel({ nodes, assigns, defaultsValues, launchSummaryConfig, className }: { nodes: ConfigNode[]; assigns: ConfigAssign[]; defaultsValues?: ConfigurationDefaultsValues; launchSummaryConfig?: { httpBind?: string; mmap?: string }; className?: string }) {
  const localNode = nodes[0]
  const gpuCount = localNode?.gpus.length ?? 0
  const flashAttention = defaultsValues?.['flash-attention'] ?? 'on'
  const kvCache = defaultsValues?.['kv-cache'] ?? 'auto'
  const httpBind = launchSummaryConfig?.httpBind ?? '0.0.0.0:9337'
  const mmap = launchSummaryConfig?.mmap ?? 'off'
  const rows = [
    ['node:', localNode?.hostname ?? 'local'],
    ['placements:', `${assigns.length} models on ${gpuCount} GPUs`],
    ['http:', httpBind],
    ['flash attn:', flashAttention],
    ['kv cache:', `${kvCache} (q8_0/q4_0 above 5GB)`],
    ['mmap:', mmap],
  ]

  return (
    <ReviewPanel title="Effective launch summary" className={className}>
      <div className="px-[14px] py-[10px] text-[11.5px] leading-[1.7] text-fg-dim">
        {rows.map(([label, value]) => (
          <div key={label}>
            <span className="text-fg-faint">{label}</span> <span className={label === 'flash attn:' ? 'font-mono text-[var(--color-good)]' : 'font-mono text-foreground'}>{value}</span>
          </div>
        ))}
      </div>
    </ReviewPanel>
  )
}

export function TomlView({ nodes, assigns, models, defaults, defaultsValues, reviewMode = false, configPath, validationWarnings, launchSummaryConfig }: TomlViewProps) {
  const toml = buildTOML(nodes, assigns, models, { defaults, defaultsValues })
  const { copyState, copyText } = useClipboardCopy()
  const [scrollOffset, setScrollOffset] = useState({ left: 0, top: 0 })
  const copyLabel = copyStateLabel(copyState, 'TOML')
  const lines = toml.split('\n')
  const lineCount = lines.length
  const sourceHeight = reviewMode ? undefined : `min(1040px, max(440px, calc(${lineCount} * 1.6em + 1.5rem)))`
  const highlighted = <HighlightedTomlLines toml={toml} />

  const editor = (
    <TomlEditorPanel
      toml={toml}
      lineCount={lineCount}
      highlighted={highlighted}
      scrollOffset={scrollOffset}
      setScrollOffset={setScrollOffset}
      sourceHeight={sourceHeight}
      copyLabel={copyLabel}
      onCopy={() => { void copyText(toml) }}
      reviewMode={reviewMode}
      configPath={configPath}
    />
  )

  if (!reviewMode) return editor

  return (
    <div className="grid min-h-[700px] gap-[14px] xl:grid-cols-[minmax(0,1fr)_360px] xl:items-stretch">
      {editor}
      <aside className="flex flex-col gap-3" aria-label="TOML review actions">
        <ValidationPanel warnings={validationWarnings} className="flex-1 min-h-0" />
        <LaunchSummaryPanel nodes={nodes} assigns={assigns} defaultsValues={defaultsValues} launchSummaryConfig={launchSummaryConfig} className="flex-1 min-h-0" />
      </aside>
    </div>
  )
}
