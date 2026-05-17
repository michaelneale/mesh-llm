import { useState, type CSSProperties, type ReactNode } from 'react'
import { cn } from '@/lib/cn'
import { Copy } from 'lucide-react'
import { buildTOML } from '@/features/configuration/lib/build-toml'
import { HighlightedTomlLines } from '@/features/configuration/components/toml-highlight'
import type {
  ConfigAssign,
  ConfigModel,
  ConfigNode,
  ConfigurationDefaultsHarnessData,
  ConfigurationDefaultsValues,
  TomlValidationWarning
} from '@/features/app-tabs/types'
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
  sourceStyle?: TomlSourceStyle
  copyLabel: string
  onCopy: () => void
  reviewMode: boolean
  configPath?: string
}

type TomlSourceStyle = CSSProperties & { '--toml-line-count'?: number }

function TomlEditorPanel({
  toml,
  lineCount,
  highlighted,
  scrollOffset,
  setScrollOffset,
  sourceStyle,
  copyLabel,
  onCopy,
  reviewMode,
  configPath
}: TomlPanelProps) {
  return (
    <section
      className={cn('toml-panel panel-shell overflow-hidden rounded-lg border border-border bg-panel', {
        'flex h-full min-h-0 flex-col': reviewMode
      })}
    >
      <header className="toml-panel-header panel-divider flex items-center justify-between border-b border-border-soft">
        <div className="toml-panel-heading flex min-w-0 items-baseline">
          <h2 className="type-panel-title shrink-0">{reviewMode ? 'Generated TOML' : 'Configuration TOML'}</h2>
          {reviewMode && configPath ? (
            <span className="toml-config-path mono truncate font-normal text-fg-faint">{configPath}</span>
          ) : null}
        </div>
        <div className="toml-panel-actions flex shrink-0 items-center">
          <span className="type-caption tabular-nums text-fg-faint">{lineCount} lines</span>
          {reviewMode ? (
            <span className="toml-review-badge rounded-full border border-border-soft text-fg-faint">
              edits this node only
            </span>
          ) : null}
          <button
            className="toml-copy-button ui-control inline-flex shrink-0 items-center justify-center rounded border"
            onClick={onCopy}
            type="button"
            aria-label={copyLabel}
            title={copyLabel}
          >
            <Copy className="toml-copy-icon" strokeWidth={1.75} />
          </button>
        </div>
      </header>
      <div
        className={cn('toml-source relative overflow-hidden bg-background font-mono', {
          'min-h-0 flex-1': reviewMode,
          'toml-source-standalone': !reviewMode
        })}
        style={sourceStyle}
      >
        <pre
          aria-hidden="true"
          className="toml-source-content pointer-events-none absolute inset-0 m-0 whitespace-pre text-fg-dim"
          style={{ transform: `translate(${-scrollOffset.left}px, ${-scrollOffset.top}px)` }}
        >
          {highlighted}
        </pre>
        <textarea
          aria-label="Configuration TOML source"
          className="toml-source-content toml-source-input absolute inset-0 block h-full w-full resize-none overflow-auto bg-transparent font-mono text-transparent caret-transparent"
          onScroll={(event) =>
            setScrollOffset({ left: event.currentTarget.scrollLeft, top: event.currentTarget.scrollTop })
          }
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
    <section
      className={cn('toml-panel panel-shell overflow-hidden rounded-lg border border-border bg-panel', className)}
    >
      <header className="toml-panel-header border-b border-border-soft">
        <h3 className="type-panel-title">{title}</h3>
      </header>
      {children}
    </section>
  )
}

function warningDotClass(kind: TomlValidationWarning['kind']): string {
  if (kind === 'ok') return 'toml-status-dot toml-status-dot-ok shrink-0 rounded-full'
  if (kind === 'warn') return 'toml-status-dot toml-status-dot-warn shrink-0 rounded-full'
  return 'toml-status-dot shrink-0 rounded-full bg-fg-faint'
}

function WarningItem({ kind, text }: TomlValidationWarning) {
  return (
    <div className="toml-warning-item flex items-start border-t border-border-soft first:border-t-0">
      <span aria-hidden="true" className={warningDotClass(kind)} />
      <span className="type-caption text-fg-dim">{text}</span>
    </div>
  )
}

const DEFAULT_VALIDATION_WARNINGS: TomlValidationWarning[] = [
  { kind: 'ok', text: 'All local model entries use canonical nested model_fit and hardware sections.' },
  {
    kind: 'warn',
    text: 'carrack · GPU 0 · GLM-4.7-Flash will exceed 80% VRAM at 16K context. Consider 8K or moving to GPU 1.'
  },
  { kind: 'ok', text: 'Remote nodes remain read-only context and are excluded from the saved TOML preview.' },
  { kind: 'info', text: 'Request defaults merge at request time, explicit request payload fields still win.' }
]

function ValidationPanel({ warnings, className }: { warnings?: TomlValidationWarning[]; className?: string }) {
  const resolvedWarnings = warnings ?? DEFAULT_VALIDATION_WARNINGS

  return (
    <ReviewPanel title="Validation" className={className}>
      <div className="toml-warning-list">
        {resolvedWarnings.map((warning) => (
          <WarningItem key={`${warning.kind}-${warning.text}`} kind={warning.kind} text={warning.text} />
        ))}
      </div>
    </ReviewPanel>
  )
}

function LaunchSummaryPanel({
  nodes,
  assigns,
  defaultsValues,
  launchSummaryConfig,
  className
}: {
  nodes: ConfigNode[]
  assigns: ConfigAssign[]
  defaultsValues?: ConfigurationDefaultsValues
  launchSummaryConfig?: { httpBind?: string; mmap?: string }
  className?: string
}) {
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
    ['mmap:', mmap]
  ]

  return (
    <ReviewPanel title="Effective launch summary" className={className}>
      <div className="toml-summary-list type-caption text-fg-dim">
        {rows.map(([label, value]) => (
          <div className="toml-summary-row" key={label}>
            <span className="text-fg-faint">{label}</span>{' '}
            <span className={cn('font-mono', label === 'flash attn:' ? 'toml-summary-value-good' : 'text-foreground')}>
              {value}
            </span>
          </div>
        ))}
      </div>
    </ReviewPanel>
  )
}

export function TomlView({
  nodes,
  assigns,
  models,
  defaults,
  defaultsValues,
  reviewMode = false,
  configPath,
  validationWarnings,
  launchSummaryConfig
}: TomlViewProps) {
  const toml = buildTOML(nodes, assigns, models, { defaults, defaultsValues })
  const { copyState, copyText } = useClipboardCopy()
  const [scrollOffset, setScrollOffset] = useState({ left: 0, top: 0 })
  const copyLabel = copyStateLabel(copyState, 'TOML')
  const lines = toml.split('\n')
  const lineCount = lines.length
  const sourceStyle: TomlSourceStyle | undefined = reviewMode ? undefined : { '--toml-line-count': lineCount }
  const highlighted = <HighlightedTomlLines toml={toml} />

  const editor = (
    <TomlEditorPanel
      toml={toml}
      lineCount={lineCount}
      highlighted={highlighted}
      scrollOffset={scrollOffset}
      setScrollOffset={setScrollOffset}
      sourceStyle={sourceStyle}
      copyLabel={copyLabel}
      onCopy={() => {
        void copyText(toml)
      }}
      reviewMode={reviewMode}
      configPath={configPath}
    />
  )

  if (!reviewMode) return editor

  return (
    <div className="toml-review-layout grid xl:items-stretch">
      {editor}
      <aside className="toml-review-aside flex flex-col" aria-label="TOML review actions">
        <ValidationPanel warnings={validationWarnings} className="flex-1 min-h-0" />
        <LaunchSummaryPanel
          nodes={nodes}
          assigns={assigns}
          defaultsValues={defaultsValues}
          launchSummaryConfig={launchSummaryConfig}
          className="flex-1 min-h-0"
        />
      </aside>
    </div>
  )
}
