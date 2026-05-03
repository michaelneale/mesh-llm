import type { CSSProperties } from 'react'
import {
  BrainCircuit,
  ChevronDown,
  Cloud,
  Cpu,
  Eye,
  Network,
  PanelsTopLeft,
  SlidersHorizontal,
  Sparkles,
  Wind,
  type LucideIcon
} from 'lucide-react'
import { AccentIconFrame } from '@/components/ui/AccentIconFrame'
import { cn } from '@/lib/cn'
import type { ModelFamilyColorKey, ModelSummary } from '@/features/app-tabs/types'

type ModelCatalogProps = {
  models: ModelSummary[]
  filterLabel: string
  onFilterClick?: () => void
  onSelect?: (model: ModelSummary) => void
  selectedModelName?: string
}

type ModelProvider = { label: string; Icon: LucideIcon; prefixes: string[] }
type ModelIconStyle = CSSProperties & { '--model-card-icon-color': string }

const MODEL_FAMILY_COLOR_FALLBACK: ModelFamilyColorKey = 'family-0'
const MODEL_FAMILY_COLOR_KEYS: readonly ModelFamilyColorKey[] = [
  'family-0',
  'family-1',
  'family-2',
  'family-3',
  'family-4',
  'family-5',
  'family-6',
  'family-7'
]

const MODEL_PROVIDERS: ModelProvider[] = [
  { label: 'Google', Icon: Sparkles, prefixes: ['gemma'] },
  { label: 'Alibaba', Icon: Cloud, prefixes: ['qwen'] },
  { label: 'Zhipu AI', Icon: BrainCircuit, prefixes: ['glm'] },
  { label: 'Meta', Icon: Network, prefixes: ['llama'] },
  { label: 'Mistral AI', Icon: Wind, prefixes: ['mixtral'] },
  { label: 'Microsoft', Icon: PanelsTopLeft, prefixes: ['phi'] },
  { label: 'Community', Icon: Eye, prefixes: ['llava'] }
]

const UNKNOWN_PROVIDER: ModelProvider = { label: 'Unknown', Icon: Cpu, prefixes: [] }

function modelMatchesProvider(candidate: string, prefix: string) {
  return candidate === prefix || candidate.startsWith(prefix) || candidate.includes(`/${prefix}`)
}

function getModelProvider(model: ModelSummary) {
  const candidates = [model.family, model.name].map((value) => value.trim().toLowerCase()).filter(Boolean)

  return (
    MODEL_PROVIDERS.find((provider) =>
      provider.prefixes.some((prefix) => candidates.some((candidate) => modelMatchesProvider(candidate, prefix)))
    ) ?? UNKNOWN_PROVIDER
  )
}

function modelFamilyColorKey(model: Pick<ModelSummary, 'family' | 'familyColor'>): ModelFamilyColorKey {
  if (model.familyColor) return model.familyColor

  const family = model.family.trim().toLowerCase()
  if (!family) return MODEL_FAMILY_COLOR_FALLBACK

  let hash = 0
  for (const character of family) hash = (hash * 31 + character.charCodeAt(0)) % MODEL_FAMILY_COLOR_KEYS.length
  return MODEL_FAMILY_COLOR_KEYS[hash] ?? MODEL_FAMILY_COLOR_FALLBACK
}

function modelIconStyle(colorKey: ModelFamilyColorKey): ModelIconStyle {
  return {
    '--model-card-icon-color': `var(--model-family-color-${colorKey}, var(--model-family-color-fallback))`,
    background: 'color-mix(in oklab, var(--model-card-icon-color) 34%, var(--color-panel-strong))',
    border: '1px solid color-mix(in oklab, var(--model-card-icon-color) 42%, var(--color-border))',
    color: 'color-mix(in oklab, var(--model-card-icon-color) 58%, var(--color-fg-dim))'
  }
}

function StatusBadge({ status }: { status: ModelSummary['status'] }) {
  const label = status === 'ready' ? 'Ready' : status === 'warm' ? 'Warm' : status === 'warming' ? 'Warming' : 'Offline'
  const pillBg =
    status === 'ready' || status === 'warm'
      ? 'color-mix(in oklab, var(--color-good) 18%, var(--color-background))'
      : 'color-mix(in oklab, var(--color-muted) 18%, var(--color-background))'
  const pillFg = status === 'ready' || status === 'warm' ? 'var(--color-good)' : 'var(--color-muted-foreground)'
  const pillBorder =
    status === 'ready' || status === 'warm'
      ? 'color-mix(in oklab, var(--color-good) 30%, var(--color-background))'
      : 'var(--color-border)'
  return (
    <span
      className="inline-flex items-center gap-[5px] rounded-full px-2 py-px text-[length:var(--density-type-caption)] font-medium"
      style={{ background: pillBg, color: pillFg, border: `1px solid ${pillBorder}` }}
    >
      <span className="size-[5px] rounded-full" style={{ background: 'currentColor' }} />
      {label}
    </span>
  )
}

function ModelRow({
  model,
  active,
  onSelect
}: {
  model: ModelSummary
  active: boolean
  onSelect?: (model: ModelSummary) => void
}) {
  const sizeLabel = model.sizeGB === undefined ? model.size : `${model.sizeGB} GB`
  const contextLabel = model.ctxMaxK === undefined ? model.context : `${model.ctxMaxK}k ctx`
  const architectureLabel = model.moe ? 'MoE' : 'Dense'
  const provider = getModelProvider(model)
  const ProviderIcon = provider.Icon
  const familyColorKey = modelFamilyColorKey(model)

  return (
    <button
      aria-label={`View ${model.name} model from ${provider.label}${active ? ' (selected)' : ''}`}
      data-active={active ? 'true' : undefined}
      className={cn(
        'ui-row-action grid w-full gap-x-2.5 border-b border-border-soft px-3 py-2.5 text-left',
        active ? 'bg-[color-mix(in_oklab,var(--color-accent)_10%,var(--color-panel))]' : 'bg-transparent'
      )}
      style={{ gridTemplateColumns: 'auto 1fr auto' }}
      onClick={() => onSelect?.(model)}
      type="button"
    >
      <AccentIconFrame className="size-8 self-start" style={modelIconStyle(familyColorKey)} tone="subtle">
        <ProviderIcon className="size-4" aria-hidden="true" strokeWidth={1.8} />
      </AccentIconFrame>
      <div className="min-w-0">
        <div className="truncate font-mono text-[length:var(--density-type-control-lg)] font-medium">{model.name}</div>
        {model.fullId && (
          <div className="mt-0.5 truncate font-mono text-[length:var(--density-type-label)] text-fg-faint">
            {model.fullId}
          </div>
        )}
        <div className="mt-1.5 flex items-center gap-1.5">
          <span className="text-[length:var(--density-type-label)] font-medium text-fg-dim">{provider.label}</span>
          <span className="text-[length:var(--density-type-label)] text-fg-faint">·</span>
          <span className="font-mono text-[length:var(--density-type-label)] text-fg-faint">
            {model.nodeCount ?? 1} node
          </span>
          <span className="text-[length:var(--density-type-label)] text-fg-faint">·</span>
          <span className="font-mono text-[length:var(--density-type-label)] text-fg-dim">{sizeLabel}</span>
          <span className="text-[length:var(--density-type-label)] text-fg-faint">·</span>
          <span className="font-mono text-[length:var(--density-type-label)] text-fg-dim">{contextLabel}</span>
        </div>
      </div>
      <div className="flex flex-col items-end gap-[5px]">
        <StatusBadge status={model.status} />
        <span className="inline-flex items-center rounded-full border border-border px-[7px] py-px text-[length:var(--density-type-label)] font-medium text-fg-faint">
          {architectureLabel}
        </span>
      </div>
    </button>
  )
}

export function ModelCatalog({ models, filterLabel, onFilterClick, onSelect, selectedModelName }: ModelCatalogProps) {
  return (
    <aside className="panel-shell overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
      <header className="flex items-center justify-between border-b border-border-soft px-3.5 py-2.5">
        <h2 className="type-panel-title">Model catalog</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={onFilterClick}
            type="button"
            className="ui-control-ghost inline-flex items-center gap-1.5 rounded-[var(--radius)] px-1.5 py-1 text-[length:var(--density-type-caption)]"
          >
            <SlidersHorizontal className="size-[11px]" /> Filter
          </button>
          <div className="inline-flex items-center gap-1 rounded-[var(--radius)] border border-border bg-panel-strong px-1.5 py-px text-[length:var(--density-type-caption-lg)]">
            {filterLabel} <ChevronDown className="size-[11px]" />
          </div>
        </div>
      </header>
      <div className="overflow-y-auto" style={{ maxHeight: 492 }}>
        {models.map((model) => (
          <ModelRow key={model.name} active={model.name === selectedModelName} model={model} onSelect={onSelect} />
        ))}
      </div>
    </aside>
  )
}
