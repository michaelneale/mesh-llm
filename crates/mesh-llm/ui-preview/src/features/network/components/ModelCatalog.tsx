import type { CSSProperties } from 'react'
import { useMemo, useState } from 'react'
import { BrainCircuit, Cloud, Cpu, Eye, Network, PanelsTopLeft, Sparkles, Wind, type LucideIcon } from 'lucide-react'
import { AccentIconFrame } from '@/components/ui/AccentIconFrame'
import { FilterPopover } from '@/components/ui/FilterPopover'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { cn } from '@/lib/cn'
import { modelStatusBadge } from '@/features/drawers/lib/model-status'
import { modelFamilyColorKey } from '@/features/configuration/lib/config-math'
import type { ModelFamilyColorKey, ModelSummary } from '@/features/app-tabs/types'
import {
  buildModelFilterOptions,
  isModelColumnFiltered,
  modelFilterColumns,
  modelFilterOptionLabel,
  modelFilterValues,
  modelProviderLabel,
  selectedModelFilterSet,
  type FilterOption,
  type ModelFilterKey,
  type ModelFilterState
} from '@/features/network/lib/model-catalog-utils'

type ModelCatalogProps = {
  models: ModelSummary[]
  onSelect?: (model: ModelSummary) => void
  selectedModelName?: string
}

type ModelProvider = { label: string; Icon: LucideIcon }
type ModelIconStyle = CSSProperties & { '--model-card-icon-color': string }

const MODEL_PROVIDERS: ModelProvider[] = [
  { label: 'Cohere', Icon: Sparkles },
  { label: 'Z.ai', Icon: BrainCircuit },
  { label: 'OpenAI', Icon: Sparkles },
  { label: 'StepFun', Icon: Sparkles },
  { label: 'Google', Icon: Sparkles },
  { label: 'Alibaba', Icon: Cloud },
  { label: 'Nvidia', Icon: Cpu },
  { label: 'Ant Group', Icon: Network },
  { label: 'MiniMax', Icon: Sparkles },
  { label: 'Meta', Icon: Network },
  { label: 'Mistral AI', Icon: Wind },
  { label: 'Microsoft', Icon: PanelsTopLeft },
  { label: 'Community', Icon: Eye }
]

const UNKNOWN_PROVIDER: ModelProvider = { label: 'Unknown', Icon: Cpu }

function getModelProvider(model: ModelSummary) {
  return MODEL_PROVIDERS.find((provider) => provider.label === modelProviderLabel(model)) ?? UNKNOWN_PROVIDER
}

function modelIconStyle(colorKey: ModelFamilyColorKey): ModelIconStyle {
  return {
    '--model-card-icon-color': `var(--model-family-color-${colorKey}, var(--model-family-color-fallback))`,
    background: 'color-mix(in oklab, var(--model-card-icon-color) 34%, var(--color-panel-strong))',
    border: '1px solid color-mix(in oklab, var(--model-card-icon-color) 42%, var(--color-border))',
    color: 'color-mix(in oklab, var(--model-card-icon-color) 58%, var(--color-fg-dim))'
  }
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
  const status = modelStatusBadge(model.status)

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
        <StatusBadge dot tone={status.tone}>
          {status.label}
        </StatusBadge>
        <span className="inline-flex items-center rounded-full border border-border px-[7px] py-px text-[length:var(--density-type-label)] font-medium text-fg-faint">
          {architectureLabel}
        </span>
      </div>
    </button>
  )
}

type ModelFilterPopoverProps = {
  optionsByColumn: Record<ModelFilterKey, { value: string; count: number }[]>
  selectedValues: Record<ModelFilterKey, Set<string>>
  activeFilterGroups: number
  visibleCount: number
  totalCount: number
  onValueChange: (key: ModelFilterKey, value: string, checked: boolean) => void
  onSelectAll: (key: ModelFilterKey) => void
  onSelectNone: (key: ModelFilterKey) => void
  onClear: () => void
}

function ModelFilterPopover({
  optionsByColumn,
  selectedValues,
  activeFilterGroups,
  visibleCount,
  totalCount,
  onValueChange,
  onSelectAll,
  onSelectNone,
  onClear
}: ModelFilterPopoverProps) {
  return (
    <FilterPopover
      activeFilterGroups={activeFilterGroups}
      categories={modelFilterColumns}
      contentLabel="Filter model catalog"
      formatOptionLabel={modelFilterOptionLabel}
      id="model-catalog-filter"
      itemLabel="models"
      optionsByCategory={optionsByColumn}
      selectedValuesByCategory={selectedValues}
      title="Filter models"
      totalCount={totalCount}
      triggerLabel="Filter models"
      visibleCount={visibleCount}
      onClear={onClear}
      onSelectAll={onSelectAll}
      onSelectNone={onSelectNone}
      onValueChange={onValueChange}
    />
  )
}

export function ModelCatalog({ models, onSelect, selectedModelName }: ModelCatalogProps) {
  const [filters, setFilters] = useState<ModelFilterState>({})

  const filterOptionsByColumn = useMemo<Record<ModelFilterKey, FilterOption[]>>(
    () =>
      Object.fromEntries(
        modelFilterColumns.map((column) => [column.key, buildModelFilterOptions(models, column.key)])
      ) as Record<ModelFilterKey, FilterOption[]>,
    [models]
  )

  const selectedFilterValues = useMemo<Record<ModelFilterKey, Set<string>>>(
    () =>
      Object.fromEntries(
        modelFilterColumns.map((column) => [
          column.key,
          selectedModelFilterSet(filters, column.key, filterOptionsByColumn[column.key])
        ])
      ) as Record<ModelFilterKey, Set<string>>,
    [filterOptionsByColumn, filters]
  )

  const activeFilterGroups = modelFilterColumns.filter((column) =>
    isModelColumnFiltered(filters, column.key, filterOptionsByColumn[column.key])
  ).length

  const filteredModels = useMemo(
    () =>
      models.filter((model) =>
        modelFilterColumns.every((column) =>
          modelFilterValues(model, column.key).some((value) => selectedFilterValues[column.key].has(value))
        )
      ),
    [models, selectedFilterValues]
  )

  function handleFilterValueChange(key: ModelFilterKey, value: string, checked: boolean) {
    const options = filterOptionsByColumn[key]

    setFilters((current) => {
      const currentSelected = selectedModelFilterSet(current, key, options)

      if (checked) {
        currentSelected.add(value)
      } else {
        currentSelected.delete(value)
      }

      const optionValues = options.map((option) => option.value)
      const nextSelected = optionValues.filter((optionValue) => currentSelected.has(optionValue))
      const nextFilters = { ...current }

      if (nextSelected.length === optionValues.length) {
        delete nextFilters[key]
      } else {
        nextFilters[key] = nextSelected
      }

      return nextFilters
    })
  }

  function handleSelectAll(key: ModelFilterKey) {
    setFilters((current) => {
      const nextFilters = { ...current }
      delete nextFilters[key]
      return nextFilters
    })
  }

  function handleSelectNone(key: ModelFilterKey) {
    setFilters((current) => ({ ...current, [key]: [] }))
  }

  function handleClearFilters() {
    setFilters({})
  }

  return (
    <aside className="panel-shell flex h-full min-h-0 flex-col overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
      <header className="flex shrink-0 items-center justify-between border-b border-border-soft px-3.5 py-2.5">
        <h2 className="type-panel-title">Model catalog</h2>
        <div className="flex items-center gap-2">
          <ModelFilterPopover
            activeFilterGroups={activeFilterGroups}
            optionsByColumn={filterOptionsByColumn}
            selectedValues={selectedFilterValues}
            totalCount={models.length}
            visibleCount={filteredModels.length}
            onClear={handleClearFilters}
            onSelectAll={handleSelectAll}
            onSelectNone={handleSelectNone}
            onValueChange={handleFilterValueChange}
          />
        </div>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {filteredModels.length > 0 ? (
          filteredModels.map((model) => (
            <ModelRow key={model.name} active={model.name === selectedModelName} model={model} onSelect={onSelect} />
          ))
        ) : (
          <div className="px-3.5 py-8 text-center">
            <p className="text-[length:var(--density-type-control)] font-semibold text-fg">
              {activeFilterGroups > 0 ? 'No models match these filters.' : 'No models available.'}
            </p>
            {activeFilterGroups > 0 ? (
              <button
                type="button"
                className="mt-2 rounded-[var(--radius)] px-2 py-1 text-[length:var(--density-type-caption)] text-accent transition-colors hover:bg-panel-strong focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                onClick={handleClearFilters}
              >
                Clear filters
              </button>
            ) : null}
          </div>
        )}
      </div>
    </aside>
  )
}
