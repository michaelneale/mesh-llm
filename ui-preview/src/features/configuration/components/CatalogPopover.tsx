import { Search } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { flushSync } from 'react-dom'
import { SegmentedControl } from '@/components/ui/SegmentedControl'
import { CommandBarModal } from '@/features/app-shell/components/command-bar/CommandBarModal'
import { CommandBarProvider } from '@/features/app-shell/components/command-bar/CommandBarProvider'
import { Badge } from '@/features/app-shell/components/command-bar/command-bar-primitives'
import type { CommandBarMode, CommandBarResultContainerProps } from '@/features/app-shell/components/command-bar/command-bar-types'
import { useCommandBar } from '@/features/app-shell/components/command-bar/useCommandBar'
import { CFG_CATALOG } from '@/features/app-tabs/data'
import { ModelCatalogRow } from '@/features/configuration/components/ModelCatalogRow'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'

type CatalogPopoverProps = {
  open: boolean
  onClose: () => void
  selectedNode: ConfigNode
  assigns: ConfigAssign[]
  models?: ConfigModel[]
  errorMessage?: string | null
  onSelectModel?: (model: ConfigModel) => boolean | void
}

const filters = ['All', 'MoE', 'Vision', '<8', '8-32', '>32'] as const
const filterOptions = filters.map((filter) => ({ value: filter, label: filter }))
const CATALOG_MODE_ID = 'catalog'

function isCatalogFilter(value: string): value is (typeof filters)[number] {
  return filters.some((filter) => filter === value)
}

function matchesCatalogFilter(model: ConfigModel, filter: (typeof filters)[number]) {
  return filter === 'All'
    || (filter === 'MoE' && model.moe)
    || (filter === 'Vision' && model.vision)
    || (filter === '<8' && model.paramsB < 8)
    || (filter === '8-32' && model.paramsB >= 8 && model.paramsB <= 32)
    || (filter === '>32' && model.paramsB > 32)
}

function CatalogResultContainer<T>({ children }: CommandBarResultContainerProps<T>) {
  return <div className="space-y-2 px-1 py-1">{children}</div>
}

function CatalogPopoverDialog({ onClose, selectedNode, assigns, models = CFG_CATALOG, errorMessage, onSelectModel }: Omit<CatalogPopoverProps, 'open'>) {
  const { openCommandBar } = useCommandBar()
  const [draggingModel, setDraggingModel] = useState(false)
  const [filter, setFilter] = useState<(typeof filters)[number]>('All')

  useEffect(() => {
    openCommandBar(CATALOG_MODE_ID)
  }, [openCommandBar])

  const filteredModels = useMemo(() => models.filter((model) => matchesCatalogFilter(model, filter)), [filter, models])

  const closeAfterDragStarts = useCallback(() => {
    flushSync(() => setDraggingModel(true))
    window.setTimeout(onClose, 0)
  }, [onClose])

  const modes = useMemo<readonly CommandBarMode<ConfigModel>[]>(() => [{
    id: CATALOG_MODE_ID,
    label: 'Catalog',
    leadingIcon: Search,
    source: filteredModels,
    getItemKey: (model) => model.id,
    getSearchText: (model) => model.name,
    getKeywords: (model) => [
      model.family,
      model.quant,
      model.paramsLabel ?? `${model.paramsB}B`,
      `${model.paramsB}`,
      `${model.paramsB}b`,
      ...model.tags,
    ],
    ResultContainer: CatalogResultContainer,
    ResultItem: ({ item, selected }) => (
      <ModelCatalogRow
        model={item}
        node={selectedNode}
        assigns={assigns}
        models={models}
        selected={selected}
        onDragEnd={() => {
          setDraggingModel(false)
          onClose()
        }}
        onDragStart={closeAfterDragStarts}
      />
    ),
    onSelect: (model) => onSelectModel?.(model),
    optionClassName: 'mx-0 px-0 py-0 bg-transparent',
  }], [assigns, closeAfterDragStarts, filteredModels, models, onClose, onSelectModel, selectedNode])

  const interstitial = (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge>
          {filteredModels.length} / {models.length}
        </Badge>
        <span className="text-[length:var(--density-type-caption)] text-muted-foreground">Drag a model onto a VRAM bar</span>
      </div>
      {errorMessage ? (
        <div className="rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-warn)_36%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-warn)_10%,var(--color-panel))] px-3 py-2 text-[length:var(--density-type-caption)] leading-relaxed text-warn" role="alert">
          {errorMessage}
        </div>
      ) : null}
      <SegmentedControl
        ariaLabel="Catalog filters"
        className="flex flex-wrap gap-1"
        itemClassName="h-auto rounded-full px-2.5 py-[3px] text-[length:var(--density-type-caption)]"
        onValueChange={(nextFilter) => {
          if (isCatalogFilter(nextFilter)) setFilter(nextFilter)
        }}
        orientation="horizontal"
        options={filterOptions}
        value={filter}
      />
    </div>
  )

  return (
    <CommandBarModal
      modes={modes}
      behavior="distinct"
      defaultModeId={CATALOG_MODE_ID}
      title="Model catalog"
      description="Search locally scanned GGUF models and drag them onto a VRAM bar."
      placeholder="Search locally-scanned GGUF models…"
      emptyMessage="No models match."
      interstitial={interstitial}
      onClose={onClose}
      overlayClassName={draggingModel ? 'pointer-events-none opacity-0' : undefined}
      contentClassName={draggingModel ? 'pointer-events-none opacity-0' : undefined}
    />
  )
}

export function CatalogPopover({ open, onClose, selectedNode, assigns, models = CFG_CATALOG, errorMessage, onSelectModel }: CatalogPopoverProps) {
  if (!open) return null

  return (
    <CommandBarProvider>
      <CatalogPopoverDialog onClose={onClose} selectedNode={selectedNode} assigns={assigns} models={models} errorMessage={errorMessage} onSelectModel={onSelectModel} />
    </CommandBarProvider>
  )
}
