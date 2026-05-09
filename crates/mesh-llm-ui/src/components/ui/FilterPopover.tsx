import * as Popover from '@radix-ui/react-popover'
import { Check, Funnel } from 'lucide-react'
import { cn } from '@/lib/cn'

export type FilterCategory<Key extends string> = {
  key: Key
  label: string
}

export type FilterValueOption = {
  value: string
  count: number
}

export type FilterPopoverProps<Key extends string> = {
  id: string
  title: string
  triggerLabel: string
  contentLabel: string
  itemLabel: string
  categories: Array<FilterCategory<Key>>
  optionsByCategory: Record<Key, FilterValueOption[]>
  selectedValuesByCategory: Record<Key, Set<string>>
  activeFilterGroups: number
  visibleCount: number
  totalCount: number
  formatOptionLabel: (value: string) => string
  onValueChange: (key: Key, value: string, checked: boolean) => void
  onSelectAll: (key: Key) => void
  onSelectNone: (key: Key) => void
  onClear: () => void
}

export function FilterPopover<Key extends string>({
  id,
  title,
  triggerLabel,
  contentLabel,
  itemLabel,
  categories,
  optionsByCategory,
  selectedValuesByCategory,
  activeFilterGroups,
  visibleCount,
  totalCount,
  formatOptionLabel,
  onValueChange,
  onSelectAll,
  onSelectNone,
  onClear
}: FilterPopoverProps<Key>) {
  const filtersActive = activeFilterGroups > 0
  const triggerAriaLabel = filtersActive ? `${triggerLabel}, ${activeFilterGroups} active` : triggerLabel

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button
          type="button"
          aria-controls={id}
          aria-haspopup="dialog"
          aria-label={triggerAriaLabel}
          className={cn(
            'ui-control inline-flex h-7 shrink-0 items-center gap-1.5 rounded-[var(--radius)] px-2 text-[length:var(--density-type-caption)] outline-none transition-[border-color,background,color,box-shadow]',
            'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
            filtersActive &&
              'border-accent/45 bg-[color-mix(in_oklab,var(--color-accent)_12%,var(--color-panel))] text-fg shadow-surface-selected'
          )}
        >
          <Funnel aria-hidden={true} className="size-3.5" strokeWidth={1.8} />
          <span className="hidden sm:inline">Filter</span>
          {filtersActive ? (
            <span className="rounded-full bg-accent px-1.5 py-px font-mono text-[10px] leading-none text-accent-ink">
              {activeFilterGroups}
            </span>
          ) : null}
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          id={id}
          align="end"
          aria-label={contentLabel}
          className="surface-popover-panel z-50 w-[min(22rem,calc(100vw-2rem))] rounded-[var(--radius-lg)] border border-border bg-panel p-3 text-[length:var(--density-type-caption)] text-fg shadow-surface-popover outline-none"
          collisionPadding={12}
          side="bottom"
          sideOffset={8}
        >
          <div className="flex items-start justify-between gap-3 border-b border-border-soft pb-2.5">
            <div className="min-w-0">
              <div className="type-panel-title text-[length:var(--density-type-control)]">{title}</div>
              <p className="mt-0.5 text-[length:var(--density-type-caption)] text-fg-faint">
                Showing <span className="font-mono text-fg-dim">{visibleCount}</span> of{' '}
                <span className="font-mono text-fg-dim">{totalCount}</span>
              </p>
            </div>
            <button
              type="button"
              className="rounded-[var(--radius-sm)] px-2 py-1 text-[length:var(--density-type-caption)] text-fg-dim transition-colors hover:bg-panel-strong hover:text-fg disabled:pointer-events-none disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              disabled={!filtersActive}
              onClick={onClear}
            >
              Reset
            </button>
          </div>
          <div className="mt-2.5 grid max-h-[min(28rem,70vh)] gap-3 overflow-y-auto pr-1">
            {categories.map((category) => {
              const options = optionsByCategory[category.key]
              const selected = selectedValuesByCategory[category.key]
              const selectedCount = options.filter((option) => selected.has(option.value)).length
              const categoryActive = selectedCount < options.length
              const noneActive = selectedCount === 0

              return (
                <section key={category.key} className="min-w-0">
                  <div className="mb-1.5 flex items-center justify-between gap-2">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <div className="type-label text-fg-faint">{category.label}</div>
                      <span className="font-mono text-[10px] text-fg-faint">
                        {selectedCount}/{options.length}
                      </span>
                    </div>
                    <div className="flex items-center gap-0.5">
                      <button
                        type="button"
                        className="rounded-[var(--radius-sm)] px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.07em] text-fg-dim transition-colors hover:bg-panel-strong hover:text-fg disabled:pointer-events-none disabled:opacity-35 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                        disabled={!categoryActive}
                        onClick={() => onSelectAll(category.key)}
                      >
                        All
                      </button>
                      <button
                        type="button"
                        className="rounded-[var(--radius-sm)] px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.07em] text-fg-dim transition-colors hover:bg-panel-strong hover:text-fg disabled:pointer-events-none disabled:opacity-35 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                        disabled={noneActive || options.length === 0}
                        onClick={() => onSelectNone(category.key)}
                      >
                        None
                      </button>
                    </div>
                  </div>
                  <div className="grid gap-1">
                    {options.map((option) => {
                      const checked = selected.has(option.value)
                      const label = formatOptionLabel(option.value)

                      return (
                        <label
                          key={option.value}
                          className="flex min-w-0 cursor-pointer items-center gap-2 rounded-[var(--radius)] px-2 py-1.5 transition-colors hover:bg-panel-strong focus-within:bg-panel-strong"
                        >
                          <input
                            type="checkbox"
                            className="sr-only"
                            aria-label={`${label}, ${option.count} ${itemLabel}`}
                            checked={checked}
                            onChange={(event) => onValueChange(category.key, option.value, event.currentTarget.checked)}
                          />
                          <span
                            aria-hidden={true}
                            className={cn(
                              'grid size-4 shrink-0 place-items-center rounded-[4px] border transition-colors',
                              checked
                                ? 'border-accent bg-accent text-accent-ink'
                                : 'border-border bg-panel-strong text-transparent'
                            )}
                          >
                            <Check className="size-3" strokeWidth={2.2} />
                          </span>
                          <span className="min-w-0 flex-1 truncate font-mono text-fg-dim">{label}</span>
                          <span className="shrink-0 font-mono text-[10px] text-fg-faint">{option.count}</span>
                        </label>
                      )
                    })}
                  </div>
                </section>
              )
            })}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  )
}
