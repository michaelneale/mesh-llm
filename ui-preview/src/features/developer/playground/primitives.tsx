import { useState, type ReactNode } from 'react'
import { SidebarNavigation } from '@/components/ui/SidebarNavigation'
import type { TabPanelItem } from '@/components/ui/TabPanel'

export function PlaygroundPanel({ title, description, actions, children }: { title: string; description?: ReactNode; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
      <header className="flex flex-wrap items-start justify-between gap-3 border-b border-border-soft px-3.5 py-2.5">
        <div>
          <h2 className="type-panel-title text-foreground">{title}</h2>
          {description ? <div className="mt-1 text-[length:var(--density-type-caption-lg)] text-fg-dim">{description}</div> : null}
        </div>
        {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
      </header>
      <div className="p-3.5">{children}</div>
    </section>
  )
}

export function OptionGroup<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (value: T) => void
}) {
  return (
    <div className="space-y-1.5">
      <div className="type-label text-fg-faint">{label}</div>
      <div className="flex flex-wrap gap-1.5">
        {options.map((option) => (
          <button
            key={option.value}
            aria-pressed={value === option.value}
            className="ui-control inline-flex items-center rounded-[var(--radius)] border px-2.5 py-1 text-[length:var(--density-type-caption)] font-medium"
            data-active={value === option.value ? 'true' : undefined}
            onClick={() => onChange(option.value)}
            type="button"
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export function TextField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="block space-y-1.5">
      <span className="type-label text-fg-faint">{label}</span>
      <input
        aria-label={label}
        className="w-full rounded-[var(--radius)] border border-border bg-background px-3 py-2 text-[length:var(--density-type-body)] text-foreground outline-none transition-[border-color,box-shadow] focus:border-accent focus:shadow-[var(--shadow-focus-accent)]"
        onChange={(event) => onChange(event.target.value)}
        type="text"
        value={value}
      />
    </label>
  )
}

export function TextAreaField({ label, value, onChange, rows = 3 }: { label: string; value: string; onChange: (value: string) => void; rows?: number }) {
  return (
    <label className="block space-y-1.5">
      <span className="type-label text-fg-faint">{label}</span>
      <textarea
        aria-label={label}
        className="block w-full resize-y rounded-[var(--radius)] border border-border bg-background px-3 py-2 text-[length:var(--density-type-body)] text-foreground outline-none transition-[border-color,box-shadow] focus:border-accent focus:shadow-[var(--shadow-focus-accent)]"
        onChange={(event) => onChange(event.target.value)}
        rows={rows}
        value={value}
      />
    </label>
  )
}

export function ToggleChip({ label, pressed, onToggle }: { label: string; pressed: boolean; onToggle: () => void }) {
  return (
    <button
      aria-pressed={pressed}
      className="ui-control inline-flex items-center rounded-full border px-2.5 py-1 text-[length:var(--density-type-caption)] font-medium"
      data-active={pressed ? 'true' : undefined}
      onClick={onToggle}
      type="button"
    >
      {label}
    </button>
  )
}

export function SidebarTabs<TValue extends string>({
  ariaLabel,
  tabs,
  defaultValue,
}: {
  ariaLabel: string
  tabs: TabPanelItem<TValue>[]
  defaultValue: TValue
}) {
  const [activeValue, setActiveValue] = useState<TValue>(defaultValue)
  const activeTab = tabs.find((tab) => tab.value === activeValue && !tab.disabled) ?? tabs.find((tab) => !tab.disabled) ?? tabs[0]

  if (!activeTab) return null

  return (
    <div className="grid gap-4 lg:grid-cols-[210px_minmax(0,1fr)]">
      <SidebarNavigation
        activeId={activeTab.value}
        ariaLabel={ariaLabel}
        className="lg:sticky lg:top-[76px]"
        eyebrow="Previews"
        items={tabs.map((tab) => {
          const Icon = tab.icon

          return {
            id: tab.value,
            label: tab.label,
            summary: tab.description,
            icon: Icon ? <Icon aria-hidden={true} className="size-3.5" strokeWidth={1.7} /> : undefined,
            count: typeof tab.accessory === 'function' ? undefined : tab.accessory,
            disabled: tab.disabled,
          }
        })}
        onSelect={setActiveValue}
      />
      <section aria-label={ariaLabel} className="min-w-0 space-y-4">
        {activeTab.content}
      </section>
    </div>
  )
}
