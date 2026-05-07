import { useState, type ReactNode } from 'react'
import { AccentIconFrame } from '@/components/ui/AccentIconFrame'
import { InfoBanner } from '@/components/ui/InfoBanner'
import { SidebarNavigation } from '@/components/ui/SidebarNavigation'
import { HighlightedTomlLines } from '@/features/configuration/components/toml-highlight'
import { cn } from '@/lib/cn'

export type SettingsCategoryItem = {
  id: string
  label: string
  summary: string
  count: number
  icon?: ReactNode
}

type SettingsSummaryBannerProps = {
  eyebrow?: string
  title: string
  description: ReactNode
  status: string
  action?: ReactNode
}

type SettingsCategoryRailProps = {
  categories: readonly SettingsCategoryItem[]
  activeId: string
  footer: ReactNode
  onSelect: (id: string) => void
}

type SettingsSectionProps = {
  id: string
  icon: ReactNode
  title: string
  subtitle: string
  children: ReactNode
}

type SettingsRowProps = {
  className?: string
  label: string
  hint: string
  children: ReactNode
}

type SettingsPreviewRailProps = {
  title: string
  code: string
  tip: ReactNode
}

export function SettingsSummaryBanner({ eyebrow, title, description, status, action }: SettingsSummaryBannerProps) {
  return (
    <InfoBanner
      action={action}
      descriptionClassName="max-w-none whitespace-normal"
      description={
        <>
          {eyebrow ? <span className="type-label mb-1 block text-fg-faint">{eyebrow}</span> : null}
          <span className="text-[length:var(--density-type-control)] leading-relaxed">{description}</span>
        </>
      }
      status={
        <span className="inline-flex rounded-full border border-border-soft bg-transparent px-2 py-0.5 font-mono text-[length:var(--density-type-annotation)] leading-none text-fg-dim">
          {status}
        </span>
      }
      title={title}
      titleId="defaults-summary-heading"
    />
  )
}

export function SettingsCategoryRail({ categories, activeId, footer, onSelect }: SettingsCategoryRailProps) {
  return (
    <SidebarNavigation
      activeId={activeId}
      ariaLabel="Defaults sections"
      className="static self-start lg:sticky lg:top-[76px]"
      eyebrow="Categories"
      footer={footer}
      items={categories.map((category) => ({
        id: category.id,
        label: category.label,
        count: category.count,
        icon: category.icon
      }))}
      onSelect={onSelect}
    />
  )
}

export function SettingsSection({ id, icon, title, subtitle, children }: SettingsSectionProps) {
  return (
    <section
      id={id}
      aria-labelledby={`${id}-heading`}
      className="panel-shell scroll-mt-20 rounded-[var(--radius-lg)] border border-border bg-panel px-[18px] pb-[18px] pt-4 shadow-surface-panel"
      data-panel-soft-elevation="none"
    >
      <header className="mb-1 flex items-start gap-2.5">
        <AccentIconFrame className="size-9">{icon}</AccentIconFrame>
        <div>
          <h3
            id={`${id}-heading`}
            className="text-[length:var(--density-type-control-lg)] font-semibold leading-tight text-foreground"
          >
            {title}
          </h3>
          <p className="mt-1 text-[length:var(--density-type-caption)] leading-snug text-fg-faint">{subtitle}</p>
        </div>
      </header>
      <div>{children}</div>
    </section>
  )
}

export function SettingsRow({ className, label, hint, children }: SettingsRowProps) {
  return (
    <div
      className={cn(
        'grid gap-3 border-t border-border-soft py-3 md:grid-cols-[minmax(0,1fr)_auto] md:items-center',
        className
      )}
      data-settings-row="true"
    >
      <div className="min-w-0">
        <p className="text-[length:var(--density-type-control)] font-medium leading-tight text-foreground">{label}</p>
        <p className="mt-1 text-[length:var(--density-type-caption)] leading-relaxed text-fg-faint">{hint}</p>
      </div>
      <div className="min-w-0 md:justify-self-end">{children}</div>
    </div>
  )
}

export function SettingsPreviewRail({ title, code, tip }: SettingsPreviewRailProps) {
  const [scrollOffset, setScrollOffset] = useState({ left: 0, top: 0 })

  return (
    <aside aria-label={title} className="sticky top-[76px] space-y-2.5">
      <section
        className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-3 shadow-surface-panel"
        data-panel-soft-elevation="none"
      >
        <h3 className="flex items-center gap-2 text-[length:var(--density-type-control)] font-semibold text-foreground">
          <span>Preview</span>
          <span className="font-mono text-fg-dim">{title}</span>
        </h3>
        <div className="relative mt-2.5 max-h-[360px] min-h-[220px] overflow-hidden rounded-[var(--radius)] border border-border-soft bg-background font-mono text-[length:var(--density-type-caption-lg)] leading-relaxed">
          <pre
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 m-0 whitespace-pre p-3 text-fg-dim"
            style={{ transform: `translate(${-scrollOffset.left}px, ${-scrollOffset.top}px)` }}
          >
            <HighlightedTomlLines toml={code} />
          </pre>
          <textarea
            aria-label={`${title} preview code`}
            className="absolute inset-0 block h-full w-full resize-none overflow-auto bg-transparent p-3 font-mono leading-relaxed text-transparent caret-transparent outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            onScroll={(event) =>
              setScrollOffset({ left: event.currentTarget.scrollLeft, top: event.currentTarget.scrollTop })
            }
            readOnly
            value={code}
            wrap="off"
          />
        </div>
      </section>
      <section className="panel-shell rounded-[var(--radius-lg)] border border-dashed border-border bg-panel p-3 text-[length:var(--density-type-caption)] leading-relaxed text-fg-dim">
        <div className="mb-1.5 text-[9.5px] font-semibold uppercase tracking-[0.06em] text-fg-faint">TIP</div>
        {tip}
      </section>
    </aside>
  )
}
