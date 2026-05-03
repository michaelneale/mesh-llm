import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

export type SidebarNavigationItem<TId extends string = string> = {
  id: TId
  label: ReactNode
  summary?: ReactNode
  count?: ReactNode
  icon?: ReactNode
  disabled?: boolean
}

export type SidebarNavigationSection<TId extends string = string> = {
  id: string
  title?: ReactNode
  items: readonly SidebarNavigationItem<TId>[]
}

export type SidebarNavigationProps<TId extends string = string> = {
  ariaLabel: string
  items?: readonly SidebarNavigationItem<TId>[]
  sections?: readonly SidebarNavigationSection<TId>[]
  activeId: TId
  onSelect: (id: TId) => void
  eyebrow?: ReactNode
  footer?: ReactNode
  className?: string
  navClassName?: string
  itemClassName?: string
  sectionTitleClassName?: string
  sectionItemsClassName?: string
}

export function SidebarNavigation<TId extends string = string>({
  ariaLabel,
  items,
  sections,
  activeId,
  onSelect,
  eyebrow,
  footer,
  className,
  navClassName,
  itemClassName,
  sectionTitleClassName,
  sectionItemsClassName,
}: SidebarNavigationProps<TId>) {
  const navigationSections = sections ?? [{ id: 'items', items: items ?? [] }]

  return (
    <aside className={cn('min-w-0', className)}>
      {eyebrow ? <p className="px-2 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.07em] text-fg-faint">{eyebrow}</p> : null}
      <nav aria-label={ariaLabel} className={cn('space-y-0.5', navClassName)}>
        {navigationSections.map((section, sectionIndex) => (
          <div key={section.id}>
            {section.title ? (
              <p className={cn('px-0.5 pb-1 text-[length:var(--density-type-micro)] font-semibold uppercase tracking-[0.05em] text-fg-faint', sectionIndex === 0 ? 'pt-1' : 'pt-3', sectionTitleClassName)}>
                {section.title}
              </p>
            ) : null}
            <div className={cn('space-y-0.5', sectionItemsClassName)}>
              {section.items.map((item) => {
                const active = item.id === activeId
                const hasSummary = item.summary !== undefined

                return (
                  <button
                    aria-current={active ? 'true' : undefined}
                    className={cn(
                      'ui-row-action relative w-full select-none rounded-[var(--radius)] border border-transparent px-2.5 py-[7px] text-left outline-none transition-[background,color] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50',
                      active
                        ? 'text-foreground before:absolute before:-left-3 before:bottom-[7px] before:top-[7px] before:w-[2.5px] before:rounded-[2px] before:bg-accent before:content-[""]'
                        : 'text-fg-dim',
                      itemClassName,
                    )}
                    data-active={active ? 'true' : undefined}
                    disabled={item.disabled}
                    key={item.id}
                    onClick={() => onSelect(item.id)}
                    style={active ? { background: 'color-mix(in oklab, var(--color-accent) 16%, transparent)' } : undefined}
                    type="button"
                  >
                    <span className={cn('flex gap-2.5', hasSummary ? 'items-start' : 'items-center')}>
                      {item.icon ? <span className="mt-0.5 grid shrink-0 place-items-center text-current opacity-85">{item.icon}</span> : null}
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[length:var(--density-type-control-lg)] leading-none" style={{ fontWeight: active ? 500 : 400 }}>{item.label}</span>
                        {item.summary ? <span className="mt-1 block line-clamp-2 text-[length:var(--density-type-caption)] leading-snug text-fg-faint">{item.summary}</span> : null}
                      </span>
                      {item.count !== undefined ? <span className="shrink-0 font-mono text-[length:var(--density-type-annotation)] text-fg-faint">{item.count}</span> : null}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </nav>
      {footer ? <div className="mt-3 border-t border-border-soft px-2 pb-1 pt-3 text-[length:var(--density-type-caption)] leading-relaxed text-fg-faint">{footer}</div> : null}
    </aside>
  )
}
