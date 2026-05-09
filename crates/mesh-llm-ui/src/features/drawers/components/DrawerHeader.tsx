import { X } from 'lucide-react'
import type { ReactNode } from 'react'

type DrawerHeaderProps = {
  title: string
  titleId?: string
  subtitle?: string
  badges?: ReactNode
  onClose: () => void
}

export function DrawerHeader({ title, titleId, subtitle, badges, onClose }: DrawerHeaderProps) {
  return (
    <header className="sticky top-0 z-10 border-b border-border-soft bg-panel px-[18px] py-[18px]">
      <div className="flex items-start gap-[12px]">
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-1">
            <h2 className="type-headline min-w-0 truncate font-mono" id={titleId}>
              {title}
            </h2>
            {badges ? <div className="flex shrink-0 flex-wrap items-center gap-1.5">{badges}</div> : null}
          </div>
          {subtitle ? <p className="type-caption mt-[3px] truncate font-mono text-fg-faint">{subtitle}</p> : null}
        </div>

        <button
          aria-label="Close drawer"
          className="ui-control inline-flex size-[28px] shrink-0 items-center justify-center rounded-[var(--radius)] border"
          onClick={onClose}
          type="button"
        >
          <X aria-hidden="true" className="size-3.5" strokeWidth={1.9} />
        </button>
      </div>
    </header>
  )
}
