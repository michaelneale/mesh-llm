import { useState, type KeyboardEvent, type ReactNode } from 'react'
import * as HoverCard from '@radix-ui/react-hover-card'

export type HeaderHoverCardTriggerProps = {
  'aria-expanded': boolean
  'aria-haspopup': 'dialog'
  onClick: () => void
  onKeyDown: (event: KeyboardEvent<HTMLElement>) => void
}

type HeaderHoverCardProps = {
  trigger: (props: HeaderHoverCardTriggerProps) => ReactNode
  eyebrow: string
  title: string
  description: string
  children: ReactNode
  align?: 'start' | 'center' | 'end'
  triggerMode?: 'hover' | 'click'
}

export function HeaderHoverCard({
  trigger,
  eyebrow,
  title,
  description,
  children,
  align = 'end',
  triggerMode = 'hover',
}: HeaderHoverCardProps) {
  const [open, setOpen] = useState(false)
  const handleOpenChange = (nextOpen: boolean) => {
    if (triggerMode === 'hover') setOpen(nextOpen)
  }
  const triggerProps: HeaderHoverCardTriggerProps = {
    'aria-expanded': open,
    'aria-haspopup': 'dialog',
    onClick: () => setOpen((value) => !value),
    onKeyDown: (event) => {
      if (event.key === 'Escape') setOpen(false)
    },
  }

  return (
    <HoverCard.Root open={open} onOpenChange={handleOpenChange} openDelay={140} closeDelay={180}>
      <HoverCard.Trigger asChild>{trigger(triggerProps)}</HoverCard.Trigger>
      <HoverCard.Portal>
        <HoverCard.Content
          align={align}
          aria-label={title}
          className="shadow-surface-popover pointer-events-auto z-40 w-[min(34rem,calc(100vw-1.5rem))] rounded-[var(--radius-lg)] border border-border bg-panel"
          onEscapeKeyDown={() => setOpen(false)}
          onPointerDownOutside={() => setOpen(false)}
          role="dialog"
          side="bottom"
          sideOffset={10}
        >
          <div className="border-b border-border-soft px-4 py-3">
            <div className="type-label text-fg-faint">{eyebrow}</div>
            <h2 className="mt-1 type-panel-title text-foreground">{title}</h2>
            <p className="mt-1 max-w-none text-pretty text-[length:var(--density-type-caption-lg)] text-fg-dim">{description}</p>
          </div>
          <div className="space-y-3 p-4">{children}</div>
          <HoverCard.Arrow className="fill-[var(--color-panel)]" />
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  )
}
