import type { ReactNode } from 'react'
import { AccentIconFrame } from '@/components/ui/AccentIconFrame'
import { cn } from '@/lib/cn'

type InfoBannerProps = {
  title: ReactNode
  description: ReactNode
  action?: ReactNode
  actionClassName?: string
  className?: string
  contentClassName?: string
  descriptionClassName?: string
  leadingIcon?: ReactNode
  leadingIconClassName?: string
  status?: ReactNode
  titleClassName?: string
  titleId?: string
  titleLevel?: 'h1' | 'h2' | 'h3'
}

export function InfoBanner({
  title,
  description,
  action,
  actionClassName,
  className,
  contentClassName,
  descriptionClassName,
  leadingIcon,
  leadingIconClassName,
  status,
  titleClassName,
  titleId,
  titleLevel = 'h2'
}: InfoBannerProps) {
  const Heading = titleLevel

  return (
    <section
      aria-labelledby={titleId}
      className={cn(
        'panel-shell flex items-center gap-5 rounded-[var(--radius-lg)] border border-border px-5 py-4',
        className
      )}
      style={{
        background:
          'linear-gradient(90deg, color-mix(in oklab, var(--color-accent) 10%, var(--color-panel)) 0%, var(--color-panel) 60%)'
      }}
    >
      {leadingIcon ? <AccentIconFrame className={leadingIconClassName}>{leadingIcon}</AccentIconFrame> : null}
      <div className={cn('min-w-0 flex-1', contentClassName)}>
        <div className="flex flex-wrap items-center gap-2">
          <Heading
            id={titleId}
            className={cn(
              titleLevel === 'h1'
                ? 'type-headline'
                : 'text-[length:var(--density-type-title)] font-semibold leading-tight text-foreground',
              titleClassName
            )}
          >
            {title}
          </Heading>
          {status ? <div>{status}</div> : null}
        </div>
        <div className={cn('type-caption mt-1.5 text-fg-dim', descriptionClassName)}>{description}</div>
      </div>
      {action ? (
        <div className={cn('flex shrink-0 items-center justify-end self-center', actionClassName)}>{action}</div>
      ) : null}
    </section>
  )
}
