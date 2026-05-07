import { Slot } from '@radix-ui/react-slot'
import type { ComponentPropsWithoutRef, ReactNode } from 'react'
import { cn } from '@/lib/cn'

type EmptyStateTone = 'default' | 'accent' | 'destructive'

type EmptyStateProps = ComponentPropsWithoutRef<'div'> & {
  asChild?: boolean
  icon: ReactNode
  title: ReactNode
  description: ReactNode
  hint?: ReactNode
  tone?: EmptyStateTone
}

const iconToneClass: Record<EmptyStateTone, string> = {
  default: 'text-fg-faint',
  accent: 'text-accent',
  destructive: 'text-bad'
}

export function EmptyState({
  asChild = false,
  className,
  description,
  hint,
  icon,
  title,
  tone = 'default',
  ...props
}: EmptyStateProps) {
  const Component = asChild ? Slot : 'div'

  return (
    <Component
      className={cn('flex min-h-full items-center justify-center px-3 py-12 text-center', className)}
      {...props}
    >
      <div className="max-w-[30rem] space-y-5">
        <div className={cn('mx-auto flex size-10 items-center justify-center', iconToneClass[tone])}>{icon}</div>
        <div className="space-y-2">
          <h2 className="text-[length:var(--density-type-title)] font-semibold text-fg">{title}</h2>
          <p className="mx-auto max-w-[52ch] break-words text-[length:var(--density-type-body)] leading-6 text-fg-muted">
            {description}
          </p>
          {hint ? (
            <p className="mx-auto max-w-[52ch] text-[length:var(--density-type-body)] leading-6 text-fg-faint">
              {hint}
            </p>
          ) : null}
        </div>
      </div>
    </Component>
  )
}
