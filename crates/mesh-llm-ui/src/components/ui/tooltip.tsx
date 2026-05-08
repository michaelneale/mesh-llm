import * as RadixTooltip from '@radix-ui/react-tooltip'
import type { ReactElement, ReactNode } from 'react'

// eslint-disable-next-line react-refresh/only-export-components -- re-exports of Radix UI components for convenience
export const TooltipProvider = RadixTooltip.Provider
// eslint-disable-next-line react-refresh/only-export-components -- re-exports of Radix UI components for convenience
export const TooltipRoot = RadixTooltip.Root
// eslint-disable-next-line react-refresh/only-export-components -- re-exports of Radix UI components for convenience
export const TooltipTrigger = RadixTooltip.Trigger
// eslint-disable-next-line react-refresh/only-export-components -- re-exports of Radix UI components for convenience
export const TooltipPortal = RadixTooltip.Portal
// eslint-disable-next-line react-refresh/only-export-components -- re-exports of Radix UI components for convenience
export const TooltipArrow = RadixTooltip.Arrow

export function TooltipContent({
  className,
  sideOffset = 6,
  ...props
}: React.ComponentPropsWithoutRef<typeof RadixTooltip.Content>) {
  return (
    <RadixTooltip.Portal>
      <RadixTooltip.Content
        sideOffset={sideOffset}
        className={`surface-menu-panel z-50 max-w-[240px] rounded-[var(--radius)] px-2.5 py-1.5 font-mono text-[length:var(--density-type-annotation)] leading-snug text-fg outline-none ${className ?? ''}`}
        collisionPadding={8}
        {...props}
      >
        {props.children}
        <RadixTooltip.Arrow className="fill-panel-strong stroke-border" height={5} width={9} />
      </RadixTooltip.Content>
    </RadixTooltip.Portal>
  )
}

type TooltipProps = {
  children: ReactElement
  content: ReactNode
  side?: RadixTooltip.TooltipContentProps['side']
}

export function Tooltip({ children, content, side = 'top' }: TooltipProps) {
  return (
    <RadixTooltip.Provider delayDuration={250} skipDelayDuration={120}>
      <RadixTooltip.Root>
        <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
        <TooltipContent side={side}>{content}</TooltipContent>
      </RadixTooltip.Root>
    </RadixTooltip.Provider>
  )
}
