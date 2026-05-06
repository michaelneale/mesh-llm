import * as RadixTooltip from '@radix-ui/react-tooltip'
import type { ReactElement, ReactNode } from 'react'

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
        <RadixTooltip.Portal>
          <RadixTooltip.Content
            className="surface-menu-panel z-50 max-w-[240px] rounded-[var(--radius)] px-2.5 py-1.5 font-mono text-[length:var(--density-type-annotation)] leading-snug text-fg outline-none"
            collisionPadding={8}
            side={side}
            sideOffset={6}
          >
            {content}
            <RadixTooltip.Arrow className="fill-panel-strong stroke-border" height={5} width={9} />
          </RadixTooltip.Content>
        </RadixTooltip.Portal>
      </RadixTooltip.Root>
    </RadixTooltip.Provider>
  )
}
