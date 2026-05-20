import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

type ReserveSummaryCellProps = {
  label: string
  value: ReactNode
  subLabel?: ReactNode
  children?: ReactNode
  mono?: boolean
  labelFirst?: boolean
  valueClassName?: string
}

export function ReserveSummaryCell({
  label,
  value,
  subLabel,
  children,
  mono,
  labelFirst,
  valueClassName
}: ReserveSummaryCellProps) {
  const labelElement = (
    <div className="text-[10.5px] font-semibold uppercase leading-none tracking-[0.08em] text-fg-faint">{label}</div>
  )
  const valueElement = (
    <div
      className={cn(
        'text-[22px] font-medium leading-[1.1] tracking-[-0.02em] text-foreground',
        mono ? 'font-mono' : '',
        valueClassName
      )}
    >
      {value}
    </div>
  )

  return (
    <div className="flex min-w-0 flex-col justify-center whitespace-nowrap border-b border-r border-border/60 px-[20px] py-3 lg:border-b-0 lg:px-[20px]">
      {labelFirst ? labelElement : valueElement}
      <div className={labelFirst ? 'mt-1.5' : 'mt-[3px]'}>{labelFirst ? valueElement : labelElement}</div>
      {subLabel ? <div className="mt-1 text-[10.5px] leading-snug text-fg-dim">{subLabel}</div> : null}
      {children ? <div className="mt-2">{children}</div> : null}
    </div>
  )
}
