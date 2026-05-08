import type { ReactNode } from 'react'

type SectionHeadProps = { icon?: ReactNode; children: ReactNode; right?: ReactNode }

export function SectionHead({ icon, children, right }: SectionHeadProps) {
  return (
    <div className="mt-[18px] mb-[8px] flex items-center justify-between gap-[12px] px-[18px]">
      <h3 className="type-panel-title flex items-center gap-[6px] text-foreground">
        {icon}
        {children}
      </h3>
      {right}
    </div>
  )
}
