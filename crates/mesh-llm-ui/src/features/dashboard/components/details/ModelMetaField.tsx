import type { ReactNode } from 'react'

export function ModelMetaField({
  label,
  icon,
  action,
  children
}: {
  label: string
  icon?: ReactNode
  action?: ReactNode
  children: ReactNode
}) {
  return (
    <div className="rounded-lg border bg-muted/25 px-3 py-2">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          {icon ? <span className="shrink-0">{icon}</span> : null}
          <span>{label}</span>
        </div>
        {action}
      </div>
      {children}
    </div>
  )
}
