type DetailPillProps = { label: string; value: string | number }

export function DetailPill({ label, value }: DetailPillProps) {
  return (
    <span className="inline-flex min-w-0 items-baseline gap-1.5 rounded-[4px] border border-border-soft bg-background px-2 py-[3px]">
      <span className="shrink-0 text-[length:var(--density-type-micro)] font-medium uppercase tracking-[0.055em] text-fg-faint">{label}</span>
      <span className="truncate font-mono text-[length:var(--density-type-caption)] font-medium text-fg">{value}</span>
    </span>
  )
}
