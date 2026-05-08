type SectionLabelProps = { children: string; right?: string }
export function SectionLabel({ children, right }: SectionLabelProps) {
  return (
    <div className="mb-2 flex items-baseline justify-between text-[length:var(--density-type-annotation)] font-semibold uppercase tracking-[0.06em] text-fg-faint">
      <span>{children}</span>
      {right && (
        <span className="font-mono text-[length:var(--density-type-label)] font-medium normal-case tracking-normal text-fg-dim">
          {right}
        </span>
      )}
    </div>
  )
}
