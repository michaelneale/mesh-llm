type DecisionRowProps = { ok: boolean; label: string; detail?: string }
export function DecisionRow({ ok, label, detail }: DecisionRowProps) {
  return (
    <div className="flex items-start gap-2.5 px-0.5 py-2">
      <span
        className="mt-px inline-flex size-4 shrink-0 items-center justify-center rounded-full"
        style={{
          background: ok
            ? 'color-mix(in oklab, var(--color-good) 25%, transparent)'
            : 'color-mix(in oklab, var(--color-fg-faint) 15%, transparent)',
          color: ok ? 'var(--color-good)' : 'var(--color-fg-faint)'
        }}
      >
        <svg width="9" height="9" viewBox="0 0 10 10" aria-hidden="true">
          <path
            d={ok ? 'M1 5l3 3 5-6' : 'M2 2l6 6M8 2l-6 6'}
            stroke="currentColor"
            strokeWidth="1.8"
            fill="none"
            strokeLinecap="round"
          />
        </svg>
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-[length:var(--density-type-control)]">{label}</div>
        {detail && (
          <div className="mt-px font-mono text-[length:var(--density-type-label)] text-fg-faint">{detail}</div>
        )}
      </div>
    </div>
  )
}
