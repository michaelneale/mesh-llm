import { Badge } from '@/components/ui/badge'
import { ReservesSurface } from '@/features/reserves/components/ReservesSurface'
import { RESERVE_PROVIDER_FIXTURES } from '@/features/reserves/lib/reserve-fixtures'

export function ReservesPreviewArea() {
  return (
    <div className="space-y-4">
      <section className="rounded-[var(--radius-lg)] border border-border bg-panel px-4 py-3.5">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="type-panel-title text-foreground">Reserves mockup preview</h2>
          <Badge className="rounded-full px-2 py-0.5 text-[10px] uppercase tracking-[0.08em] text-fg-dim">
            Playground only
          </Badge>
        </div>
        <p className="type-body mt-2 max-w-[72ch] text-fg-dim">
          Use this playground slice to verify the high-fidelity Reserves surface, collapsed provider groups, reserve
          policy panel, and preview-only dialogs without touching live reserve data.
        </p>
      </section>

      <ReservesSurface
        configurationHref="/configuration/wake-policy"
        liveMeshVramGB={512}
        providers={RESERVE_PROVIDER_FIXTURES}
      />
    </div>
  )
}
