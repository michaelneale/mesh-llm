import { SegmentedControl, type SegmentedControlOption } from '@/components/ui/SegmentedControl'
import { Tooltip } from '@/components/ui/Tooltip'
import { cn } from '@/lib/cn'
import type { Placement } from '@/features/app-tabs/types'

const PLACEMENT_OPTIONS: readonly SegmentedControlOption[] = [
  { value: 'separate', label: 'separate' },
  { value: 'pooled', label: 'pooled' }
]

function isPlacement(value: string): value is Placement {
  return value === 'separate' || value === 'pooled'
}

type PlacementToggleProps = {
  disabled?: boolean
  disabledReason?: string
  groupId: string
  itemTabIndex?: number
  placement: Placement
  onChange: (placement: Placement) => void
}

export function PlacementToggle({
  disabled = false,
  disabledReason,
  groupId,
  itemTabIndex,
  placement,
  onChange
}: PlacementToggleProps) {
  const descriptionId = disabledReason ? `${groupId}-placement-reason` : undefined
  const legendId = `${groupId}-placement-legend`
  const control = (
    <fieldset
      aria-describedby={descriptionId}
      className={cn('inline-flex', disabled && 'opacity-60')}
      disabled={disabled}
    >
      <legend id={legendId} className="sr-only">
        VRAM placement
      </legend>
      {descriptionId ? (
        <span id={descriptionId} className="sr-only">
          {disabledReason}
        </span>
      ) : null}
      <SegmentedControl
        ariaDescribedBy={descriptionId}
        ariaLabelledBy={legendId}
        disabled={disabled}
        name={`vram-placement-${groupId}`}
        onValueChange={(nextValue) => {
          if (isPlacement(nextValue)) onChange(nextValue)
        }}
        options={PLACEMENT_OPTIONS}
        itemTabIndex={itemTabIndex}
        value={placement}
        variant="pill"
      />
    </fieldset>
  )

  if (!disabledReason) return control

  return (
    <Tooltip content={disabledReason} side="bottom">
      <span
        className="inline-flex cursor-help rounded-[var(--radius)] outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
        tabIndex={itemTabIndex ?? 0}
      >
        {control}
      </span>
    </Tooltip>
  )
}
