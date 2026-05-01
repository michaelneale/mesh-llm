import * as RadioGroup from '@radix-ui/react-radio-group'
import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

export type SegmentedControlOption = {
  value: string
  label: ReactNode
  description?: string
  disabled?: boolean
  selectedTone?: 'default' | 'accent'
}

type SegmentedControlVariant = 'buttons' | 'pill'

type SegmentedControlProps = {
  ariaDescribedBy?: string
  ariaLabel?: string
  ariaLabelledBy?: string
  className?: string
  disabled?: boolean
  itemClassName?: string
  itemTabIndex?: number
  name?: string
  orientation?: 'horizontal' | 'vertical'
  options: readonly SegmentedControlOption[]
  renderOption?: (option: SegmentedControlOption, selected: boolean) => ReactNode
  value: string
  variant?: SegmentedControlVariant
  onValueChange: (value: string) => void
}

const rootClassNameByVariant = {
  buttons: 'flex flex-wrap gap-1.5',
  pill: 'segmented-control inline-flex h-[28px] items-center rounded-full border p-[2px]',
} satisfies Record<SegmentedControlVariant, string>

const itemClassNameByVariant = {
  buttons: 'ui-control inline-flex h-[30px] items-center rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-control)] font-medium leading-none outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
  pill: 'segmented-control__item inline-flex h-6 min-w-[65px] items-center justify-center rounded-full border border-transparent px-3 text-[length:var(--density-type-caption)] font-medium leading-none outline-none transition-[background,color,box-shadow] duration-150 ease-out focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
} satisfies Record<SegmentedControlVariant, string>

export function SegmentedControl({
  ariaDescribedBy,
  ariaLabel,
  ariaLabelledBy,
  className,
  disabled = false,
  itemClassName,
  itemTabIndex,
  name,
  orientation = 'horizontal',
  options,
  renderOption,
  value,
  variant = 'buttons',
  onValueChange,
}: SegmentedControlProps) {
  return (
    <RadioGroup.Root
      aria-describedby={ariaDescribedBy}
      aria-label={ariaLabel}
      aria-labelledby={ariaLabelledBy}
      className={cn(rootClassNameByVariant[variant], className)}
      disabled={disabled}
      name={name}
      onValueChange={onValueChange}
      orientation={orientation}
      value={value}
    >
      {options.map((option) => {
        const selected = value === option.value
        const optionDisabled = disabled || option.disabled

        return (
          <RadioGroup.Item
            className={cn(itemClassNameByVariant[variant], variant === 'pill' && optionDisabled && 'cursor-not-allowed hover:bg-transparent', itemClassName)}
            data-active={variant === 'buttons' && selected ? 'true' : undefined}
            data-fixed-selected={variant === 'pill' && selected && optionDisabled ? 'true' : undefined}
            data-selected={variant === 'pill' && selected && !optionDisabled ? 'true' : undefined}
            data-selected-tone={variant === 'pill' && selected && !optionDisabled && option.selectedTone === 'accent' ? 'accent' : undefined}
            disabled={optionDisabled}
            key={option.value}
            tabIndex={itemTabIndex}
            title={option.description}
            value={option.value}
          >
            {renderOption ? renderOption(option, selected) : option.label}
          </RadioGroup.Item>
        )
      })}
    </RadioGroup.Root>
  )
}
