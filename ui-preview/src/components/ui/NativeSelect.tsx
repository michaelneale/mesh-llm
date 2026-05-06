import type { ChangeEventHandler } from 'react'

import { cn } from '@/lib/cn'

export type NativeSelectOption = {
  value: string
  label: string
}

type NativeSelectProps = {
  ariaLabel: string
  className?: string
  disabled?: boolean
  name: string
  onValueChange: (value: string) => void
  options: readonly NativeSelectOption[]
  value: string
}

export function NativeSelect({
  ariaLabel,
  className,
  disabled = false,
  name,
  onValueChange,
  options,
  value
}: NativeSelectProps) {
  const handleChange: ChangeEventHandler<HTMLSelectElement> = (event) => {
    onValueChange(event.currentTarget.value)
  }

  return (
    <select
      aria-label={ariaLabel}
      className={cn(
        'ui-control h-[32px] min-w-[240px] rounded-[var(--radius)] border bg-surface px-2.5 font-mono text-[length:var(--density-type-control)] text-foreground outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
        disabled && 'cursor-not-allowed opacity-60',
        className
      )}
      disabled={disabled}
      name={name}
      onChange={handleChange}
      value={value}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  )
}
