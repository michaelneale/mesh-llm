import type { ChangeEventHandler, CSSProperties, ReactNode } from 'react'

import { cn } from '@/lib/cn'

type SliderProgressStyle = CSSProperties & {
  '--slider-progress': string
}

export type SliderBoundary = {
  inclusive?: boolean
  value: ReactNode
}

export type SliderValueLabelAlign = 'left' | 'center' | 'right'

export type SliderValueLabelPlacement = 'inline' | 'top' | 'bottom'

export type SliderProps = {
  ariaDescribedBy?: string
  ariaLabel: string
  ariaValueText?: string
  className?: string
  disabled?: boolean
  formatValue?: (value: string) => ReactNode
  inputClassName?: string
  label?: ReactNode
  lowerBound?: SliderBoundary
  max: number
  min: number
  name: string
  onValueChange: (value: string) => void
  step?: number | string
  unit?: ReactNode
  upperBound?: SliderBoundary
  value: string
  valueClassName?: string
  valueLabelAlign?: SliderValueLabelAlign
  valueLabelPlacement?: SliderValueLabelPlacement
  valueLabel?: ReactNode
}

function sliderProgress(value: string, min: number, max: number) {
  const numericValue = Number(value)
  if (!Number.isFinite(numericValue) || max <= min) return '0%'

  const clamped = Math.max(min, Math.min(max, numericValue))
  return `${((clamped - min) / (max - min)) * 100}%`
}

function valueLabelContent(
  value: string,
  valueLabel: ReactNode | undefined,
  unit: ReactNode | undefined,
  formatValue: ((value: string) => ReactNode) | undefined
) {
  if (valueLabel !== undefined && valueLabel !== null) return valueLabel

  const formattedValue = formatValue ? formatValue(value) : value
  if (unit === undefined || unit === null) return formattedValue

  return (
    <>
      <span className="font-mono text-[length:var(--density-type-caption)] font-medium tabular-nums text-fg-dim">
        {formattedValue}
      </span>
      <span className="text-[length:var(--density-type-caption)] font-medium text-fg-dim">{unit}</span>
    </>
  )
}

function textFromNode(node: ReactNode) {
  return typeof node === 'string' || typeof node === 'number' ? String(node) : undefined
}

function valueLabelText(
  value: string,
  valueLabel: ReactNode | undefined,
  unit: ReactNode | undefined,
  formatValue: ((value: string) => ReactNode) | undefined
) {
  const labelText = textFromNode(valueLabel)
  if (labelText !== undefined) return labelText

  const formattedText = textFromNode(formatValue ? formatValue(value) : value)
  if (formattedText === undefined) return undefined

  const unitText = textFromNode(unit)
  return unitText === undefined ? formattedText : `${formattedText} ${unitText}`
}

function boundaryLabel(boundary: SliderBoundary | undefined, side: 'lower' | 'upper') {
  if (!boundary) return null

  const inclusive = boundary.inclusive ?? true
  const marker = side === 'lower' ? (inclusive ? '[' : '(') : inclusive ? ']' : ')'

  return (
    <span className="inline-flex items-baseline gap-1 font-mono text-[length:var(--density-type-annotation)] text-fg-faint">
      {side === 'lower' ? <span aria-hidden="true">{marker}</span> : null}
      <span>{boundary.value}</span>
      {side === 'upper' ? <span aria-hidden="true">{marker}</span> : null}
    </span>
  )
}

function valueLabelAlignClassName(align: SliderValueLabelAlign) {
  if (align === 'left') return 'justify-start justify-self-start text-left'
  if (align === 'center') return 'justify-center justify-self-center text-center'
  return 'justify-end justify-self-end text-right'
}

export function Slider({
  ariaDescribedBy,
  ariaLabel,
  ariaValueText,
  className,
  disabled = false,
  formatValue,
  inputClassName,
  label,
  lowerBound,
  max,
  min,
  name,
  onValueChange,
  step,
  unit,
  upperBound,
  value,
  valueLabelAlign = 'right',
  valueClassName,
  valueLabelPlacement = 'inline',
  valueLabel
}: SliderProps) {
  const handleChange: ChangeEventHandler<HTMLInputElement> = (event) => {
    onValueChange(event.currentTarget.value)
  }

  const progressStyle: SliderProgressStyle = {
    '--slider-progress': sliderProgress(value, min, max)
  }
  const resolvedValueLabel = valueLabelContent(value, valueLabel, unit, formatValue)
  const resolvedAriaValueText = ariaValueText ?? valueLabelText(value, valueLabel, unit, formatValue)
  const hasLabel = label !== undefined && label !== null
  const hasBounds = lowerBound !== undefined || upperBound !== undefined
  const valueLabelClassName = cn(
    'inline-flex min-w-[64px] items-baseline gap-1 text-[length:var(--density-type-caption)] font-medium tabular-nums text-fg-dim',
    valueLabelAlignClassName(valueLabelAlign),
    valueClassName
  )

  return (
    <div className={cn('grid min-w-[280px] gap-1.5', disabled && 'opacity-60', className)}>
      {hasLabel || valueLabelPlacement === 'top' ? (
        <div className="flex items-baseline justify-between gap-3">
          {hasLabel ? (
            <span className="text-[length:var(--density-type-control)] font-medium text-foreground">{label}</span>
          ) : (
            <span />
          )}
          {valueLabelPlacement === 'top' ? <span className={valueLabelClassName}>{resolvedValueLabel}</span> : null}
        </div>
      ) : null}
      <div className="flex items-center gap-3">
        <input
          aria-describedby={ariaDescribedBy}
          aria-label={ariaLabel}
          aria-valuetext={resolvedAriaValueText}
          className={cn(
            'ui-slider min-w-[180px] flex-1 outline-none',
            disabled ? 'cursor-not-allowed' : 'cursor-pointer',
            inputClassName
          )}
          disabled={disabled}
          max={max}
          min={min}
          name={name}
          onChange={handleChange}
          step={step}
          style={progressStyle}
          type="range"
          value={value}
        />
        {valueLabelPlacement === 'inline' ? <span className={valueLabelClassName}>{resolvedValueLabel}</span> : null}
      </div>
      {valueLabelPlacement === 'bottom' ? <span className={valueLabelClassName}>{resolvedValueLabel}</span> : null}
      {hasBounds ? (
        <div className="flex items-center justify-between gap-3 px-px">
          {boundaryLabel(lowerBound, 'lower') ?? <span />}
          {boundaryLabel(upperBound, 'upper') ?? <span />}
        </div>
      ) : null}
    </div>
  )
}
