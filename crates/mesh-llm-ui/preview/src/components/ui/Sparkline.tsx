import type { SVGProps } from 'react'

type SparklineProps = Omit<SVGProps<SVGSVGElement>, 'children' | 'color' | 'height' | 'values' | 'width'> & {
  values: number[]
  color?: string
  width?: number
  height?: number
  pointCount?: number
  emptyValue?: number
  strokeWidth?: number
  ariaLabel?: string
}

function normalizePointCount(pointCount: number | undefined) {
  return typeof pointCount === 'number' && Number.isFinite(pointCount) ? Math.max(1, Math.floor(pointCount)) : undefined
}

function cleanSparklineValues(values: number[], pointCount: number | undefined, emptyValue: number) {
  const cleanValues = values.map((value) => (Number.isFinite(value) ? value : emptyValue))
  const normalizedPointCount = normalizePointCount(pointCount)

  if (!normalizedPointCount) return cleanValues

  if (cleanValues.length >= normalizedPointCount) return cleanValues.slice(-normalizedPointCount)

  return [...Array.from({ length: normalizedPointCount - cleanValues.length }, () => emptyValue), ...cleanValues]
}

function sparklineY(value: number, max: number, min: number, baseline: number) {
  if (max === min) return baseline

  const verticalRange = Math.max(1, baseline - 1)
  const magnitude = Math.max(Math.abs(max), Math.abs(min), 1)
  return baseline - (value / magnitude) * verticalRange
}

export function Sparkline({
  values,
  color = 'var(--color-accent)',
  width = 72,
  height = 18,
  pointCount,
  emptyValue = 0,
  strokeWidth = 1.2,
  ariaLabel,
  className = 'shrink-0',
  style,
  ...svgProps
}: SparklineProps) {
  const cleanValues = cleanSparklineValues(values, pointCount, emptyValue)

  if (!cleanValues.length) return null

  const max = Math.max(...cleanValues)
  const min = Math.min(...cleanValues)
  const range = max - min
  const denominator = Math.max(1, cleanValues.length - 1)
  const baseline = height / 2
  const points = cleanValues
    .map((value, index) => {
      const x = (index / denominator) * width
      const y = range > 0 ? sparklineY(value, max, min, baseline) : baseline
      return `${x},${y}`
    })
    .join(' ')

  return (
    <svg
      aria-hidden={ariaLabel ? undefined : true}
      aria-label={ariaLabel}
      className={className}
      height={height}
      role={ariaLabel ? 'img' : undefined}
      style={{ display: 'block', ...style }}
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      {...svgProps}
    >
      <polyline
        fill="none"
        points={points}
        stroke={color}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={strokeWidth}
        vectorEffect="non-scaling-stroke"
      />
      <polyline points={`0,${baseline} ${points} ${width},${baseline}`} fill={color} opacity="0.08" stroke="none" />
    </svg>
  )
}
