import { Fragment } from 'react'

type ResponseStatsBarProps = {
  tokens?: string
  tokPerSec?: string
  ttft?: string
  stopped?: boolean
}

type ResponseStat = {
  label: string
  value: string
  prefix?: string
  title?: string
}

function splitNumericValue(value: string): { number: string; unit: string } {
  const match = value.match(/^([+-]?(?:\d+(?:\.\d+)?|\.\d+))(.*)$/)

  if (!match) return { number: value, unit: '' }

  return { number: match[1], unit: match[2] }
}

export function ResponseStatsBar({ tokens, tokPerSec, ttft, stopped = false }: ResponseStatsBarProps) {
  if (!tokens && !tokPerSec && !ttft && !stopped) return null

  const stats: ResponseStat[] = [
    { label: 'tokens', value: tokens ?? '0 tok' },
    { label: 'throughput', value: tokPerSec ?? '0.0 tok/s' },
    { label: 'TTFT', prefix: 'TTFT', value: ttft ?? '0ms' }
  ]

  return (
    <span className="mt-2 flex select-none items-center gap-3 text-[length:var(--density-type-label)] text-fg-faint">
      {stopped ? <span className="font-mono text-fg-muted">(stopped)</span> : null}
      {stopped && stats.length > 0 ? <span aria-hidden="true">·</span> : null}
      {stats.map((stat, index) => {
        const value = splitNumericValue(stat.value)

        return (
          <Fragment key={stat.label}>
            {index > 0 ? <span aria-hidden="true">·</span> : null}
            <span className="font-mono" title={stat.title}>
              {stat.prefix ? <span>{stat.prefix} </span> : null}
              <span className="text-primary">{value.number}</span>
              {value.unit ? <span>{value.unit}</span> : null}
            </span>
          </Fragment>
        )
      })}
    </span>
  )
}
