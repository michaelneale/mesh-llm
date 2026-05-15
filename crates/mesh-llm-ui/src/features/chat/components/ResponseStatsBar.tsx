import { Eye } from 'lucide-react'
import { Fragment } from 'react'

type ResponseStatsBarProps = {
  tokens?: string
  tokPerSec?: string
  ttft?: string
  stopped?: boolean
  inspect?: () => void
  inspectLabel?: string
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

export function ResponseStatsBar({
  tokens,
  tokPerSec,
  ttft,
  stopped = false,
  inspect,
  inspectLabel
}: ResponseStatsBarProps) {
  if (!tokens && !tokPerSec && !ttft && !stopped && !inspect) return null

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
      {inspect ? (
        <>
          <span aria-hidden="true">·</span>
          <button
            type="button"
            className="inline-flex items-center gap-1 text-fg-faint outline-none transition-colors hover:text-fg-dim focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
            aria-label={inspectLabel ?? 'Inspect message'}
            onClick={(event) => {
              event.stopPropagation()
              inspect()
            }}
          >
            <Eye className="size-3" aria-hidden={true} />
            <span>Inspect</span>
          </button>
        </>
      ) : null}
    </span>
  )
}
