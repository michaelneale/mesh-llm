import * as Popover from '@radix-ui/react-popover'
import { Check, Funnel } from 'lucide-react'
import type { ReactElement } from 'react'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { Tooltip } from '@/components/ui/Tooltip'
import { meshStatusTone } from '@/features/network/lib/mesh-status'
import {
  displayHostname,
  filterColumns,
  filterOptionLabel,
  roleLabel,
  rowAriaLabel,
  statusLabel,
  type FilterKey,
  type FilterOption,
  type SortDirection
} from '@/features/network/lib/peers-table-utils'
import { cn } from '@/lib/cn'
import { formatLatency } from '@/lib/format-latency'
import type { Peer } from '@/features/app-tabs/types'

export const PEERS_TABLE_GRID_COLUMNS =
  'minmax(7.75rem,0.58fr) minmax(4.25rem,5rem) minmax(5.5rem,8rem) minmax(5.5rem,6rem) minmax(14rem,1.6fr) minmax(4.75rem,5.25rem) minmax(4.5rem,5rem) minmax(7rem,7rem)'

function PeerValueTooltip({ content, children }: { content: string | undefined; children: ReactElement }) {
  if (!content) return children
  return (
    <Tooltip content={content} side="top">
      {children}
    </Tooltip>
  )
}

function RolePill({ role }: { role: NonNullable<Peer['role']> }) {
  const isYou = role === 'you'
  const label = roleLabel(role)
  return (
    <span
      className="inline-flex items-center rounded-full px-2 py-px text-[length:var(--density-type-caption)] font-medium"
      style={{
        background: isYou ? 'color-mix(in oklab, var(--color-accent) 16%, var(--color-background))' : 'transparent',
        color: isYou ? 'var(--color-accent)' : 'var(--color-fg-faint)',
        border: isYou
          ? '1px solid color-mix(in oklab, var(--color-accent) 28%, var(--color-background))'
          : '1px solid var(--color-border)'
      }}
    >
      {label}
    </span>
  )
}

function StatusPill({ peer }: { peer: Peer }) {
  const label = statusLabel(peer)
  const tone = meshStatusTone(peer)

  return (
    <StatusBadge dot size="caption" tone={tone}>
      {label}
    </StatusBadge>
  )
}

function ShareMeter({ sharePct, compact = false }: { sharePct: number; compact?: boolean }) {
  if (compact) {
    return (
      <div className="flex min-w-0 items-center justify-end gap-1.5">
        <span className="w-7 shrink-0 text-right font-mono text-[length:var(--density-type-caption)] text-fg-dim">
          {sharePct}%
        </span>
        <div
          className="h-[3px] w-14 shrink-0 overflow-hidden rounded-[3px] sm:w-20"
          style={{ background: 'color-mix(in oklab, var(--color-accent) 15%, transparent)' }}
        >
          <div className="h-full rounded-[3px] bg-accent" style={{ width: `${sharePct}%` }} />
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-w-0 items-center justify-end gap-1.5">
      <div
        className="h-[3px] shrink-0 flex-1 overflow-hidden rounded-[3px]"
        style={{ background: 'color-mix(in oklab, var(--color-accent) 15%, transparent)' }}
      >
        <div className="h-full rounded-[3px] bg-accent" style={{ width: `${sharePct}%` }} />
      </div>
      <span className="w-7 shrink-0 text-right font-mono text-[length:var(--density-type-caption)] text-fg-dim">
        {sharePct}%
      </span>
    </div>
  )
}

export function SortIndicator({ active, direction }: { active: boolean; direction: SortDirection }) {
  return (
    <span aria-hidden="true" className={cn('text-[10px]', active ? 'text-fg-dim' : 'text-fg-faint')}>
      {active ? (direction === 'asc' ? '↑' : '↓') : '↕'}
    </span>
  )
}

type PeerFilterPopoverProps = {
  optionsByColumn: Record<FilterKey, FilterOption[]>
  selectedValues: Record<FilterKey, Set<string>>
  activeFilterGroups: number
  visibleCount: number
  totalCount: number
  onValueChange: (key: FilterKey, value: string, checked: boolean) => void
  onSelectAll: (key: FilterKey) => void
  onClear: () => void
}

export function PeerFilterPopover({
  optionsByColumn,
  selectedValues,
  activeFilterGroups,
  visibleCount,
  totalCount,
  onValueChange,
  onSelectAll,
  onClear
}: PeerFilterPopoverProps) {
  const filtersActive = activeFilterGroups > 0

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button
          type="button"
          aria-controls="peer-table-filter"
          aria-haspopup="dialog"
          aria-label={filtersActive ? `Filter peers, ${activeFilterGroups} active` : 'Filter peers'}
          className={cn(
            'ui-control inline-flex h-7 shrink-0 items-center gap-1.5 rounded-[var(--radius)] px-2 text-[length:var(--density-type-caption)] outline-none transition-[border-color,background,color,box-shadow]',
            'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
            filtersActive &&
              'border-accent/45 bg-[color-mix(in_oklab,var(--color-accent)_12%,var(--color-panel))] text-fg shadow-surface-selected'
          )}
        >
          <Funnel aria-hidden={true} className="size-3.5" strokeWidth={1.8} />
          <span className="hidden sm:inline">Filter</span>
          {filtersActive ? (
            <span className="rounded-full bg-accent px-1.5 py-px font-mono text-[10px] leading-none text-accent-ink">
              {activeFilterGroups}
            </span>
          ) : null}
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          id="peer-table-filter"
          align="end"
          aria-label="Filter connected peers"
          className="surface-popover-panel z-50 w-[min(22rem,calc(100vw-2rem))] rounded-[var(--radius-lg)] border border-border bg-panel p-3 text-[length:var(--density-type-caption)] text-fg shadow-surface-popover outline-none"
          collisionPadding={12}
          side="bottom"
          sideOffset={8}
        >
          <div className="flex items-start justify-between gap-3 border-b border-border-soft pb-2.5">
            <div className="min-w-0">
              <div className="type-panel-title text-[length:var(--density-type-control)]">Filter peers</div>
              <p className="mt-0.5 text-[length:var(--density-type-caption)] text-fg-faint">
                Showing <span className="font-mono text-fg-dim">{visibleCount}</span> of{' '}
                <span className="font-mono text-fg-dim">{totalCount}</span>
              </p>
            </div>
            <button
              type="button"
              className="rounded-[var(--radius-sm)] px-2 py-1 text-[length:var(--density-type-caption)] text-fg-dim transition-colors hover:bg-panel-strong hover:text-fg disabled:pointer-events-none disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              disabled={!filtersActive}
              onClick={onClear}
            >
              Reset
            </button>
          </div>
          <div className="mt-2.5 grid max-h-[min(28rem,70vh)] gap-3 overflow-y-auto pr-1">
            {filterColumns.map((column) => {
              const options = optionsByColumn[column.key]
              const selected = selectedValues[column.key]
              const selectedCount = options.filter((option) => selected.has(option.value)).length
              const columnActive = selectedCount < options.length

              return (
                <section key={column.key} className="min-w-0">
                  <div className="mb-1.5 flex items-center justify-between gap-2">
                    <div className="type-label text-fg-faint">{column.label}</div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[10px] text-fg-faint">
                        {selectedCount}/{options.length}
                      </span>
                      <button
                        type="button"
                        className="rounded-[var(--radius-sm)] px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.07em] text-fg-dim transition-colors hover:bg-panel-strong hover:text-fg disabled:pointer-events-none disabled:opacity-35 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                        disabled={!columnActive}
                        onClick={() => onSelectAll(column.key)}
                      >
                        All
                      </button>
                    </div>
                  </div>
                  <div className="grid gap-1">
                    {options.map((option) => {
                      const checked = selected.has(option.value)
                      const label = filterOptionLabel(option.value)
                      return (
                        <label
                          key={option.value}
                          className="flex min-w-0 cursor-pointer items-center gap-2 rounded-[var(--radius)] px-2 py-1.5 transition-colors hover:bg-panel-strong focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-accent"
                        >
                          <input
                            type="checkbox"
                            className="sr-only"
                            aria-label={`${label}, ${option.count} peers`}
                            checked={checked}
                            onChange={(event) => onValueChange(column.key, option.value, event.currentTarget.checked)}
                          />
                          <span
                            aria-hidden={true}
                            className={cn(
                              'grid size-4 shrink-0 place-items-center rounded-[4px] border transition-colors',
                              checked
                                ? 'border-accent bg-accent text-accent-ink'
                                : 'border-border bg-panel-strong text-transparent'
                            )}
                          >
                            <Check className="size-3" strokeWidth={2.2} />
                          </span>
                          <span className="min-w-0 flex-1 truncate font-mono text-fg-dim">{label}</span>
                          <span className="shrink-0 font-mono text-[10px] text-fg-faint">{option.count}</span>
                        </label>
                      )
                    })}
                  </div>
                </section>
              )
            })}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  )
}

type PeerRowProps = {
  peer: Peer
  active: boolean
  isLast: boolean
  onSelect?: (peer: Peer) => void
  onHoverPeerIdChange?: (peerId: string | undefined) => void
}

export function PeerRow({ peer, active, isLast, onSelect, onHoverPeerIdChange }: PeerRowProps) {
  const hostedModels = peer.hostedModels.join(' ')
  const primaryId = peer.shortId ?? peer.id
  const hostname = displayHostname(peer)
  const peerTooltip = hostname ? `${peer.id} · ${hostname}` : peer.id

  return (
    <button
      aria-label={rowAriaLabel(peer, active)}
      data-active={active ? 'true' : undefined}
      onClick={() => onSelect?.(peer)}
      onFocus={() => onHoverPeerIdChange?.(peer.id)}
      onBlur={() => onHoverPeerIdChange?.(undefined)}
      onPointerEnter={() => onHoverPeerIdChange?.(peer.id)}
      onPointerLeave={() => onHoverPeerIdChange?.(undefined)}
      type="button"
      className={cn(
        'ui-row-action w-full min-w-0 px-3.5 py-[11px] text-left lg:grid lg:min-w-[760px] lg:items-center lg:gap-x-4',
        !isLast && 'border-b border-border-soft',
        active ? 'bg-[color-mix(in_oklab,var(--color-accent)_10%,var(--color-panel))]' : 'bg-transparent'
      )}
      style={{ gridTemplateColumns: PEERS_TABLE_GRID_COLUMNS }}
    >
      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-2 lg:contents">
        <div className="flex min-w-0 items-center gap-2 lg:contents">
          <PeerValueTooltip content={peerTooltip}>
            <div className="flex min-w-0 items-center gap-1.5">
              <span className="min-w-0 max-w-[5rem] truncate font-mono text-[length:var(--density-type-control)]">
                {primaryId}
              </span>
              {hostname && (
                <>
                  <span className="text-[length:var(--density-type-caption)] text-fg-faint">·</span>
                  <span className="min-w-0 truncate font-mono text-[length:var(--density-type-caption)] text-fg-dim">
                    {hostname}
                  </span>
                </>
              )}
            </div>
          </PeerValueTooltip>
          <div className="hidden lg:block">{peer.role && <RolePill role={peer.role} />}</div>
          <PeerValueTooltip content={peer.version}>
            <div className="hidden min-w-0 truncate font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim lg:block">
              {peer.version ?? '—'}
            </div>
          </PeerValueTooltip>
          <div className="hidden lg:block">
            <StatusPill peer={peer} />
          </div>
          <PeerValueTooltip content={hostedModels || undefined}>
            <div className="hidden min-w-0 truncate font-mono text-[length:var(--density-type-caption)] text-fg-dim lg:block">
              {hostedModels || '—'}
            </div>
          </PeerValueTooltip>
        </div>
        <div className="flex items-center justify-end gap-1.5 lg:hidden">
          {peer.role && <RolePill role={peer.role} />}
          <StatusPill peer={peer} />
        </div>
        <PeerValueTooltip content={hostedModels || undefined}>
          <div className="col-span-2 hidden min-w-0 truncate font-mono text-[length:var(--density-type-caption)] text-fg-dim min-[401px]:block lg:hidden">
            {hostedModels || '—'}
          </div>
        </PeerValueTooltip>
        <div className="col-span-2 grid min-w-0 grid-cols-[auto_auto_minmax(92px,1fr)] items-center gap-x-3 gap-y-1 lg:hidden">
          <div className="text-right font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim">
            {formatLatency(peer.latencyMs)} ms
          </div>
          <div className="text-right font-mono text-[length:var(--density-type-caption-lg)]">
            {peer.vramGB?.toFixed(1) ?? '—'} GB
          </div>
          <ShareMeter sharePct={peer.sharePct} compact />
        </div>
        <div className="hidden lg:contents">
          <div className="text-right font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim">
            {formatLatency(peer.latencyMs)} ms
          </div>
          <div className="text-right font-mono text-[length:var(--density-type-caption-lg)]">
            {peer.vramGB?.toFixed(1) ?? '—'} GB
          </div>
          <ShareMeter sharePct={peer.sharePct} />
        </div>
      </div>
    </button>
  )
}
