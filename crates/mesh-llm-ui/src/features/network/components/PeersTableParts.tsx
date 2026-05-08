import type { ReactElement, ReactNode } from 'react'
import { FilterPopover } from '@/components/ui/FilterPopover'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { Tooltip } from '@/components/ui/tooltip'
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
import type { PeerDTO } from '@/features/app-tabs/types'

export const PEERS_TABLE_GRID_COLUMNS =
  'minmax(7.75rem,0.58fr) minmax(14rem,1.6fr) minmax(5.5rem,8rem) minmax(4.5rem,5rem) minmax(7rem,7rem) minmax(4.75rem,5.25rem) minmax(4.25rem,5rem) minmax(5.5rem,6rem)'

function PeerValueTooltip({ content, children }: { content: ReactNode; children: ReactElement }) {
  if (!content) return children
  return (
    <Tooltip content={content} side="top">
      {children}
    </Tooltip>
  )
}

function RolePill({ role }: { role: NonNullable<PeerDTO['role']> }) {
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

function StatusPill({ peer }: { peer: PeerDTO }) {
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
  onSelectNone: (key: FilterKey) => void
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
  onSelectNone,
  onClear
}: PeerFilterPopoverProps) {
  return (
    <FilterPopover
      activeFilterGroups={activeFilterGroups}
      categories={filterColumns}
      contentLabel="Filter connected peers"
      formatOptionLabel={filterOptionLabel}
      id="peer-table-filter"
      itemLabel="peers"
      optionsByCategory={optionsByColumn}
      selectedValuesByCategory={selectedValues}
      title="Filter peers"
      totalCount={totalCount}
      triggerLabel="Filter peers"
      visibleCount={visibleCount}
      onClear={onClear}
      onSelectAll={onSelectAll}
      onSelectNone={onSelectNone}
      onValueChange={onValueChange}
    />
  )
}

type PeerRowProps = {
  peer: PeerDTO
  active: boolean
  isLast: boolean
  onSelect?: () => void
  onHoverPeerIdChange?: (peerId: string | undefined) => void
}

export function PeerRow({ peer, active, isLast, onSelect, onHoverPeerIdChange }: PeerRowProps) {
  const firstModel = peer.hostedModels[0] ?? null
  const extraModelCount = peer.hostedModels.length - 1
  const hostedModelsTooltip =
    peer.hostedModels.length > 0 ? (
      <div className="flex flex-col gap-0.5">
        {peer.hostedModels.map((model) => (
          <span key={model.name}>{model.name}</span>
        ))}
      </div>
    ) : undefined
  const primaryId = peer.shortId ?? peer.id
  const hostname = displayHostname(peer)
  const peerTooltip = hostname ? `${peer.id} · ${hostname}` : peer.id

  return (
    <button
      aria-label={rowAriaLabel(peer, active)}
      data-active={active ? 'true' : undefined}
      onClick={() => onSelect?.()}
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
            <div className="flex min-w-0 flex-col gap-0.5">
              <span className="min-w-0 truncate font-mono text-[length:var(--density-type-control)] leading-tight">
                {primaryId}
              </span>
              <span className="min-w-0 truncate font-mono text-[length:var(--density-type-caption)] leading-tight text-fg-dim">
                {hostname ?? '—'}
              </span>
            </div>
          </PeerValueTooltip>
          <PeerValueTooltip content={hostedModelsTooltip}>
            <div className="hidden min-w-0 lg:flex lg:items-center lg:gap-1.5">
              {firstModel ? (
                <>
                  <span className="min-w-0 truncate font-mono text-[length:var(--density-type-caption)] text-fg-dim">
                    {firstModel.name}
                  </span>
                  {extraModelCount > 0 && (
                    <span className="shrink-0 rounded-full border border-border px-1.5 py-px font-mono text-[10px] text-fg-faint">
                      +{extraModelCount} more
                    </span>
                  )}
                </>
              ) : (
                <span className="font-mono text-[length:var(--density-type-caption)] text-fg-dim">—</span>
              )}
            </div>
          </PeerValueTooltip>
          <PeerValueTooltip content={peer.version}>
            <div className="hidden min-w-0 truncate font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim lg:block">
              {peer.version ?? '—'}
            </div>
          </PeerValueTooltip>
          <div className="hidden text-right font-mono text-[length:var(--density-type-caption-lg)] lg:block">
            {peer.vramGB?.toFixed(1) ?? '—'} GB
          </div>
          <div className="hidden lg:block">
            <ShareMeter sharePct={peer.sharePct} />
          </div>
          <div className="hidden text-right font-mono text-[length:var(--density-type-caption-lg)] text-fg-dim lg:block">
            {formatLatency(peer.latencyMs)} ms
          </div>
          <div className="hidden justify-end text-right lg:flex">{peer.role && <RolePill role={peer.role} />}</div>
          <div className="hidden justify-end text-right lg:flex">
            <StatusPill peer={peer} />
          </div>
        </div>
        <div className="flex items-center justify-end gap-1.5 lg:hidden">
          {peer.role && <RolePill role={peer.role} />}
          <StatusPill peer={peer} />
        </div>
        <PeerValueTooltip content={hostedModelsTooltip}>
          <div className="col-span-2 hidden min-w-0 min-[401px]:flex min-[401px]:items-center min-[401px]:gap-1.5 lg:hidden">
            {firstModel ? (
              <>
                <span className="min-w-0 truncate font-mono text-[length:var(--density-type-caption)] text-fg-dim">
                  {firstModel.name}
                </span>
                {extraModelCount > 0 && (
                  <span className="shrink-0 rounded-full border border-border px-1.5 py-px font-mono text-[10px] text-fg-faint">
                    +{extraModelCount} more
                  </span>
                )}
              </>
            ) : (
              <span className="font-mono text-[length:var(--density-type-caption)] text-fg-dim">—</span>
            )}
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
      </div>
    </button>
  )
}
