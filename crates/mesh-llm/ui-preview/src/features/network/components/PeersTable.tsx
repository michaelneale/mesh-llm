import { useEffect, useMemo, useState } from 'react'
import { PEERS_TABLE_GRID_COLUMNS, PeerFilterPopover, PeerRow, SortIndicator } from '@/features/network/components/PeersTableParts'
import {
  buildFilterOptions,
  clampPage,
  comparePeers,
  filterColumns,
  isColumnFiltered,
  nextSortState,
  peerFilterValue,
  selectedFilterSet,
  type FilterKey,
  type PeerFilterState,
  type SortKey,
  type SortState
} from '@/features/network/lib/peers-table-utils'
import { cn } from '@/lib/cn'
import type { Peer, PeerSummary } from '@/features/app-tabs/types'

type PeersTableProps = {
  peers: Peer[]
  summary: PeerSummary
  selectedPeerId?: string
  onSelect?: (peer: Peer) => void
  onHoverPeerIdChange?: (peerId: string | undefined) => void
  onFilteredPeerIdsChange?: (peerIds: string[]) => void
}
const PEERS_PAGE_SIZE = 10

const columns: Array<{ key: SortKey; label: string; align?: 'right' }> = [
  { key: 'id', label: 'ID' },
  { key: 'role', label: 'Role' },
  { key: 'version', label: 'Version' },
  { key: 'status', label: 'Status' },
  { key: 'hosted', label: 'Hosted' },
  { key: 'latency', label: 'Latency', align: 'right' },
  { key: 'vram', label: 'VRAM', align: 'right' },
  { key: 'share', label: 'Share', align: 'right' }
]

export function PeersTable({
  peers,
  summary,
  selectedPeerId,
  onSelect,
  onHoverPeerIdChange,
  onFilteredPeerIdsChange
}: PeersTableProps) {
  const [page, setPage] = useState(0)
  const [sort, setSort] = useState<SortState>({ key: 'id', direction: 'asc' })
  const [filters, setFilters] = useState<PeerFilterState>({})
  const filterOptionsByColumn = useMemo(
    () => ({
      role: buildFilterOptions(peers, 'role'),
      status: buildFilterOptions(peers, 'status'),
      version: buildFilterOptions(peers, 'version'),
      hosted: buildFilterOptions(peers, 'hosted')
    }),
    [peers]
  )
  const selectedFilterValues = useMemo<Record<FilterKey, Set<string>>>(
    () => ({
      role: selectedFilterSet(filters, 'role', filterOptionsByColumn.role),
      status: selectedFilterSet(filters, 'status', filterOptionsByColumn.status),
      version: selectedFilterSet(filters, 'version', filterOptionsByColumn.version),
      hosted: selectedFilterSet(filters, 'hosted', filterOptionsByColumn.hosted)
    }),
    [filterOptionsByColumn, filters]
  )
  const activeFilterGroups = filterColumns.filter((column) =>
    isColumnFiltered(filters, column.key, filterOptionsByColumn[column.key])
  ).length
  const filteredPeers = useMemo(
    () =>
      peers.filter((peer) =>
        filterColumns.every((column) => selectedFilterValues[column.key].has(peerFilterValue(peer, column.key)))
      ),
    [peers, selectedFilterValues]
  )
  useEffect(() => {
    onFilteredPeerIdsChange?.(filteredPeers.map((peer) => peer.id))
  }, [filteredPeers, onFilteredPeerIdsChange])
  const sortedPeers = useMemo(() => [...filteredPeers].sort((a, b) => comparePeers(a, b, sort)), [filteredPeers, sort])
  const pageCount = Math.max(1, Math.ceil(sortedPeers.length / PEERS_PAGE_SIZE))
  const selectedPeerIndex = useMemo(
    () => sortedPeers.findIndex((peer) => peer.id === selectedPeerId),
    [sortedPeers, selectedPeerId]
  )
  const selectedPeerPage = selectedPeerIndex >= 0 ? Math.floor(selectedPeerIndex / PEERS_PAGE_SIZE) : undefined
  const currentPage = clampPage(selectedPeerPage ?? page, pageCount)
  const showPagination = sortedPeers.length > PEERS_PAGE_SIZE
  const firstPeerIndex = currentPage * PEERS_PAGE_SIZE
  const visiblePeers = useMemo(
    () => sortedPeers.slice(firstPeerIndex, firstPeerIndex + PEERS_PAGE_SIZE),
    [firstPeerIndex, sortedPeers]
  )
  const visibleStart = sortedPeers.length === 0 ? 0 : firstPeerIndex + 1
  const visibleEnd = firstPeerIndex + visiblePeers.length

  function handleSort(key: SortKey) {
    setSort((current) => nextSortState(current, key))
    setPage(0)
  }

  function handleFilterValueChange(key: FilterKey, value: string, checked: boolean) {
    const options = filterOptionsByColumn[key]

    setFilters((current) => {
      const currentSelected = selectedFilterSet(current, key, options)

      if (checked) {
        currentSelected.add(value)
      } else {
        currentSelected.delete(value)
      }

      const optionValues = options.map((option) => option.value)
      const nextSelected = optionValues.filter((optionValue) => currentSelected.has(optionValue))
      const nextFilters = { ...current }

      if (nextSelected.length === optionValues.length) {
        delete nextFilters[key]
      } else {
        nextFilters[key] = nextSelected
      }

      return nextFilters
    })
    setPage(0)
  }

  function handleSelectAll(key: FilterKey) {
    setFilters((current) => {
      const nextFilters = { ...current }
      delete nextFilters[key]
      return nextFilters
    })
    setPage(0)
  }

  function handleClearFilters() {
    setFilters({})
    setPage(0)
  }

  return (
    <section className="panel-shell min-w-0 overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel [contain:inline-size]">
      <header className="flex items-center justify-between gap-3 border-b border-border-soft px-3.5 py-2.5">
        <h2 className="type-panel-title">Connected peers</h2>
        <div className="flex min-w-0 items-center gap-2 text-[length:var(--density-type-caption)] text-fg-faint">
          <PeerFilterPopover
            activeFilterGroups={activeFilterGroups}
            optionsByColumn={filterOptionsByColumn}
            selectedValues={selectedFilterValues}
            totalCount={summary.total}
            visibleCount={sortedPeers.length}
            onClear={handleClearFilters}
            onSelectAll={handleSelectAll}
            onValueChange={handleFilterValueChange}
          />
          {showPagination && (
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                aria-label="Previous peers page"
                className="grid size-6 place-items-center rounded-full border border-border text-fg-dim transition-colors hover:border-border-strong hover:text-fg disabled:pointer-events-none disabled:opacity-40"
                disabled={currentPage === 0}
                onClick={() => setPage((previous) => clampPage(previous - 1, pageCount))}
              >
                ‹
              </button>
              <span className="whitespace-nowrap font-mono text-fg-dim">
                {visibleStart}-{visibleEnd}
              </span>
              <button
                type="button"
                aria-label="Next peers page"
                className="grid size-6 place-items-center rounded-full border border-border text-fg-dim transition-colors hover:border-border-strong hover:text-fg disabled:pointer-events-none disabled:opacity-40"
                disabled={currentPage >= pageCount - 1}
                onClick={() => setPage((previous) => clampPage(previous + 1, pageCount))}
              >
                ›
              </button>
              <span aria-hidden="true" className="text-fg-faint">
                ·
              </span>
            </div>
          )}
          <span className="sr-only" aria-live="polite">
            {sortedPeers.length} of {summary.total} peers visible
          </span>
          <span className="whitespace-nowrap">
            {activeFilterGroups > 0 ? `${sortedPeers.length} of ${summary.total}` : summary.total} total
          </span>
          <span aria-hidden="true" className="text-fg-faint">
            ·
          </span>
          <span className="whitespace-nowrap text-good">● {summary.capacity}</span>
        </div>
      </header>
      <div className="max-w-full overflow-x-auto [contain:inline-size]">
        <div
          className="type-label hidden min-w-[760px] border-b border-border-soft px-3.5 py-2 text-fg-faint lg:grid lg:gap-x-4"
          style={{ gridTemplateColumns: PEERS_TABLE_GRID_COLUMNS }}
        >
          {columns.map((column) => {
            const active = sort.key === column.key
            return (
              <button
                key={column.key}
                type="button"
                aria-label={`Sort peers by ${column.label} ${active && sort.direction === 'asc' ? 'descending' : 'ascending'}`}
                aria-pressed={active}
                className={cn(
                  'flex min-w-0 items-center gap-1.5 rounded-sm text-left transition-colors hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/45',
                  column.align === 'right' && 'justify-end text-right',
                  active ? 'text-fg-dim' : 'text-fg-faint'
                )}
                onClick={() => handleSort(column.key)}
              >
                <span className="min-w-0 truncate">{column.label}</span>
                <SortIndicator active={active} direction={sort.direction} />
              </button>
            )
          })}
        </div>
        {visiblePeers.length > 0 ? (
          visiblePeers.map((peer, i) => (
            <PeerRow
              key={peer.id}
              peer={peer}
              active={peer.id === selectedPeerId}
              isLast={i === visiblePeers.length - 1}
              onSelect={onSelect}
              onHoverPeerIdChange={onHoverPeerIdChange}
            />
          ))
        ) : (
          <div className="px-3.5 py-8 text-center">
            <p className="text-[length:var(--density-type-control)] font-semibold text-fg">
              {activeFilterGroups > 0 ? 'No peers match these filters.' : 'No connected peers yet.'}
            </p>
            {activeFilterGroups > 0 ? (
              <button
                type="button"
                className="mt-2 rounded-[var(--radius)] px-2 py-1 text-[length:var(--density-type-caption)] text-accent transition-colors hover:bg-panel-strong focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                onClick={handleClearFilters}
              >
                Clear filters
              </button>
            ) : null}
          </div>
        )}
      </div>
    </section>
  )
}
