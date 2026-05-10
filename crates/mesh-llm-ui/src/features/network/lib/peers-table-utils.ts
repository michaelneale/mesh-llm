import { meshStatusLabel } from '@/features/network/lib/mesh-status'
import type { Peer } from '@/features/app-tabs/types'

export type SortKey = 'id' | 'role' | 'version' | 'status' | 'hosted' | 'latency' | 'vram' | 'share'
export type SortDirection = 'asc' | 'desc'
export type SortState = { key: SortKey; direction: SortDirection }
export type FilterKey = 'role' | 'version' | 'status' | 'hosted'
export type PeerFilterState = Partial<Record<FilterKey, string[]>>
export type FilterOption = { value: string; count: number }

export const filterColumns = [
  { key: 'role', label: 'Role' },
  { key: 'status', label: 'Status' },
  { key: 'version', label: 'Version' },
  { key: 'hosted', label: 'Hosted' }
] satisfies Array<{ key: FilterKey; label: string }>

export function roleLabel(role: NonNullable<Peer['role']>): string {
  if (role === 'you') return 'You'
  if (role === 'host') return 'Host'
  if (role === 'client') return 'Client'
  if (role === 'worker') return 'Worker'
  return 'Peer'
}

export function statusLabel(peer: { nodeState?: Peer['nodeState']; status: Peer['status'] }): string {
  return meshStatusLabel(peer)
}

export function peerFilterValue(peer: Peer, key: FilterKey): string {
  if (key === 'role') return peer.role ? roleLabel(peer.role) : 'Peer'
  if (key === 'status') return statusLabel(peer)
  if (key === 'version') return peer.version ?? '—'
  return peer.hostedModels.join(' ') || '—'
}

export function buildFilterOptions(peers: Peer[], key: FilterKey): FilterOption[] {
  const counts = new Map<string, number>()

  for (const peer of peers) {
    const value = peerFilterValue(peer, key)
    counts.set(value, (counts.get(value) ?? 0) + 1)
  }

  return [...counts.entries()].sort(([a], [b]) => compareText(a, b)).map(([value, count]) => ({ value, count }))
}

export function selectedFilterSet(filters: PeerFilterState, key: FilterKey, options: FilterOption[]): Set<string> {
  const optionValues = options.map((option) => option.value)
  const selected = filters[key]

  if (!selected) return new Set(optionValues)

  return new Set(selected.filter((value) => optionValues.includes(value)))
}

export function isColumnFiltered(filters: PeerFilterState, key: FilterKey, options: FilterOption[]): boolean {
  if (options.length === 0) return false
  return selectedFilterSet(filters, key, options).size < options.length
}

export function compareText(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })
}

export function comparePeers(a: Peer, b: Peer, sort: SortState): number {
  const numericCompare = (left: number | null | undefined, right: number | null | undefined) =>
    (left ?? -1) - (right ?? -1)
  const result =
    sort.key === 'id'
      ? compareText(`${a.shortId ?? a.id} ${a.hostname}`, `${b.shortId ?? b.id} ${b.hostname}`)
      : sort.key === 'role'
        ? compareText(a.role ? roleLabel(a.role) : '', b.role ? roleLabel(b.role) : '')
        : sort.key === 'version'
          ? compareText(a.version ?? '', b.version ?? '')
          : sort.key === 'status'
            ? compareText(statusLabel(a), statusLabel(b))
            : sort.key === 'hosted'
              ? compareText(a.hostedModels.join(' '), b.hostedModels.join(' '))
              : sort.key === 'latency'
                ? numericCompare(a.latencyMs, b.latencyMs)
                : sort.key === 'vram'
                  ? numericCompare(a.vramGB, b.vramGB)
                  : numericCompare(a.sharePct, b.sharePct)

  const resolvedResult = result === 0 ? compareText(a.hostname, b.hostname) : result
  return sort.direction === 'asc' ? resolvedResult : -resolvedResult
}

function isIdentifierLike(value: string): boolean {
  return /^[a-f0-9-]{8,}$/i.test(value)
}

export function displayHostname(peer: { hostname: string; id: string; shortId?: string }): string | undefined {
  const hostname = peer.hostname.trim()
  if (!hostname) return undefined
  if (hostname === peer.id || hostname === peer.shortId) return undefined
  if (peer.id.startsWith(hostname) && isIdentifierLike(hostname)) return undefined
  if (peer.shortId && hostname.startsWith(peer.shortId) && isIdentifierLike(hostname)) return undefined
  return hostname
}

export function rowAriaLabel(peer: { shortId?: string; id: string; hostname: string }, active: boolean): string {
  const primaryId = peer.shortId ?? peer.id
  const hostname = displayHostname(peer)
  const nodeLabel = hostname
    ? `${hostname} node, peer ID ${peer.id}`
    : primaryId === peer.id
      ? `peer ${peer.id}`
      : `peer ${primaryId}, peer ID ${peer.id}`
  return `View ${nodeLabel}${active ? ' (selected)' : ''}`
}

export function nextSortState(current: SortState, key: SortKey): SortState {
  if (current.key !== key) return { key, direction: 'asc' }
  return { key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
}

export function filterOptionLabel(value: string): string {
  return value === '—' ? 'None' : value
}

export function clampPage(page: number, pageCount: number) {
  return Math.min(Math.max(page, 0), Math.max(pageCount - 1, 0))
}
