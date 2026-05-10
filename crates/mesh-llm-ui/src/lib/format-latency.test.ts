import { describe, expect, it } from 'vitest'
import { LatencySource } from '@/lib/api/types'
import { formatPeerLatency, formatPeerLatencySummary, peerLatencyHint, peerLatencyState } from '@/lib/format-latency'

describe('formatPeerLatency', () => {
  it('returns question mark for null latency', () => {
    expect(formatPeerLatency(null)).toBe('?')
  })

  it('returns <1ms for sub-millisecond values', () => {
    expect(formatPeerLatency(0.4)).toBe('<1ms')
    expect(formatPeerLatency(0.9)).toBe('<1ms')
  })

  it('rounds to nearest millisecond for normal values', () => {
    expect(formatPeerLatency(1.2)).toBe('1ms')
    expect(formatPeerLatency(3.4)).toBe('3ms')
    expect(formatPeerLatency(7.8)).toBe('8ms')
  })

  it('clamps negative values to zero', () => {
    expect(formatPeerLatency(-5)).toBe('0ms')
  })
})

describe('formatPeerLatencySummary', () => {
  it('returns question mark when latency is null', () => {
    expect(
      formatPeerLatencySummary({ latencyMs: null, source: LatencySource.UNSPECIFIED, ageMs: null, observerId: null })
    ).toBe('?')
  })

  it('renders base value for DIRECT source without suffix', () => {
    expect(
      formatPeerLatencySummary({ latencyMs: 7.2, source: LatencySource.DIRECT, ageMs: 100, observerId: 'observer-1' })
    ).toBe('7ms')
  })

  it('appends tilde for ESTIMATED source', () => {
    expect(
      formatPeerLatencySummary({ latencyMs: 3.4, source: LatencySource.ESTIMATED, ageMs: null, observerId: null })
    ).toBe('3ms~')
  })

  it('appends exclamation mark for UNKNOWN source', () => {
    expect(
      formatPeerLatencySummary({ latencyMs: 1.2, source: LatencySource.UNKNOWN, ageMs: 500, observerId: 'observer-2' })
    ).toBe('1ms!')
  })

  it('renders base value for UNSPECIFIED source', () => {
    expect(
      formatPeerLatencySummary({ latencyMs: 5.6, source: LatencySource.UNSPECIFIED, ageMs: null, observerId: null })
    ).toBe('6ms')
  })

  it('handles sub-millisecond values correctly with source suffix', () => {
    expect(
      formatPeerLatencySummary({ latencyMs: 0.4, source: LatencySource.ESTIMATED, ageMs: null, observerId: null })
    ).toBe('<1ms~')
  })
})

describe('peerLatencyHint', () => {
  it('returns unknown when latency is null', () => {
    expect(peerLatencyHint({ latencyMs: null, source: LatencySource.UNSPECIFIED, ageMs: null, observerId: null })).toBe(
      'unknown'
    )
  })

  it('returns direct for DIRECT source without age suffix when age is zero', () => {
    expect(peerLatencyHint({ latencyMs: 5, source: LatencySource.DIRECT, ageMs: 0, observerId: null })).toBe('direct')
  })

  it('returns direct with age suffix for DIRECT source when age is positive', () => {
    expect(peerLatencyHint({ latencyMs: 5, source: LatencySource.DIRECT, ageMs: 1200, observerId: 'observer-1' })).toBe(
      'direct (1s old)'
    )
  })

  it('returns estimated for ESTIMATED source', () => {
    expect(peerLatencyHint({ latencyMs: 5, source: LatencySource.ESTIMATED, ageMs: null, observerId: null })).toBe(
      'estimated'
    )
  })

  it('returns unknown for UNKNOWN source', () => {
    expect(peerLatencyHint({ latencyMs: 5, source: LatencySource.UNKNOWN, ageMs: null, observerId: null })).toBe(
      'unknown'
    )
  })

  it('returns unspecified for UNSPECIFIED source', () => {
    expect(peerLatencyHint({ latencyMs: 5, source: LatencySource.UNSPECIFIED, ageMs: null, observerId: null })).toBe(
      'unspecified'
    )
  })

  it('includes age suffix with rounding for ESTIMATED source', () => {
    expect(peerLatencyHint({ latencyMs: 5, source: LatencySource.ESTIMATED, ageMs: 2400, observerId: null })).toBe(
      'estimated (2s old)'
    )
  })
})

describe('peerLatencyState', () => {
  it('returns the passed state unchanged', () => {
    const input = { latencyMs: 7.2, source: LatencySource.DIRECT, ageMs: 100, observerId: 'observer-1' }
    expect(peerLatencyState(input)).toEqual(input)
  })

  it('handles null values correctly', () => {
    const input = { latencyMs: null, source: LatencySource.UNSPECIFIED, ageMs: null, observerId: null }
    expect(peerLatencyState(input)).toEqual(input)
  })
})
