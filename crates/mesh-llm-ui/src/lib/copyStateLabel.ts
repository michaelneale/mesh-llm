import type { ClipboardCopyState } from '@/lib/useClipboardCopy'

export function copyStateLabel(copyState: ClipboardCopyState | undefined, suffix?: string): string {
  const s = suffix ? ` ${suffix}` : ''
  if (copyState === 'copied') return `Copied${s}`
  if (copyState === 'failed') return `Copy${s} failed`
  return `Copy${s}`
}
