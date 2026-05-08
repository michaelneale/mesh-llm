import type { StatusBadgeTone } from '@/components/ui/StatusBadge'
import type { ModelSummary } from '@/features/app-tabs/types'

export function modelStatusBadge(status: ModelSummary['status'] | undefined): { label: string; tone: StatusBadgeTone } {
  if (status === 'offline') return { label: 'Offline', tone: 'bad' }
  if (status === 'warming') return { label: 'Warming', tone: 'warn' }
  if (status === 'ready') return { label: 'Ready', tone: 'good' }
  if (status === 'warm') return { label: 'Warm', tone: 'good' }
  return { label: 'Unknown', tone: 'muted' }
}
