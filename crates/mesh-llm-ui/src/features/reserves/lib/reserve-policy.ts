import type { ReservePolicyTileData, ReserveWakePolicySettings } from '@/features/reserves/lib/reserve-types'

export const DEFAULT_RESERVE_WAKE_POLICY_SETTINGS: ReserveWakePolicySettings = {
  autoWakeEnabled: true,
  thresholdPercent: 75,
  sustainedSeconds: 30,
  providerOrder: ['LAN', 'Bare metal', 'Cloud'],
  idleMinutes: 8
}

export function reservePolicyTilesFromSettings(settings: ReserveWakePolicySettings): ReservePolicyTileData[] {
  return [
    {
      title: 'Auto-wake',
      value: settings.autoWakeEnabled ? 'Enabled' : 'Paused',
      status: settings.autoWakeEnabled ? 'Enabled' : 'Paused',
      explanation: settings.autoWakeEnabled
        ? `Wake reserves when mesh utilization > ${settings.thresholdPercent}% for ${formatDuration(settings.sustainedSeconds)}`
        : 'Keep reserve providers parked until an operator starts them manually.'
    },
    {
      title: 'Provider order',
      value: settings.providerOrder.join(' → '),
      explanation: 'Cheapest viable provider tried first'
    },
    {
      title: 'Sleep idle reserves',
      value: `after ${settings.idleMinutes} min idle`,
      explanation: 'Cloud nodes return to standby; LAN stays online'
    }
  ]
}

function formatDuration(seconds: number) {
  if (seconds < 60) return `${seconds}s`
  if (seconds % 60 === 0) return `${seconds / 60} min`
  return `${Math.round(seconds / 60)} min`
}
