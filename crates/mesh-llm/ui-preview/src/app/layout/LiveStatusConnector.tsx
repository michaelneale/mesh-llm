import { useDataMode } from '@/lib/data-mode'
import { useStatusStream } from '@/features/network/api/use-status-stream'

export function LiveStatusConnector() {
  const { mode } = useDataMode()
  const liveMode = mode === 'live'

  useStatusStream({ enabled: liveMode && typeof EventSource !== 'undefined' })

  return null
}
