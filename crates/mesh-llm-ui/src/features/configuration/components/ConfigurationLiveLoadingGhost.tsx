import { LiveLoadingGhostRoot } from '@/components/ui/LiveLoadingGhostRoot'
import {
  ConfigurationDeploymentLoadingGhost,
  ConfigurationHeaderLoadingGhost,
  ConfigurationRailLoadingGhost,
  ConfigurationSettingsLoadingGhost
} from '@/features/configuration/components/ConfigurationLoadingGhostSections'
import { ConfigurationLayout } from '@/features/configuration/layouts/ConfigurationLayout'

export function ConfigurationLiveLoadingGhost() {
  return (
    <LiveLoadingGhostRoot>
      <ConfigurationLayout header={<ConfigurationHeaderLoadingGhost />}>
        <div className="grid gap-3.5 px-5" style={{ gridTemplateColumns: '220px minmax(0, 1fr)' }}>
          <ConfigurationRailLoadingGhost />
          <section className="space-y-3">
            <ConfigurationDeploymentLoadingGhost />
            <ConfigurationSettingsLoadingGhost />
          </section>
        </div>
      </ConfigurationLayout>
    </LiveLoadingGhostRoot>
  )
}
