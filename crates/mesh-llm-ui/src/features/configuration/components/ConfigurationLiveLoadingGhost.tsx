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
        <div className="grid min-w-0 gap-[14px] lg:grid-cols-[220px_minmax(0,1fr)]">
          <ConfigurationRailLoadingGhost />
          <section className="space-y-[14px]">
            <ConfigurationDeploymentLoadingGhost />
            <ConfigurationSettingsLoadingGhost />
          </section>
        </div>
      </ConfigurationLayout>
    </LiveLoadingGhostRoot>
  )
}
