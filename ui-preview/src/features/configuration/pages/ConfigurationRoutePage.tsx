import { Navigate, useNavigate, useParams } from '@tanstack/react-router'
import { useCallback } from 'react'
import { ConfigurationPageContent } from '@/features/configuration/pages/ConfigurationPage'
import {
  isConfigurationTabId,
  type ConfigurationTabId
} from '@/features/configuration/components/configuration-tab-ids'
import { useBooleanFeatureFlag } from '@/lib/feature-flags'

function configurationTabFromParams(params: object) {
  if (!('configurationTab' in params) || typeof params.configurationTab !== 'string') return undefined
  return params.configurationTab
}

export function ConfigurationRoutePage() {
  const navigate = useNavigate()
  const activeTab = configurationTabFromParams(useParams({ strict: false }))
  const newConfigurationPageEnabled = useBooleanFeatureFlag('global/newConfigurationPage')
  const signingAttestationEnabled = useBooleanFeatureFlag('configuration/signingAttestation')
  const integrationsEnabled = useBooleanFeatureFlag('configuration/integrations')

  const navigateToTab = useCallback(
    (configurationTab: ConfigurationTabId) => {
      void navigate({ to: '/configuration/$configurationTab', params: { configurationTab }, replace: true })
    },
    [navigate]
  )

  if (!newConfigurationPageEnabled) {
    return (
      <section className="panel-shell mx-auto max-w-3xl rounded-[var(--radius-lg)] border border-border bg-panel p-6">
        <div className="type-label text-fg-faint">Feature flag disabled</div>
        <h1 className="type-display mt-1 text-foreground">Configuration is gated</h1>
        <p className="type-body mt-2 max-w-[68ch] text-fg-dim">
          Enable <span className="font-mono text-foreground">global/newConfigurationPage</span> in the developer
          playground to expose this app surface.
        </p>
      </section>
    )
  }

  if (activeTab === undefined || !isConfigurationTabId(activeTab)) {
    return <Navigate replace to="/configuration/$configurationTab" params={{ configurationTab: 'defaults' }} />
  }

  if (
    (activeTab === 'signing' && !signingAttestationEnabled) ||
    (activeTab === 'integrations' && !integrationsEnabled)
  ) {
    return <Navigate replace to="/configuration/$configurationTab" params={{ configurationTab: 'defaults' }} />
  }

  return <ConfigurationPageContent activeTab={activeTab} onTabChange={navigateToTab} />
}
