import { HeadContent, Outlet, useRouter, useRouterState } from '@tanstack/react-router'
import { useState } from 'react'
import { LiveStatusConnector } from '@/app/layout/LiveStatusConnector'
import { resolveHarnessTopNavData, resolveLiveTopNavData } from '@/app/layout/shell-adapter'
import { ChatSessionProvider } from '@/features/chat/api/chat-session'
import { Footer } from '@/features/shell/components/Footer'
import { TopNav } from '@/features/shell/components/TopNav'
import { PreferencesPanel } from '@/features/shell/components/PreferencesPanel'
import {
  isConfigurationTabId,
  type ConfigurationTabId
} from '@/features/configuration/components/configuration-tab-ids'
import { DEFAULT_DEVELOPER_PLAYGROUND_TAB } from '@/features/developer/playground/developer-playground-tabs'
import { useStatusQuery } from '@/features/network/api/use-status-query'
import { useUIPreferences } from '@/features/shell/hooks/useUiPreferences'
import { SHELL_HARNESS } from '@/features/app-tabs/data'
import { env, hrefWithBasePath, stripBasePath } from '@/lib/env'
import { useDataMode } from '@/lib/data-mode'
import { useBooleanFeatureFlag } from '@/lib/feature-flags'
import type { ShellHarnessData, AppTab } from '@/features/app-tabs/types'

function pathToTab(pathname: string): AppTab | null {
  if (pathname.startsWith('/chat')) return 'chat'
  if (pathname.startsWith('/configuration')) return 'configuration'
  if (import.meta.env.DEV && pathname.startsWith('/__playground')) return null
  return 'network'
}

function tabToPath(tab: Exclude<AppTab, 'configuration'>): '/' | '/chat' {
  if (tab === 'chat') return '/chat'
  return '/'
}

function pathToConfigurationTab(pathname: string): ConfigurationTabId | null {
  const [, section, configurationTab] = pathname.split('/')
  if (section !== 'configuration' || !isConfigurationTabId(configurationTab)) return null
  return configurationTab
}

function configurationTabHref(pathname: string) {
  return `/configuration/${pathToConfigurationTab(pathname) ?? 'defaults'}`
}

function tabHrefsForPath(pathname: string) {
  return {
    network: hrefWithBasePath('/'),
    chat: hrefWithBasePath('/chat'),
    configuration: hrefWithBasePath(configurationTabHref(pathname))
  }
}

type RootLayoutProps = { data?: ShellHarnessData }

function resolveApiTargetLiveness(statusQuery: ReturnType<typeof useStatusQuery>, liveMode: boolean) {
  if (!liveMode) return undefined
  if (statusQuery.isError) return 'unavailable'
  if (statusQuery.data) return 'live'
  return 'checking'
}

export function RootLayout({ data = SHELL_HARNESS }: RootLayoutProps = {}) {
  const router = useRouter()
  const routerPathname = useRouterState({ select: (state) => state.location.pathname })
  const pathname = stripBasePath(routerPathname)
  const { mode } = useDataMode()
  const liveMode = mode === 'live'
  const statusQuery = useStatusQuery({ enabled: liveMode })
  const { theme, accent, density, panelStyle, setTheme, setAccent, setDensity, setPanelStyle } = useUIPreferences()
  const newConfigurationPageEnabled = useBooleanFeatureFlag('global/newConfigurationPage')
  const activeTab = pathToTab(pathname)
  const [preferencesOpen, setPreferencesOpen] = useState(false)
  const topNavData = liveMode ? resolveLiveTopNavData(statusQuery.data) : resolveHarnessTopNavData(data)
  const apiTargetLiveness = resolveApiTargetLiveness(statusQuery, liveMode)
  const showDevelopmentNavControls = import.meta.env.DEV

  return (
    <>
      <HeadContent />
      <LiveStatusConnector />
      <div className="min-h-screen">
        <TopNav
          enabledTabs={{ configuration: newConfigurationPageEnabled }}
          tab={activeTab === 'configuration' && !newConfigurationPageEnabled ? null : activeTab}
          tabHrefs={tabHrefsForPath(pathname)}
          onTabChange={(tab) => {
            if (tab === 'configuration' && !newConfigurationPageEnabled) return
            if (tab === 'configuration') {
              void router.navigate({
                to: '/configuration/$configurationTab',
                params: { configurationTab: pathToConfigurationTab(pathname) ?? 'defaults' }
              })
              return
            }

            void router.navigate({ to: tabToPath(tab) })
          }}
          apiUrl={topNavData.apiUrl}
          apiTargetLiveness={apiTargetLiveness}
          version={env.appVersion}
          theme={theme}
          onThemeChange={setTheme}
          onTogglePreferences={() => {
            if (showDevelopmentNavControls) setPreferencesOpen((value) => !value)
          }}
          brand={data.brand}
          apiAccessLinks={topNavData.topNavApiAccessLinks}
          joinCommands={topNavData.topNavJoinCommands}
          joinLinks={topNavData.topNavJoinLinks}
          showDeveloperPlayground={showDevelopmentNavControls}
          onOpenDeveloperPlayground={
            showDevelopmentNavControls
              ? () => {
                  void router.navigate({ to: '/__playground', search: { tab: DEFAULT_DEVELOPER_PLAYGROUND_TAB } })
                }
              : undefined
          }
          onOpenIdentity={() => setPreferencesOpen(true)}
        />
        {showDevelopmentNavControls ? (
          <PreferencesPanel
            open={preferencesOpen}
            theme={theme}
            accent={accent}
            density={density}
            panelStyle={panelStyle}
            onThemeChange={setTheme}
            onAccentChange={setAccent}
            onDensityChange={setDensity}
            onPanelStyleChange={setPanelStyle}
            onClose={() => setPreferencesOpen(false)}
          />
        ) : null}
        <ChatSessionProvider>
          <main className="density-shell mx-auto px-[var(--shell-pad-x)] pb-[var(--shell-pad-bottom)] pt-[var(--shell-pad-top)]">
            <Outlet />
          </main>
        </ChatSessionProvider>
        <Footer
          version={env.appVersion}
          productName={data.productName}
          links={data.footerLinks}
          trailingLink={data.footerTrailingLink}
        />
      </div>
    </>
  )
}
