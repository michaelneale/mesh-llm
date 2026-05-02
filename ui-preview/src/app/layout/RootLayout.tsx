import { HeadContent, Outlet, useRouter, useRouterState } from '@tanstack/react-router'
import { useState } from 'react'
import { Footer } from '@/features/shell/components/Footer'
import { TopNav } from '@/features/shell/components/TopNav'
import { PreferencesPanel } from '@/features/shell/components/PreferencesPanel'
import { isConfigurationTabId, type ConfigurationTabId } from '@/features/configuration/components/configuration-tab-ids'
import { DEFAULT_DEVELOPER_PLAYGROUND_TAB } from '@/features/developer/playground/developer-playground-tabs'
import { useUIPreferences } from '@/features/shell/hooks/useUiPreferences'
import { SHELL_HARNESS } from '@/features/app-tabs/data'
import { env, hrefWithBasePath, stripBasePath } from '@/lib/env'
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
    configuration: hrefWithBasePath(configurationTabHref(pathname)),
  }
}

type RootLayoutProps = { data?: ShellHarnessData }

export function RootLayout({ data = SHELL_HARNESS }: RootLayoutProps = {}) {
  const router = useRouter()
  const routerPathname = useRouterState({ select: (state) => state.location.pathname })
  const pathname = stripBasePath(routerPathname)
  const { theme, accent, density, panelStyle, setTheme, setAccent, setDensity, setPanelStyle } = useUIPreferences()
  const newConfigurationPageEnabled = useBooleanFeatureFlag('global/newConfigurationPage')
  const activeTab = pathToTab(pathname)
  const [preferencesOpen, setPreferencesOpen] = useState(false)

  return (
    <>
      <HeadContent />
      <div className="min-h-screen">
        <TopNav
          enabledTabs={{ configuration: newConfigurationPageEnabled }}
          tab={activeTab === 'configuration' && !newConfigurationPageEnabled ? null : activeTab}
          tabHrefs={tabHrefsForPath(pathname)}
          onTabChange={(tab) => {
            if (tab === 'configuration' && !newConfigurationPageEnabled) return
            if (tab === 'configuration') {
              void router.navigate({ to: '/configuration/$configurationTab', params: { configurationTab: pathToConfigurationTab(pathname) ?? 'defaults' } })
              return
            }

            void router.navigate({ to: tabToPath(tab) })
          }}
          apiUrl={env.apiUrl}
          version={env.appVersion}
          theme={theme}
          onThemeChange={setTheme}
          onTogglePreferences={() => setPreferencesOpen((value) => !value)}
          brand={data.brand}
          apiAccessLinks={data.topNavApiAccessLinks}
          joinCommands={data.topNavJoinCommands}
          joinLinks={data.topNavJoinLinks}
          showDeveloperPlayground={import.meta.env.DEV}
          onOpenDeveloperPlayground={import.meta.env.DEV
            ? () => {
                void router.navigate({ to: '/__playground', search: { tab: DEFAULT_DEVELOPER_PLAYGROUND_TAB } })
              }
            : undefined}
          onOpenIdentity={() => setPreferencesOpen(true)}
        />
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
        <main className="density-shell mx-auto px-[var(--shell-pad-x)] pb-[var(--shell-pad-bottom)] pt-[var(--shell-pad-top)]">
          <Outlet />
        </main>
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
