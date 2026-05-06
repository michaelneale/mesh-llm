import { useState } from 'react'
import { Network } from 'lucide-react'
import { ModelSelect } from '@/features/chat/components/ModelSelect'
import { CommandBarModal } from '@/features/app-shell/components/command-bar/CommandBarModal'
import { CommandBarProvider } from '@/features/app-shell/components/command-bar/CommandBarProvider'
import { useCommandBar } from '@/features/app-shell/components/command-bar/useCommandBar'
import { ConnectBlock } from '@/features/network/components/ConnectBlock'
import { Footer } from '@/features/shell/components/Footer'
import { PreferencesPanel } from '@/features/shell/components/PreferencesPanel'
import { TopNav } from '@/features/shell/components/TopNav'
import { DASHBOARD_HARNESS } from '@/features/app-tabs/data'
import type { Accent, Density, PanelStyle, Theme, AppTab } from '@/features/app-tabs/types'
import { env, hrefWithBasePath } from '@/lib/env'
import { PlaygroundPanel, SidebarTabs, TextAreaField, TextField } from '../primitives'
import type { DeveloperPlaygroundState } from '../useDeveloperPlaygroundState'

type PreviewCommand = { id: string; label: string; summary: string }

const commandBarModes = [
  {
    id: 'routes',
    label: 'Routes',
    leadingIcon: Network,
    source: [
      { id: 'network', label: 'Network workspace', summary: 'Inspect peers, models, and mesh routing.' },
      { id: 'chat', label: 'Chat workspace', summary: 'Draft prompts against harness threads.' },
      { id: 'configuration', label: 'Configuration workspace', summary: 'Tune placements and local model assignments.' }
    ] satisfies PreviewCommand[],
    getItemKey: (item: PreviewCommand) => item.id,
    getSearchText: (item: PreviewCommand) => `${item.label} ${item.summary}`,
    ResultItem: ({ item, selected }: { item: PreviewCommand; selected: boolean }) => (
      <div className={selected ? 'bg-panel-strong px-3 py-2' : 'px-3 py-2'}>
        <div className="text-[length:var(--density-type-control)] font-medium text-foreground">{item.label}</div>
        <div className="mt-0.5 text-[length:var(--density-type-caption)] text-fg-faint">{item.summary}</div>
      </div>
    ),
    onSelect: () => true
  }
]

function CommandBarPreview() {
  const { openCommandBar } = useCommandBar()

  return (
    <PlaygroundPanel
      title="Command bar"
      description="Open the production command search shell with a small harness command set."
      actions={
        <button
          className="ui-control-primary inline-flex items-center rounded-[var(--radius)] px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
          onClick={() => openCommandBar('routes')}
          type="button"
        >
          Open command bar
        </button>
      }
    >
      <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5 text-[length:var(--density-type-caption-lg)] text-fg-dim">
        Preview modes: routes. Shortcut guidance and keyboard result navigation are owned by the real command bar
        implementation.
      </div>
      <CommandBarModal<PreviewCommand>
        behavior="distinct"
        defaultModeId="routes"
        description="Search preview commands."
        modes={commandBarModes}
        title="Playground command bar"
      />
    </PlaygroundPanel>
  )
}

export function ShellControlsArea({ state }: { state: DeveloperPlaygroundState }) {
  const [tab, setTab] = useState<AppTab>('network')
  const [theme, setTheme] = useState<Theme>('auto')
  const [accent, setAccent] = useState<Accent>('cyan')
  const [density, setDensity] = useState<Density>('compact')
  const [panelStyle, setPanelStyle] = useState<PanelStyle>('solid')
  const [preferencesOpen, setPreferencesOpen] = useState(false)

  return (
    <SidebarTabs
      ariaLabel="Shell control previews"
      defaultValue="connect"
      tabs={[
        {
          value: 'connect',
          label: 'Connect block',
          content: (
            <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
              <ConnectBlock
                installHref={DASHBOARD_HARNESS.connect.installHref}
                apiStatus={DASHBOARD_HARNESS.connect.apiStatus}
                apiUrl={env.apiUrl}
                description={state.connectDescription}
                runCommand={state.connectRunCommand}
                copyState={state.connectCopyState}
                onCopy={() => {
                  void state.copyConnectText(state.connectRunCommand)
                }}
              />
              <PlaygroundPanel
                title="Editable shell copy"
                description="Adjust compact command text and description without drifting from the restrained console tone."
              >
                <div className="space-y-3">
                  <TextField
                    label="Connect description"
                    value={state.connectDescription}
                    onChange={state.setConnectDescription}
                  />
                  <TextAreaField
                    label="Run command"
                    rows={4}
                    value={state.connectRunCommand}
                    onChange={state.setConnectRunCommand}
                  />
                </div>
              </PlaygroundPanel>
            </div>
          )
        },
        {
          value: 'chrome',
          label: 'Shell chrome',
          content: (
            <div className="space-y-4">
              <PlaygroundPanel
                title="Top navigation"
                description="Exercise primary tabs, API copy affordances, developer playground entry, theme selection, and preferences trigger in one shell strip."
              >
                <div className="relative overflow-hidden rounded-[var(--radius-lg)] border border-border bg-background">
                  <TopNav
                    apiUrl={env.apiUrl}
                    onOpenDeveloperPlayground={() => undefined}
                    onTabChange={setTab}
                    onTogglePreferences={() => setPreferencesOpen((open) => !open)}
                    onThemeChange={setTheme}
                    showDeveloperPlayground={true}
                    tab={tab}
                    tabHrefs={{
                      network: hrefWithBasePath('/'),
                      chat: hrefWithBasePath('/chat'),
                      configuration: hrefWithBasePath('/configuration/defaults')
                    }}
                    theme={theme}
                    version="v0.1.0-preview"
                  />
                  <div className="px-4 py-3 text-[length:var(--density-type-caption-lg)] text-fg-dim">
                    Active tab: <span className="font-mono text-foreground">{tab}</span> · theme preview:{' '}
                    <span className="font-mono text-foreground">{theme}</span>
                  </div>
                </div>
              </PlaygroundPanel>

              <PlaygroundPanel
                title="Preferences panel"
                description="Keep theme, accent, density, panel style, and data source controls visible without booting the full app shell."
                actions={
                  <button
                    className="ui-control inline-flex items-center rounded-[var(--radius)] border px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
                    onClick={() => setPreferencesOpen((open) => !open)}
                    type="button"
                  >
                    Toggle preferences
                  </button>
                }
              >
                <div className="relative min-h-[420px] overflow-hidden rounded-[var(--radius-lg)] border border-border bg-background">
                  <PreferencesPanel
                    accent={accent}
                    density={density}
                    onAccentChange={setAccent}
                    onClose={() => setPreferencesOpen(false)}
                    onDensityChange={setDensity}
                    onPanelStyleChange={setPanelStyle}
                    onThemeChange={setTheme}
                    open={preferencesOpen}
                    panelStyle={panelStyle}
                    theme={theme}
                  />
                  <div className="grid gap-2 p-4 text-[length:var(--density-type-caption-lg)] text-fg-dim sm:grid-cols-2">
                    <div>
                      Accent: <span className="font-mono text-foreground">{accent}</span>
                    </div>
                    <div>
                      Density: <span className="font-mono text-foreground">{density}</span>
                    </div>
                    <div>
                      Panels: <span className="font-mono text-foreground">{panelStyle}</span>
                    </div>
                    <div>
                      Theme: <span className="font-mono text-foreground">{theme}</span>
                    </div>
                  </div>
                </div>
              </PlaygroundPanel>

              <Footer
                links={[
                  { href: hrefWithBasePath('/configuration/defaults'), label: 'Configuration' },
                  { href: hrefWithBasePath('/chat'), label: 'Chat' }
                ]}
                productName="mesh-llm/ui-preview"
                trailingLink={{ href: 'https://github.com/anarchai/mesh-llm', label: 'GitHub' }}
                version="0.1.0"
              />

              <CommandBarProvider>
                <CommandBarPreview />
              </CommandBarProvider>
            </div>
          )
        },
        {
          value: 'selectors',
          label: 'Selectors',
          content: (
            <PlaygroundPanel
              title="Compact picker surfaces"
              description="Preview the shared picker styling without tying the controls to any one product page."
            >
              <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
                <div className="space-y-3">
                  <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                    <div className="type-label text-fg-faint">Selected model</div>
                    <div className="mt-2">
                      <ModelSelect
                        options={state.chatOptions}
                        value={state.shellSelectedModel}
                        onChange={state.setShellSelectedModel}
                      />
                    </div>
                  </div>
                  <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                    <div className="type-label text-fg-faint">Preview state</div>
                    <div className="mt-2 font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
                      {state.shellSelectedModel}
                    </div>
                  </div>
                </div>
                <div className="rounded-[var(--radius-lg)] border border-border bg-background px-4 py-4">
                  <div className="type-panel-title text-foreground">Shell control notes</div>
                  <p className="mt-2 max-w-[62ch] text-[length:var(--density-type-caption-lg)] text-fg-dim">
                    Keep shell actions short, direct, and easy to scan. Selection state should be obvious from border
                    and tint, not from decoration.
                  </p>
                </div>
              </div>
            </PlaygroundPanel>
          )
        }
      ]}
    />
  )
}
