import type { ReactNode } from 'react'
import * as NavigationMenu from '@radix-ui/react-navigation-menu'
import { Code2, ExternalLink, Link2, Moon, Settings, Sun } from 'lucide-react'
import { HeaderHoverCard } from '@/features/shell/components/HeaderHoverCard'
import { CopyInstructionRow } from '@/components/ui/CopyInstructionRow'
import type { LinkItem, TopNavJoinCommand, Theme, AppTab } from '@/features/app-tabs/types'
import { cn } from '@/lib/cn'
import { useDataMode, type DataMode } from '@/lib/data-mode'
import { useI18n } from '@/lib/i18n'

type TopNavProps = {
  tab: AppTab | null
  enabledTabs?: Partial<Record<AppTab, boolean>>
  tabHrefs?: Partial<Record<AppTab, string>>
  onTabChange: (tab: AppTab) => void
  apiUrl: string
  version: string
  theme: Theme
  onToggleTheme: () => void
  showDeveloperPlayground?: boolean
  onOpenDeveloperPlayground?: () => void
  onOpenIdentity?: () => void
  onTogglePreferences: () => void
  renderLogo?: () => ReactNode
  brand?: { primary: string; accent: string }
  apiAccessLinks?: LinkItem[]
  joinCommands?: TopNavJoinCommand[]
  joinLinks?: LinkItem[]
}

const tabs: { value: AppTab; href: string; labelKey: 'tabs.network' | 'tabs.chat' | 'tabs.configuration' }[] = [
  { value: 'network', href: '/', labelKey: 'tabs.network' },
  { value: 'chat', href: '/chat', labelKey: 'tabs.chat' },
  { value: 'configuration', href: '/configuration', labelKey: 'tabs.configuration' },
]

const supportsNavigationMenu = typeof globalThis.ResizeObserver === 'function'

function MeshLogo() {
  return (
    <img src="/meshllm-apple-touch-icon.png" width={32} height={32} alt="" aria-hidden="true" className="size-8 shrink-0" />
  )
}

function BrandCluster({ version, renderLogo, brand }: { version: string; renderLogo?: () => ReactNode; brand?: { primary: string; accent: string } }) {
  const primary = brand?.primary ?? 'mesh'
  const accent = brand?.accent ?? 'llm'
  return (
    <div className="mr-[var(--brand-margin-end)] flex items-center gap-[var(--brand-gap)]">
      {renderLogo ? renderLogo() : <MeshLogo />}
      <span style={{ fontSize: 'var(--brand-font-size)', fontWeight: 700, letterSpacing: -0.3 }}>
        {primary}<span className="text-accent">{accent}</span>
      </span>
      <span className="inline-flex items-center rounded-full border border-border px-[var(--version-pad-x)] py-px font-sans text-[length:var(--version-font-size)] font-medium text-fg-faint" style={{ letterSpacing: 0.02, lineHeight: 1.4 }}>
        {version}
      </span>
    </div>
  )
}

function PrimaryTabs({ enabledTabs, tab, tabHrefs, onTabChange }: { enabledTabs?: Partial<Record<AppTab, boolean>>; tab: AppTab | null; tabHrefs?: Partial<Record<AppTab, string>>; onTabChange: (tab: AppTab) => void }) {
  const { t } = useI18n()
  const visibleTabs = tabs.filter((item) => enabledTabs?.[item.value] !== false)

  if (!supportsNavigationMenu) {
    return (
      <nav aria-label="Primary" className="flex items-center gap-[var(--nav-tab-gap)]">
        {visibleTabs.map((item) => {
          const active = tab === item.value
          const href = tabHrefs?.[item.value] ?? item.href

          return (
            <a
              key={item.value}
              href={href}
              className={cn(
                'rounded-[var(--radius)] border border-transparent px-[var(--nav-tab-pad-x)] py-[var(--nav-tab-pad-y)] text-[length:var(--nav-tab-font-size)] leading-[var(--nav-tab-line-height)] font-medium tracking-normal',
                active ? 'ui-control-primary' : 'ui-control-ghost',
              )}
              aria-current={active ? 'page' : undefined}
              onClick={(event) => {
                if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.altKey || event.ctrlKey || event.shiftKey) return
                event.preventDefault()
                onTabChange(item.value)
              }}
            >
              {t(item.labelKey)}
            </a>
          )
        })}
      </nav>
    )
  }

  return (
    <NavigationMenu.Root aria-label="Primary" className="flex items-center">
      <NavigationMenu.List className="m-0 flex list-none items-center gap-[var(--nav-tab-gap)] p-0">
        {visibleTabs.map((item) => {
          const active = tab === item.value
          const href = tabHrefs?.[item.value] ?? item.href

          return (
            <NavigationMenu.Item key={item.value}>
              <NavigationMenu.Link asChild active={active}>
                <a
                  href={href}
                  className={cn(
                    'rounded-[var(--radius)] border border-transparent px-[var(--nav-tab-pad-x)] py-[var(--nav-tab-pad-y)] text-[length:var(--nav-tab-font-size)] leading-[var(--nav-tab-line-height)] font-medium tracking-normal',
                    active ? 'ui-control-primary' : 'ui-control-ghost',
                  )}
                  aria-current={active ? 'page' : undefined}
                  onClick={(event) => {
                    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.altKey || event.ctrlKey || event.shiftKey) return
                    event.preventDefault()
                    onTabChange(item.value)
                  }}
                >
                  {t(item.labelKey)}
                </a>
              </NavigationMenu.Link>
            </NavigationMenu.Item>
          )
        })}
      </NavigationMenu.List>
    </NavigationMenu.Root>
  )
}

function HeaderLinks({ links }: { links: { href: string; label: string }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 pt-1 text-[length:var(--density-type-caption)] text-fg-faint">
      {links.map((link) => (
        <a key={link.href} href={link.href} className="ui-link inline-flex items-center gap-1">
          {link.label}
          <ExternalLink className="size-[11px]" aria-hidden="true" />
        </a>
      ))}
    </div>
  )
}

const DEFAULT_API_ACCESS_LINKS: LinkItem[] = [
  { href: 'https://docs.anarchai.org/', label: 'Docs' },
  { href: 'https://docs.anarchai.org/#install', label: 'Install' },
]

function ApiAccessContent({ apiUrl, dataMode, links }: { apiUrl: string; dataMode: DataMode; links?: LinkItem[] }) {
  const listModelsCommand = `curl ${apiUrl}/models`
  const harnessMode = dataMode === 'harness'

  return (
    <>
      {harnessMode ? (
        <CopyInstructionRow
          label="Active data source"
          value="test harness"
          hint={<span>App pages are using local fixture data. Switch Data source to Live API in Tweaks to fetch from the backend.</span>}
        />
      ) : null}
      <CopyInstructionRow
        label="API endpoint"
        value={apiUrl}
        hint={<span>{harnessMode ? 'Available for copying, but not currently driving the UI.' : 'Point OpenAI-compatible tools at this local mesh target.'}</span>}
      />
      <CopyInstructionRow
        label="Export base URL"
        value={`OPENAI_BASE_URL=${apiUrl}`}
        copyValue={`export OPENAI_BASE_URL=${apiUrl}`}
        prefix="$"
      />
      <CopyInstructionRow
        label="List models command"
        value={listModelsCommand}
        prefix="$"
        hint={<span>Quick check for a live target before wiring a client.</span>}
      />
      <HeaderLinks links={links ?? DEFAULT_API_ACCESS_LINKS} />
    </>
  )
}

const DEFAULT_JOIN_COMMANDS: TopNavJoinCommand[] = [
  { label: 'Invite token', value: '<mesh-invite-token>', hint: 'Paste your issued token into any join command below.', noWrapValue: true },
  { label: 'Auto join and serve command', value: 'mesh-llm --auto --join <mesh-invite-token>', prefix: '$', hint: 'Matches the Connect panel flow: join, select a model, and serve the API.' },
  { label: 'Client-only join command', value: 'mesh-llm client --join <mesh-invite-token>', prefix: '$' },
  { label: 'Blackboard skill command', value: 'mesh-llm blackboard install-skill', prefix: '$' },
]

const DEFAULT_JOIN_LINKS: LinkItem[] = [
  { href: 'https://docs.anarchai.org/', label: 'Setup' },
  { href: 'https://docs.anarchai.org/#install', label: 'Install' },
  { href: 'https://docs.anarchai.org/#blackboard', label: 'Blackboard' },
]

function JoinInviteContent({ commands, links }: { commands?: TopNavJoinCommand[]; links?: LinkItem[] }) {
  const resolvedCommands = commands ?? DEFAULT_JOIN_COMMANDS
  const resolvedLinks = links ?? DEFAULT_JOIN_LINKS
  return (
    <>
      {resolvedCommands.map((cmd) => (
        <CopyInstructionRow
          key={cmd.label}
          label={cmd.label}
          value={cmd.value}
          prefix={cmd.prefix}
          hint={cmd.hint ? <span>{cmd.hint}</span> : undefined}
          noWrapValue={cmd.noWrapValue}
          copyValue={cmd.copyValue}
        />
      ))}
      <HeaderLinks links={resolvedLinks} />
    </>
  )
}

function ApiStatusChip({ apiUrl, apiAccessLinks }: { apiUrl: string; apiAccessLinks?: LinkItem[] }) {
  const { mode } = useDataMode()
  const harnessMode = mode === 'harness'
  const targetLabel = harnessMode ? 'test harness' : apiUrl

  return (
    <HeaderHoverCard
      trigger={(triggerProps) => (
        <button
          {...triggerProps}
          aria-label="API target instructions"
          className="ui-control hidden h-[var(--nav-action-size)] min-w-0 items-center gap-[var(--nav-chip-gap)] rounded-[var(--radius)] border px-[var(--nav-chip-pad-x)] py-[var(--nav-chip-pad-y)] text-fg-dim md:flex"
          type="button"
        >
          <span className={cn('size-[var(--nav-chip-dot-size)] shrink-0 rounded-full', harnessMode ? 'bg-warn' : 'bg-fg-faint')} aria-hidden="true" />
          <span className="text-[length:var(--nav-chip-font-size)] font-medium uppercase tracking-[0.08em] text-fg-faint">API target</span>
          <span className={cn('max-w-64 truncate font-mono text-[length:var(--nav-chip-font-size)] font-semibold tracking-normal', harnessMode ? 'text-warn' : 'text-foreground')}>{targetLabel}</span>
        </button>
      )}
      eyebrow={harnessMode ? 'Fixture data' : 'Local endpoint'}
      title="API access"
      description={harnessMode ? 'The app is not using live backend data right now. The configured endpoint is still available below.' : 'Copy the current local target and a couple of ready commands for OpenAI-compatible clients.'}
      triggerMode="click"
    >
      <ApiAccessContent apiUrl={apiUrl} dataMode={mode} links={apiAccessLinks} />
    </HeaderHoverCard>
  )
}

function UtilityActions({
  apiUrl,
  theme,
  onToggleTheme,
  onOpenDeveloperPlayground,
  showDeveloperPlayground,
  onTogglePreferences,
  apiAccessLinks,
  joinCommands,
  joinLinks,
}: Pick<TopNavProps, 'apiUrl' | 'theme' | 'onToggleTheme' | 'showDeveloperPlayground' | 'onOpenDeveloperPlayground' | 'onTogglePreferences' | 'apiAccessLinks' | 'joinCommands' | 'joinLinks'>) {
  const { mode } = useDataMode()
  const iconBtn = 'ui-control inline-flex size-[var(--nav-action-size)] items-center justify-center rounded-[var(--radius)] border'
  const utilityBtn = 'ui-control inline-flex h-[var(--nav-action-size)] items-center gap-1.5 rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-caption)] font-medium'

  return (
    <div className="flex items-center gap-[var(--nav-action-gap)]">
      <div className="md:hidden">
        <HeaderHoverCard
          trigger={(triggerProps) => (
            <button {...triggerProps} aria-label="Open API instructions" className={utilityBtn} type="button">
              API
            </button>
          )}
          align="end"
            eyebrow={mode === 'harness' ? 'Fixture data' : 'Local endpoint'}
            title={mode === 'harness' ? 'Test harness active' : 'API access'}
            description={mode === 'harness' ? 'The compact menu shows the configured endpoint, but the UI is currently using harness data.' : 'The compact menu exposes the same endpoint details on smaller screens.'}
            triggerMode="click"
          >
            <ApiAccessContent apiUrl={apiUrl} dataMode={mode} links={apiAccessLinks} />
          </HeaderHoverCard>
        </div>
      <HeaderHoverCard
        trigger={(triggerProps) => (
          <button {...triggerProps} aria-label="Mesh join and invite instructions" className={utilityBtn} type="button">
            <Link2 className="size-[11px]" aria-hidden="true" />
            Join
          </button>
        )}
        align="end"
        eyebrow="Mesh access"
        title="Join or invite"
        description="Keep the common join flows close when you need to add a node, client, or agent quickly."
        triggerMode="click"
      >
        <JoinInviteContent commands={joinCommands} links={joinLinks} />
      </HeaderHoverCard>
      {showDeveloperPlayground && onOpenDeveloperPlayground ? (
        <button className={iconBtn} onClick={onOpenDeveloperPlayground} type="button" aria-label="Open developer playground">
          <Code2 className="size-[var(--nav-icon-size)]" />
        </button>
      ) : null}
      <button className={iconBtn} onClick={onToggleTheme} type="button" aria-label="Toggle theme">
        {theme === 'dark' ? <Sun className="size-[var(--nav-icon-size)]" /> : <Moon className="size-[var(--nav-icon-size)]" />}
      </button>
      <button className={iconBtn} onClick={onTogglePreferences} type="button" aria-label="Open interface preferences">
        <Settings className="size-[var(--nav-icon-size)]" />
      </button>
    </div>
  )
}

export function TopNav(props: TopNavProps) {
  return (
    <header className="surface-chrome sticky top-0 z-30 flex items-center gap-[var(--topnav-gap)] border-b border-border-soft px-[var(--topnav-pad-x)] py-[var(--topnav-pad-y)]">
      <BrandCluster version={props.version} renderLogo={props.renderLogo} brand={props.brand} />
      <PrimaryTabs enabledTabs={props.enabledTabs} tab={props.tab} tabHrefs={props.tabHrefs} onTabChange={props.onTabChange} />
      <div className="flex-1" />
      <ApiStatusChip apiUrl={props.apiUrl} apiAccessLinks={props.apiAccessLinks} />
      <UtilityActions {...props} />
    </header>
  )
}
