import { useState, type ReactNode } from 'react'
import * as NavigationMenu from '@radix-ui/react-navigation-menu'
import * as Select from '@radix-ui/react-select'
import {
  BotMessageSquare,
  Check,
  ChevronDown,
  Code2,
  ExternalLink,
  Menu,
  Monitor,
  Moon,
  Network,
  Settings,
  Share2,
  Sun,
  type LucideIcon
} from 'lucide-react'
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
  apiTargetLiveness?: ApiTargetLiveness
  version: string
  theme: Theme
  onThemeChange: (theme: Theme) => void
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

export type ApiTargetLiveness = 'checking' | 'live' | 'unavailable'

const tabs: { value: AppTab; href: string; labelKey: 'tabs.network' | 'tabs.chat' | 'tabs.configuration' }[] = [
  { value: 'network', href: '/', labelKey: 'tabs.network' },
  { value: 'chat', href: '/chat', labelKey: 'tabs.chat' },
  { value: 'configuration', href: '/configuration', labelKey: 'tabs.configuration' }
]

const themeOptions: { value: Theme; label: string; description: string; Icon: LucideIcon }[] = [
  { value: 'auto', label: 'Auto', description: 'Follow system', Icon: Monitor },
  { value: 'dark', label: 'Dark', description: 'Dim control room', Icon: Moon },
  { value: 'light', label: 'Light', description: 'Bright workspace', Icon: Sun }
]

const tabIcons: Record<AppTab, LucideIcon> = {
  network: Network,
  chat: BotMessageSquare,
  configuration: Settings
}

const supportsNavigationMenu = typeof globalThis.ResizeObserver === 'function'

function MeshLogo() {
  return (
    <img
      src="/meshllm-apple-touch-icon.png"
      width={32}
      height={32}
      alt=""
      aria-hidden="true"
      className="size-8 shrink-0"
    />
  )
}

function isTheme(value: string): value is Theme {
  return themeOptions.some((option) => option.value === value)
}

function BrandCluster({
  version,
  renderLogo,
  brand
}: {
  version: string
  renderLogo?: () => ReactNode
  brand?: { primary: string; accent: string }
}) {
  const primary = brand?.primary ?? 'mesh'
  const accent = brand?.accent ?? 'llm'
  return (
    <div className="mr-[var(--brand-margin-end)] flex min-w-0 shrink-0 items-center gap-[var(--brand-gap)]">
      {renderLogo ? renderLogo() : <MeshLogo />}
      <span style={{ fontSize: 'var(--brand-font-size)', fontWeight: 700, letterSpacing: -0.3 }}>
        {primary}
        <span className="text-accent">{accent}</span>
      </span>
      <span
        className="hidden items-center rounded-full border border-border px-[var(--version-pad-x)] py-px font-sans text-[length:var(--version-font-size)] font-medium text-fg-faint min-[420px]:inline-flex"
        style={{ letterSpacing: 0.02, lineHeight: 1.4 }}
      >
        {version}
      </span>
    </div>
  )
}

function PrimaryTabs({
  enabledTabs,
  tab,
  tabHrefs,
  onTabChange,
  className
}: {
  enabledTabs?: Partial<Record<AppTab, boolean>>
  tab: AppTab | null
  tabHrefs?: Partial<Record<AppTab, string>>
  onTabChange: (tab: AppTab) => void
  className?: string
}) {
  const { t } = useI18n()
  const visibleTabs = tabs.filter((item) => enabledTabs?.[item.value] !== false)
  const compactTabs = (
    <nav className="ml-2 flex items-center gap-1.5" aria-label="Primary compact navigation">
      {visibleTabs.map((item) => {
        const active = tab === item.value
        const Icon = tabIcons[item.value]
        const label = t(item.labelKey)

        return (
          <button
            key={item.value}
            aria-current={active ? 'page' : undefined}
            aria-label={label}
            className={cn(
              'ui-control-primary inline-flex size-[var(--nav-action-size)] items-center justify-center rounded-[var(--radius)] border',
              !active ? 'opacity-80' : ''
            )}
            onClick={() => onTabChange(item.value)}
            type="button"
          >
            <Icon className="size-[var(--nav-icon-size)]" aria-hidden="true" />
          </button>
        )
      })}
    </nav>
  )

  if (!supportsNavigationMenu) {
    return (
      <div className={cn('min-w-0', className)}>
        <div className="md:hidden">{compactTabs}</div>
        <nav aria-label="Primary" className="hidden min-w-0 flex-nowrap items-center gap-[var(--nav-tab-gap)] md:flex">
          {visibleTabs.map((item) => {
            const active = tab === item.value
            const href = tabHrefs?.[item.value] ?? item.href

            return (
              <a
                key={item.value}
                href={href}
                className={cn(
                  'whitespace-nowrap rounded-[var(--radius)] border border-transparent px-[var(--nav-tab-pad-x)] py-[var(--nav-tab-pad-y)] text-[length:var(--nav-tab-font-size)] leading-[var(--nav-tab-line-height)] font-medium tracking-normal',
                  active ? 'ui-control-primary' : 'ui-control-ghost'
                )}
                aria-current={active ? 'page' : undefined}
                onClick={(event) => {
                  if (
                    event.defaultPrevented ||
                    event.button !== 0 ||
                    event.metaKey ||
                    event.altKey ||
                    event.ctrlKey ||
                    event.shiftKey
                  )
                    return
                  event.preventDefault()
                  onTabChange(item.value)
                }}
              >
                {t(item.labelKey)}
              </a>
            )
          })}
        </nav>
      </div>
    )
  }

  return (
    <div className={cn('min-w-0', className)}>
      <div className="md:hidden">{compactTabs}</div>
      <NavigationMenu.Root aria-label="Primary" className="hidden min-w-0 items-center md:flex">
        <NavigationMenu.List className="m-0 flex min-w-0 flex-nowrap items-center gap-[var(--nav-tab-gap)] p-0">
          {visibleTabs.map((item) => {
            const active = tab === item.value
            const href = tabHrefs?.[item.value] ?? item.href

            return (
              <NavigationMenu.Item key={item.value}>
                <NavigationMenu.Link asChild active={active}>
                  <a
                    href={href}
                    className={cn(
                      'whitespace-nowrap rounded-[var(--radius)] border border-transparent px-[var(--nav-tab-pad-x)] py-[var(--nav-tab-pad-y)] text-[length:var(--nav-tab-font-size)] leading-[var(--nav-tab-line-height)] font-medium tracking-normal',
                      active ? 'ui-control-primary' : 'ui-control-ghost'
                    )}
                    aria-current={active ? 'page' : undefined}
                    onClick={(event) => {
                      if (
                        event.defaultPrevented ||
                        event.button !== 0 ||
                        event.metaKey ||
                        event.altKey ||
                        event.ctrlKey ||
                        event.shiftKey
                      )
                        return
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
    </div>
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
  { href: 'https://docs.meshllm.cloud/', label: 'Docs' },
  { href: 'https://docs.meshllm.cloud/#install', label: 'Install' }
]

function ApiAccessContent({ apiUrl, dataMode, links }: { apiUrl: string; dataMode: DataMode; links?: LinkItem[] }) {
  const listModelsCommand = `curl ${apiUrl}/models`
  const harnessMode = dataMode === 'harness'
  const showDevelopmentNavControls = import.meta.env.DEV

  return (
    <>
      {harnessMode ? (
        <CopyInstructionRow
          label="Active data source"
          value="test harness"
          hint={
            <span>
              {showDevelopmentNavControls
                ? 'App pages are using local fixture data. Switch Data source to Live API in Tweaks to fetch from the backend.'
                : 'App pages are using local fixture data instead of live backend responses.'}
            </span>
          }
        />
      ) : null}
      <CopyInstructionRow
        label="API endpoint"
        value={apiUrl}
        hint={
          <span>
            {harnessMode
              ? 'Available for copying, but not currently driving the UI.'
              : 'Point OpenAI-compatible tools at this local mesh target.'}
          </span>
        }
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
  {
    label: 'Invite token',
    value: '<mesh-invite-token>',
    hint: 'Paste your issued token into any join command below.',
    noWrapValue: true
  },
  {
    label: 'Auto join and serve command',
    value: 'mesh-llm --auto --join <mesh-invite-token>',
    prefix: '$',
    hint: 'Matches the Connect panel flow: join, select a model, and serve the API.'
  },
  { label: 'Client-only join command', value: 'mesh-llm client --join <mesh-invite-token>', prefix: '$' },
  { label: 'Blackboard skill command', value: 'mesh-llm blackboard install-skill', prefix: '$' }
]

const DEFAULT_JOIN_LINKS: LinkItem[] = [
  { href: 'https://docs.meshllm.cloud/', label: 'Setup' },
  { href: 'https://docs.meshllm.cloud/#install', label: 'Install' },
  { href: 'https://docs.meshllm.cloud/#blackboard', label: 'Blackboard' }
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
          disabled={cmd.disabled}
        />
      ))}
      <HeaderLinks links={resolvedLinks} />
    </>
  )
}

function ApiStatusChip({
  apiUrl,
  apiAccessLinks,
  apiTargetLiveness = 'checking'
}: {
  apiUrl: string
  apiAccessLinks?: LinkItem[]
  apiTargetLiveness?: ApiTargetLiveness
}) {
  const { mode } = useDataMode()
  const harnessMode = mode === 'harness'
  const targetLabel = harnessMode ? 'test harness' : apiUrl
  const dotClass = harnessMode
    ? 'bg-warn'
    : apiTargetLiveness === 'live'
      ? 'bg-good'
      : apiTargetLiveness === 'unavailable'
        ? 'bg-bad'
        : 'bg-fg-faint'

  return (
    <HeaderHoverCard
      trigger={(triggerProps) => (
        <button
          {...triggerProps}
          aria-label="API target instructions"
          className="ui-control hidden h-[var(--nav-action-size)] min-w-0 items-center gap-[var(--nav-chip-gap)] rounded-[var(--radius)] border px-[var(--nav-chip-pad-x)] py-[var(--nav-chip-pad-y)] text-fg-dim md:flex"
          type="button"
        >
          <span className={cn('size-[var(--nav-chip-dot-size)] shrink-0 rounded-full', dotClass)} aria-hidden="true" />
          <span className="text-[length:var(--nav-chip-font-size)] font-medium uppercase tracking-[0.08em] text-fg-faint">
            API target
          </span>
          <span
            className={cn(
              'max-w-64 truncate font-mono text-[length:var(--nav-chip-font-size)] font-semibold tracking-normal',
              harnessMode ? 'text-warn' : 'text-foreground'
            )}
          >
            {targetLabel}
          </span>
        </button>
      )}
      eyebrow={harnessMode ? 'Fixture data' : 'Local endpoint'}
      title="API access"
      description={
        harnessMode
          ? 'The app is not using live backend data right now. The configured endpoint is still available below.'
          : 'Copy the current local target and a couple of ready commands for OpenAI-compatible clients.'
      }
      triggerMode="click"
    >
      <ApiAccessContent apiUrl={apiUrl} dataMode={mode} links={apiAccessLinks} />
    </HeaderHoverCard>
  )
}

function ThemeSelect({ theme, onThemeChange }: { theme: Theme; onThemeChange: (theme: Theme) => void }) {
  const selectedTheme = themeOptions.find((option) => option.value === theme) ?? themeOptions[0]
  const SelectedIcon = selectedTheme.Icon

  return (
    <Select.Root
      value={theme}
      onValueChange={(value) => {
        if (isTheme(value)) onThemeChange(value)
      }}
    >
      <Select.Trigger
        aria-label={`Theme: ${selectedTheme.label}`}
        className="ui-control inline-flex size-[var(--nav-action-size)] items-center justify-center rounded-[var(--radius)] border p-0 outline-none focus-visible:!outline-none focus-visible:!outline-offset-0 focus-visible:!ring-0 focus-visible:!shadow-none data-[state=open]:border-border data-[state=open]:!shadow-none"
      >
        <SelectedIcon className="size-[var(--nav-icon-size)]" aria-hidden="true" />
        <Select.Icon asChild>
          <ChevronDown className="sr-only" aria-hidden="true" />
        </Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Content
          align="end"
          className="shadow-surface-popover z-50 w-[7.25rem] overflow-hidden rounded-[var(--radius)] border border-border bg-panel"
          position="popper"
          side="bottom"
          sideOffset={4}
          style={{ minWidth: 116 }}
        >
          <Select.Viewport className="w-full p-0">
            <Select.Group>
              <Select.Label className="sr-only">Theme</Select.Label>
              {themeOptions.map(({ value, label, Icon }) => {
                const isSelected = value === theme

                return (
                  <Select.Item
                    key={value}
                    value={value}
                    data-active={isSelected ? 'true' : undefined}
                    className={cn(
                      'ui-row-action grid h-10 w-full min-w-full box-border grid-cols-[0.875rem_1fr_0.875rem] items-center gap-1.5 border-t border-border-soft px-2.5 text-left text-[length:var(--density-type-caption)] text-fg-dim outline-none first:border-t-0',
                      isSelected ? 'text-foreground' : '',
                      'data-[highlighted]:bg-transparent data-[highlighted]:text-foreground data-[highlighted]:outline-none'
                    )}
                  >
                    <Icon className="size-[13px] justify-self-center" aria-hidden="true" />
                    <Select.ItemText>{label}</Select.ItemText>
                    {isSelected ? (
                      <Check className="size-[13px] justify-self-center" aria-hidden="true" />
                    ) : (
                      <span aria-hidden="true" />
                    )}
                  </Select.Item>
                )
              })}
            </Select.Group>
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  )
}

type CompactActionPanel = 'api' | 'join' | null

function CompactActionsMenu({
  apiUrl,
  theme,
  onThemeChange,
  onOpenDeveloperPlayground,
  showDeveloperPlayground,
  onTogglePreferences,
  apiAccessLinks,
  joinCommands,
  joinLinks
}: Pick<
  TopNavProps,
  | 'apiUrl'
  | 'theme'
  | 'onThemeChange'
  | 'showDeveloperPlayground'
  | 'onOpenDeveloperPlayground'
  | 'onTogglePreferences'
  | 'apiAccessLinks'
  | 'joinCommands'
  | 'joinLinks'
>) {
  const { mode } = useDataMode()
  const [activePanel, setActivePanel] = useState<CompactActionPanel>(null)
  const showDevelopmentNavControls = import.meta.env.DEV
  const actionRow =
    'ui-row-action flex w-full items-center justify-between gap-3 rounded-[var(--radius)] border border-border-soft px-3 py-2 text-left text-[length:var(--density-type-caption-lg)] font-medium text-foreground'

  return (
    <div className="lg:hidden">
      <HeaderHoverCard
        trigger={(triggerProps) => (
          <button
            {...triggerProps}
            aria-label="Open navigation actions"
            className="ui-control inline-flex size-[var(--nav-action-size)] items-center justify-center rounded-[var(--radius)] border"
            type="button"
          >
            <Menu className="size-[var(--nav-icon-size)]" aria-hidden="true" />
          </button>
        )}
        align="end"
        contentClassName="p-3"
        eyebrow=""
        title="Navigation actions"
        description=""
        showHeader={false}
        triggerMode="click"
      >
        {({ close }) => (
          <div className="space-y-2">
            <button
              className={actionRow}
              onClick={() => setActivePanel((panel) => (panel === 'api' ? null : 'api'))}
              type="button"
            >
              <span>API access</span>
              <span className="type-label text-fg-faint">{mode === 'harness' ? 'Harness' : 'Endpoint'}</span>
            </button>
            {activePanel === 'api' ? (
              <div className="rounded-[var(--radius)] border border-border-soft p-3">
                <ApiAccessContent apiUrl={apiUrl} dataMode={mode} links={apiAccessLinks} />
              </div>
            ) : null}
            <button
              className={actionRow}
              onClick={() => setActivePanel((panel) => (panel === 'join' ? null : 'join'))}
              type="button"
            >
              <span className="inline-flex items-center gap-2">
                <Share2 className="size-[12px]" aria-hidden="true" /> Join mesh
              </span>
              <span className="type-label text-fg-faint">Invite</span>
            </button>
            {activePanel === 'join' ? (
              <div className="rounded-[var(--radius)] border border-border-soft p-3">
                <JoinInviteContent commands={joinCommands} links={joinLinks} />
              </div>
            ) : null}
            {showDevelopmentNavControls && showDeveloperPlayground && onOpenDeveloperPlayground ? (
              <button
                className={actionRow}
                onClick={() => {
                  onOpenDeveloperPlayground()
                  close()
                }}
                type="button"
              >
                <span className="inline-flex items-center gap-2">
                  <Code2 className="size-[12px]" aria-hidden="true" /> Playground
                </span>
                <span className="type-label text-fg-faint">Dev</span>
              </button>
            ) : null}
            <div className="flex items-center justify-between rounded-[var(--radius)] border border-border-soft px-3 py-2">
              <span className="text-[length:var(--density-type-caption-lg)] font-medium text-foreground">Theme</span>
              <div className="flex items-center gap-1.5">
                {themeOptions.map(({ value, label, Icon }) => (
                  <button
                    key={value}
                    aria-label={`Theme: ${label}`}
                    aria-pressed={theme === value}
                    className={cn(
                      'ui-control inline-flex size-[var(--nav-action-size)] items-center justify-center rounded-[var(--radius)] border',
                      theme === value ? 'text-foreground' : 'text-fg-dim'
                    )}
                    onClick={() => onThemeChange(value)}
                    type="button"
                  >
                    <Icon className="size-[var(--nav-icon-size)]" aria-hidden="true" />
                  </button>
                ))}
              </div>
            </div>
            {showDevelopmentNavControls ? (
              <button
                className={actionRow}
                onClick={() => {
                  onTogglePreferences()
                  close()
                }}
                type="button"
              >
                <span className="inline-flex items-center gap-2">
                  <Settings className="size-[12px]" aria-hidden="true" /> Preferences
                </span>
                <span className="type-label text-fg-faint">Interface</span>
              </button>
            ) : null}
          </div>
        )}
      </HeaderHoverCard>
    </div>
  )
}

function UtilityActions({
  apiUrl,
  theme,
  onThemeChange,
  onOpenDeveloperPlayground,
  showDeveloperPlayground,
  onTogglePreferences,
  apiAccessLinks,
  joinCommands,
  joinLinks
}: Pick<
  TopNavProps,
  | 'apiUrl'
  | 'theme'
  | 'onThemeChange'
  | 'showDeveloperPlayground'
  | 'onOpenDeveloperPlayground'
  | 'onTogglePreferences'
  | 'apiAccessLinks'
  | 'joinCommands'
  | 'joinLinks'
>) {
  const { mode } = useDataMode()
  const showDevelopmentNavControls = import.meta.env.DEV
  const iconBtn =
    'ui-control inline-flex size-[var(--nav-action-size)] items-center justify-center rounded-[var(--radius)] border'
  const utilityBtn =
    'ui-control inline-flex size-[var(--nav-action-size)] items-center justify-center rounded-[var(--radius)] border p-0 text-[length:var(--density-type-caption)] font-medium sm:w-auto sm:gap-1.5 sm:px-2.5'

  return (
    <div className="ml-auto flex shrink-0 items-center gap-[var(--nav-action-gap)]">
      <CompactActionsMenu
        apiUrl={apiUrl}
        theme={theme}
        onThemeChange={onThemeChange}
        onOpenDeveloperPlayground={onOpenDeveloperPlayground}
        showDeveloperPlayground={showDeveloperPlayground}
        onTogglePreferences={onTogglePreferences}
        apiAccessLinks={apiAccessLinks}
        joinCommands={joinCommands}
        joinLinks={joinLinks}
      />
      <div className="hidden items-center gap-[var(--nav-action-gap)] lg:flex">
        <div className="md:hidden">
          <HeaderHoverCard
            trigger={(triggerProps) => (
              <button {...triggerProps} aria-label="Open API instructions" className={utilityBtn} type="button">
                <span aria-hidden="true" className="sm:hidden">
                  A
                </span>
                <span className="hidden sm:inline">API</span>
              </button>
            )}
            align="end"
            eyebrow={mode === 'harness' ? 'Fixture data' : 'Local endpoint'}
            title={mode === 'harness' ? 'Test harness active' : 'API access'}
            description={
              mode === 'harness'
                ? 'The compact menu shows the configured endpoint, but the UI is currently using harness data.'
                : 'The compact menu exposes the same endpoint details on smaller screens.'
            }
            triggerMode="click"
          >
            <ApiAccessContent apiUrl={apiUrl} dataMode={mode} links={apiAccessLinks} />
          </HeaderHoverCard>
        </div>
        <HeaderHoverCard
          trigger={(triggerProps) => (
            <button
              {...triggerProps}
              aria-label="Mesh join and invite instructions"
              className={utilityBtn}
              type="button"
            >
              <Share2 className="size-[12px]" aria-hidden="true" />
              <span className="hidden sm:inline">Join</span>
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
        {showDevelopmentNavControls && showDeveloperPlayground && onOpenDeveloperPlayground ? (
          <button
            className={iconBtn}
            onClick={onOpenDeveloperPlayground}
            type="button"
            aria-label="Open developer playground"
          >
            <Code2 className="size-[var(--nav-icon-size)]" />
          </button>
        ) : null}
        <ThemeSelect theme={theme} onThemeChange={onThemeChange} />
        {showDevelopmentNavControls ? (
          <button
            className={iconBtn}
            onClick={onTogglePreferences}
            type="button"
            aria-label="Open interface preferences"
          >
            <Settings className="size-[var(--nav-icon-size)]" />
          </button>
        ) : null}
      </div>
    </div>
  )
}

export function TopNav(props: TopNavProps) {
  return (
    <header className="surface-chrome sticky top-0 z-30 flex flex-nowrap items-center gap-[var(--topnav-gap)] border-b border-border-soft px-[var(--topnav-pad-x)] py-[var(--topnav-pad-y)]">
      <BrandCluster version={props.version} renderLogo={props.renderLogo} brand={props.brand} />
      <PrimaryTabs
        enabledTabs={props.enabledTabs}
        tab={props.tab}
        tabHrefs={props.tabHrefs}
        onTabChange={props.onTabChange}
        className="order-none w-auto min-w-0 pb-0 md:order-none md:w-auto md:pb-0"
      />
      <div className="hidden flex-1 md:block" />
      <ApiStatusChip
        apiUrl={props.apiUrl}
        apiAccessLinks={props.apiAccessLinks}
        apiTargetLiveness={props.apiTargetLiveness}
      />
      <UtilityActions {...props} />
    </header>
  )
}
