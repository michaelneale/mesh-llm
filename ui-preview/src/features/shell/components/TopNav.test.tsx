import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { TopNav } from '@/features/shell/components/TopNav'

function installClipboard(writeText: (text: string) => Promise<void>) {
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText },
  })
}

function installPointerCaptureShim() {
  Object.defineProperty(HTMLElement.prototype, 'hasPointerCapture', {
    configurable: true,
    value: () => false,
  })
  Object.defineProperty(HTMLElement.prototype, 'setPointerCapture', {
    configurable: true,
    value: () => undefined,
  })
  Object.defineProperty(HTMLElement.prototype, 'releasePointerCapture', {
    configurable: true,
    value: () => undefined,
  })
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
    configurable: true,
    value: () => undefined,
  })
}

function renderTopNav(overrides: Partial<React.ComponentProps<typeof TopNav>> = {}) {
  return render(
    <TopNav
      apiUrl="http://127.0.0.1:9337/v1"
      onTabChange={vi.fn()}
      onThemeChange={vi.fn()}
      onTogglePreferences={vi.fn()}
      tab="network"
      theme="auto"
      version="0.64.0"
      {...overrides}
    />,
  )
}

async function waitForHoverDelay() {
  await new Promise((resolve) => window.setTimeout(resolve, 220))
}

describe('TopNav', () => {
  beforeEach(() => {
    installPointerCaptureShim()
  })

  afterEach(() => {
    vi.useRealTimers()
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined })
  })

  it('opens API instructions and keeps copy state scoped to the clicked row', async () => {
    const user = userEvent.setup()
    const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)
    installClipboard(writeText)

    renderTopNav()

    const apiButton = screen.getByRole('button', { name: 'API target instructions' })

    await user.hover(apiButton)
    await waitForHoverDelay()
    expect(screen.queryByRole('heading', { name: 'API access' })).not.toBeInTheDocument()

    await user.click(apiButton)

    expect(await screen.findByRole('heading', { name: 'API access' })).toBeInTheDocument()
    const apiDescription = screen.getByText('The app is not using live backend data right now. The configured endpoint is still available below.')
    expect(apiDescription).toHaveClass('max-w-none')
    expect(apiDescription).not.toHaveClass('max-w-[44ch]')

    const endpointCopyButton = screen.getByLabelText('Copy API endpoint')
    const commandCopyButton = screen.getByLabelText('Copy List models command')

    const activeDataSourceValue = screen.getAllByText('test harness').find((element) => element.classList.contains('break-words'))
    expect(activeDataSourceValue).toHaveClass('break-words')
    expect(activeDataSourceValue).not.toHaveClass('break-all')
    expect(screen.getByText('http://127.0.0.1:9337/v1')).toHaveClass('break-words')
    expect(screen.getByText('curl http://127.0.0.1:9337/v1/models')).toHaveClass('break-words')

    await user.click(endpointCopyButton)

    expect(writeText).toHaveBeenCalledWith('http://127.0.0.1:9337/v1')
    expect(endpointCopyButton).toHaveTextContent('Copied')
    expect(commandCopyButton).toHaveTextContent('Copy')
  })

  it('opens join instructions and copies the selected join command', async () => {
    const user = userEvent.setup()
    const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)
    installClipboard(writeText)

    renderTopNav()

    const joinButton = screen.getByRole('button', { name: 'Mesh join and invite instructions' })

    await user.hover(joinButton)
    await waitForHoverDelay()
    expect(screen.queryByRole('heading', { name: 'Join or invite' })).not.toBeInTheDocument()

    await user.click(joinButton)

    expect(await screen.findByRole('heading', { name: 'Join or invite' })).toBeInTheDocument()
    const joinDescription = screen.getByText('Keep the common join flows close when you need to add a node, client, or agent quickly.')
    expect(joinDescription).toHaveClass('max-w-none')
    expect(joinDescription).not.toHaveClass('max-w-[44ch]')

    expect(screen.getByText('<mesh-invite-token>')).toHaveClass('whitespace-nowrap')

    expect(screen.getByText('mesh-llm --auto --join <mesh-invite-token>')).toHaveClass('break-words')
    expect(screen.getByText('mesh-llm client --join <mesh-invite-token>')).toHaveClass('break-words')

    const clientCopyButton = screen.getByLabelText('Copy Client-only join command')
    await user.click(clientCopyButton)

    expect(writeText).toHaveBeenCalledWith('mesh-llm client --join <mesh-invite-token>')
    expect(clientCopyButton).toHaveTextContent('Copied')
  })

  it('shows the playground trigger only when the dev flag is enabled', async () => {
    const user = userEvent.setup()
    const onOpenDeveloperPlayground = vi.fn()

    const { rerender } = renderTopNav({
      showDeveloperPlayground: false,
      onOpenDeveloperPlayground: undefined,
    })

    expect(screen.queryByRole('button', { name: 'Open developer playground' })).not.toBeInTheDocument()

    rerender(
      <TopNav
        apiUrl="http://127.0.0.1:9337/v1"
        onOpenDeveloperPlayground={onOpenDeveloperPlayground}
        onTabChange={vi.fn()}
        onThemeChange={vi.fn()}
        onTogglePreferences={vi.fn()}
        showDeveloperPlayground
        tab={null}
        theme="auto"
        version="0.64.0"
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Open developer playground' }))
    expect(onOpenDeveloperPlayground).toHaveBeenCalledTimes(1)
  })

  it('uses route-provided hrefs for primary tabs without changing click handling', async () => {
    const user = userEvent.setup()
    const onTabChange = vi.fn()

    renderTopNav({ onTabChange, tabHrefs: { configuration: '/configuration/toml-review' } })

    const configurationTab = screen.getByRole('link', { name: 'Configuration' })
    expect(configurationTab).toHaveAttribute('href', '/configuration/toml-review')

    await user.click(configurationTab)

    expect(onTabChange).toHaveBeenCalledWith('configuration')
  })

  it('selects auto, dark, and light from the nav theme menu', async () => {
    const user = userEvent.setup()
    const onThemeChange = vi.fn()

    renderTopNav({ onThemeChange, theme: 'auto' })

    await user.click(screen.getByRole('combobox', { name: 'Theme: Auto' }))
    await user.click(await screen.findByRole('option', { name: /Dark/ }))

    expect(onThemeChange).toHaveBeenCalledWith('dark')
  })

  it('selects primary routes from compact icon buttons', async () => {
    const user = userEvent.setup()
    const onTabChange = vi.fn()

    renderTopNav({ onTabChange })

    const networkButton = screen.getByRole('button', { name: 'Network' })
    const chatButton = screen.getByRole('button', { name: 'Chat' })

    expect(networkButton).toHaveClass('ui-control-primary')
    expect(chatButton).toHaveClass('ui-control-primary')

    await user.click(chatButton)

    expect(onTabChange).toHaveBeenCalledWith('chat')
  })

  it('opens compact navigation actions without a heading and dismisses terminal actions', async () => {
    const user = userEvent.setup()
    const onOpenDeveloperPlayground = vi.fn()
    const onThemeChange = vi.fn()
    const onTogglePreferences = vi.fn()

    renderTopNav({ onOpenDeveloperPlayground, onThemeChange, onTogglePreferences, showDeveloperPlayground: true })

    await user.click(screen.getByRole('button', { name: 'Open navigation actions' }))

    expect(await screen.findByRole('dialog', { name: 'Navigation actions' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Quick actions' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /API access/ }))
    expect(screen.getByLabelText('Copy API endpoint')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Theme: Dark' }))
    expect(onThemeChange).toHaveBeenCalledWith('dark')

    await user.click(screen.getByRole('button', { name: /Playground/ }))
    expect(onOpenDeveloperPlayground).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('dialog', { name: 'Navigation actions' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Open navigation actions' }))
    await user.click(screen.getByRole('button', { name: /Preferences/ }))
    expect(onTogglePreferences).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('dialog', { name: 'Navigation actions' })).not.toBeInTheDocument()
  })

  it('hides disabled primary tabs while keeping enabled tabs available', () => {
    renderTopNav({ enabledTabs: { configuration: false } })

    expect(screen.getByRole('link', { name: 'Network' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Chat' })).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Configuration' })).not.toBeInTheDocument()
  })

  it('preserves basepath-aware hrefs for native link behavior', () => {
    const onTabChange = vi.fn()

    renderTopNav({
      onTabChange,
      tabHrefs: {
        network: '/mesh/llm/ui-preview/',
        chat: '/mesh/llm/ui-preview/chat',
        configuration: '/mesh/llm/ui-preview/configuration/toml-review',
      },
    })

    expect(screen.getByRole('link', { name: 'Network' })).toHaveAttribute('href', '/mesh/llm/ui-preview/')
    expect(screen.getByRole('link', { name: 'Chat' })).toHaveAttribute('href', '/mesh/llm/ui-preview/chat')
    expect(screen.getByRole('link', { name: 'Configuration' })).toHaveAttribute('href', '/mesh/llm/ui-preview/configuration/toml-review')

    fireEvent.click(screen.getByRole('link', { name: 'Chat' }), { button: 0, metaKey: true })

    expect(onTabChange).not.toHaveBeenCalled()
  })
})
