import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_DEVELOPER_PLAYGROUND_TAB, type DeveloperPlaygroundTabId } from '@/features/developer/playground/developer-playground-tabs'
import { DeveloperPlaygroundPageContent } from '@/features/developer/pages/DeveloperPlaygroundPage'
import { FeatureFlagProvider } from '@/lib/feature-flags'

const PLAYGROUND_FEATURE_FLAGS_STORAGE_KEY = 'mesh:test:feature-flags'

function renderDeveloperPlaygroundPage() {
  return render(
    <FeatureFlagProvider storageKey={PLAYGROUND_FEATURE_FLAGS_STORAGE_KEY}>
      <DeveloperPlaygroundPageHarness />
    </FeatureFlagProvider>,
  )
}

function DeveloperPlaygroundPageHarness() {
  const [activeTab, setActiveTab] = useState<DeveloperPlaygroundTabId>(DEFAULT_DEVELOPER_PLAYGROUND_TAB)

  return <DeveloperPlaygroundPageContent activeTab={activeTab} onTabChange={setActiveTab} />
}

function createMockDataTransfer() {
  const data = new Map<string, string>()

  return {
    dropEffect: 'none',
    effectAllowed: 'all',
    get types() { return Array.from(data.keys()) },
    getData: vi.fn((type: string) => data.get(type) ?? ''),
    setData: vi.fn((type: string, value: string) => { data.set(type, value) }),
    setDragImage: vi.fn(),
  }
}

describe('DeveloperPlaygroundPage', () => {
  beforeEach(() => {
    window.localStorage.removeItem(PLAYGROUND_FEATURE_FLAGS_STORAGE_KEY)
  })

  it('uses component-use oriented top-level tabs', () => {
    renderDeveloperPlaygroundPage()

    const tablist = screen.getByRole('tablist', { name: /developer playground component groups/i })

    expect(within(tablist).getByRole('tab', { name: /shell controls/i })).toBeInTheDocument()
    expect(within(tablist).getByRole('tab', { name: /data display/i })).toBeInTheDocument()
    expect(within(tablist).getByRole('tab', { name: /meshviz 200/i })).toBeInTheDocument()
    expect(within(tablist).getByRole('tab', { name: /chat components/i })).toBeInTheDocument()
    expect(within(tablist).getByRole('tab', { name: /configuration controls/i })).toBeInTheDocument()
    expect(within(tablist).getByRole('tab', { name: /tokens and foundations/i })).toBeInTheDocument()
    expect(within(tablist).getByRole('tab', { name: /feature flags/i })).toBeInTheDocument()
    expect(within(tablist).getAllByRole('tab').at(-1)).toHaveTextContent(/meshviz 200/i)
    expect(within(tablist).queryByRole('tab', { name: /^dashboard$/i })).not.toBeInTheDocument()
  })

  it('exposes the 200-node MeshViz benchmark as a full-width playground tab', async () => {
    const user = userEvent.setup()

    renderDeveloperPlaygroundPage()

    await user.click(screen.getByRole('tab', { name: /meshviz 200/i }))

    expect(screen.getByRole('heading', { name: /meshviz 200-node benchmark/i })).toBeInTheDocument()
    expect(screen.getByTestId('mesh-canvas')).toBeInTheDocument()
    expect(screen.queryByRole('navigation', { name: /data display previews/i })).not.toBeInTheDocument()
  })

  it('shows every shared control style in the foundations area', async () => {
    const user = userEvent.setup()

    renderDeveloperPlaygroundPage()

    await user.click(screen.getByRole('tab', { name: /tokens and foundations/i }))

    expect(screen.getByRole('button', { name: 'ui-control' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'ui-control-primary' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'ui-control-ghost' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'ui-control-destructive' })).toBeInTheDocument()
  })

  it('updates preview header text from the edit controls', async () => {
    const user = userEvent.setup()

    renderDeveloperPlaygroundPage()

    await user.click(screen.getByRole('tab', { name: /chat components/i }))

    const titleInput = screen.getByRole('textbox', { name: /header title/i })
    await user.clear(titleInput)
    await user.type(titleInput, 'Routing lab')

    const conversationLabelInput = screen.getByRole('textbox', { name: /conversation label/i })
    await user.clear(conversationLabelInput)
    await user.type(conversationLabelInput, 'Incident routing')

    expect(screen.getByRole('heading', { name: 'Routing lab' })).toBeInTheDocument()
    expect(screen.getAllByText('Incident routing')).not.toHaveLength(0)
  })

  it('displays model providers and architecture in the catalog', async () => {
    const user = userEvent.setup()

    renderDeveloperPlaygroundPage()

    await user.click(screen.getByRole('tab', { name: /data display/i }))
    await user.click(screen.getByRole('button', { name: /vision tag/i }))

    const selectedRow = screen.getByRole('button', { name: /view gemma-4-26b-a4b-it-ud model/i })
    const moeRow = screen.getByRole('button', { name: /view qwen3\.6-35b-a3b-ud model/i })

    expect(within(selectedRow).getByText('Google')).toBeInTheDocument()
    expect(within(selectedRow).getByText('Dense')).toBeInTheDocument()
    expect(within(moeRow).getByText('Alibaba')).toBeInTheDocument()
    expect(within(moeRow).getByText('MoE')).toBeInTheDocument()
    expect(within(selectedRow).queryByText('Text')).not.toBeInTheDocument()
    expect(within(selectedRow).queryByText('Vision')).not.toBeInTheDocument()
  })

  it('toggles the topology loading state preview', async () => {
    const user = userEvent.setup()

    renderDeveloperPlaygroundPage()

    await user.click(screen.getByRole('tab', { name: /data display/i }))
    await user.click(within(screen.getByRole('navigation', { name: /data display previews/i })).getByRole('button', { name: /topology and drawers/i }))

    const loadingToggle = screen.getByRole('button', { name: /show loading state/i })
    expect(loadingToggle).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('heading', { name: /network hero/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /open node drawer/i })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /open node drawer/i }))
    expect(screen.getByRole('dialog')).toBeInTheDocument()

    fireEvent.click(loadingToggle)

    expect(loadingToggle).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText('Live API')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /network hero/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /open node drawer/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()

    await user.click(loadingToggle)

    expect(loadingToggle).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('heading', { name: /network hero/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /open node drawer/i })).toBeInTheDocument()
    expect(screen.queryByText('Live API')).not.toBeInTheDocument()
  })

  it('adds and removes GPU controls from the configuration preview', async () => {
    const user = userEvent.setup()

    renderDeveloperPlaygroundPage()

    await user.click(screen.getByRole('tab', { name: /configuration controls/i }))

    expect(screen.getAllByRole('region', { name: /capacity/i })).toHaveLength(8)

    await user.click(screen.getByRole('button', { name: 'Add GPU' }))
    expect(screen.getAllByRole('region', { name: /capacity/i })).toHaveLength(9)

    await user.click(screen.getByRole('button', { name: /remove gpu 8/i }))
    expect(screen.getAllByRole('region', { name: /capacity/i })).toHaveLength(8)
  })

  it('clears a non-fitting drag target after mouse release', async () => {
    const user = userEvent.setup()
    const dataTransfer = createMockDataTransfer()

    dataTransfer.setData('text/model', 'mixtral')

    renderDeveloperPlaygroundPage()

    await user.click(screen.getByRole('tab', { name: /configuration controls/i }))

    const targetContainer = screen.getAllByRole('region', { name: /capacity/i })[0]

    fireEvent.dragEnter(targetContainer, { dataTransfer })
    fireEvent.dragOver(targetContainer, { dataTransfer })

    expect(within(targetContainer).getByText(/no fit/i)).toBeInTheDocument()

    fireEvent.mouseUp(window)

    expect(within(targetContainer).queryByText(/no fit/i)).not.toBeInTheDocument()
  })

  it('adds and removes model config cards', async () => {
    const user = userEvent.setup()

    renderDeveloperPlaygroundPage()

    await user.click(screen.getByRole('tab', { name: /configuration controls/i }))
    await user.click(within(screen.getByRole('navigation', { name: /configuration control previews/i })).getByRole('button', { name: /config cards/i }))

    expect(screen.getAllByRole('button', { name: /remove .* from /i })).toHaveLength(4)

    await user.click(screen.getByRole('button', { name: 'Add config card' }))

    const phiRemoveButton = screen.getByRole('button', { name: /remove phi-4-mini from gpu/i })
    expect(phiRemoveButton).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /remove .* from /i })).toHaveLength(5)

    await user.click(phiRemoveButton)

    expect(screen.queryByRole('button', { name: /remove phi-4-mini from gpu/i })).not.toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /remove .* from /i })).toHaveLength(4)
  })

  it('toggles feature flags and persists only local overrides', async () => {
    const user = userEvent.setup()

    renderDeveloperPlaygroundPage()

    await user.click(screen.getByRole('tab', { name: /feature flags/i }))

    expect(screen.getByText(/app-wide rollout switches/i)).toBeInTheDocument()
    expect(screen.getByText(/shows the configuration app surface/i)).toBeInTheDocument()

    const stateControl = screen.getByRole('radiogroup', { name: /new configuration page state/i })
    const enabledOption = within(stateControl).getByRole('radio', { name: /on/i })
    expect(enabledOption).toHaveAttribute('aria-checked', 'false')

    await user.click(enabledOption)

    expect(enabledOption).toHaveAttribute('aria-checked', 'true')
    expect(enabledOption).toHaveAttribute('data-selected-tone', 'accent')
    expect(screen.getAllByText(/local override/i)).toHaveLength(2)
    expect(screen.getByRole('button', { name: /reset new configuration page/i })).toBeInTheDocument()
    expect(screen.getByText(/"global"/)).toBeInTheDocument()

    await waitFor(() => {
      expect(window.localStorage.getItem(PLAYGROUND_FEATURE_FLAGS_STORAGE_KEY)).toBe('{"global":{"newConfigurationPage":true}}')
    })
  })

  it('groups configuration section flags in the feature flag playground', async () => {
    const user = userEvent.setup()

    renderDeveloperPlaygroundPage()

    await user.click(screen.getByRole('tab', { name: /feature flags/i }))
    await user.click(within(screen.getByRole('navigation', { name: /feature flag groups/i })).getByRole('button', { name: /configuration/i }))

    expect(screen.getByText(/rollout switches for configuration sections/i)).toBeInTheDocument()
    expect(screen.getByText(/shows the temporary signing, key binding/i)).toBeInTheDocument()
    expect(screen.getByText(/shows the temporary integrations section/i)).toBeInTheDocument()

    const signingControl = screen.getByRole('radiogroup', { name: /signing \/ attestation state/i })
    const signingEnabledOption = within(signingControl).getByRole('radio', { name: /on/i })
    expect(signingEnabledOption).toHaveAttribute('aria-checked', 'false')

    await user.click(signingEnabledOption)

    expect(signingEnabledOption).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('button', { name: /reset signing \/ attestation/i })).toBeInTheDocument()

    await waitFor(() => {
      expect(window.localStorage.getItem(PLAYGROUND_FEATURE_FLAGS_STORAGE_KEY)).toBe('{"configuration":{"signingAttestation":true}}')
    })
  })
})
