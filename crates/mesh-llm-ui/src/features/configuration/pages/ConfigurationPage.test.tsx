import type { ReactElement, ReactNode } from 'react'
import { act, fireEvent, render as rtlRender, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProviders } from '@/app/providers/AppProviders'
import * as configQueryModule from '@/features/configuration/api/use-config-query'

const blockedBlocker = vi.hoisted(() => ({ status: 'blocked', proceed: vi.fn(), reset: vi.fn() }))
const idleBlocker = vi.hoisted(() => ({ status: 'idle', proceed: vi.fn(), reset: vi.fn() }))
const mockUseBlocker = vi.hoisted(() => vi.fn())
const defaultBlockerTransition = vi.hoisted(() => ({
  current: { pathname: '/configuration/local-deployment' },
  next: { pathname: '/chat' }
}))
const featureFlagMocks = vi.hoisted(() => ({
  integrationsEnabled: false,
  signingAttestationEnabled: false
}))

vi.mock('@tanstack/react-router', () => ({
  useBlocker: mockUseBlocker
}))

vi.mock('@/lib/feature-flags', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/feature-flags')>()

  return {
    ...actual,
    useBooleanFeatureFlag: vi.fn((path: string) => {
      if (path === 'configuration/integrations') return featureFlagMocks.integrationsEnabled
      if (path === 'configuration/signingAttestation') return featureFlagMocks.signingAttestationEnabled
      return true
    })
  }
})

import { ConfigurationPage } from '@/features/configuration/pages/ConfigurationPage'
import { CONFIGURATION_HARNESS } from '@/features/app-tabs/data'
import type { DataMode } from '@/lib/data-mode/data-mode-context'

function TestProviders({ children, dataMode = 'harness' }: { children: ReactNode; dataMode?: DataMode }) {
  return (
    <AppProviders initialDataMode={dataMode} persistDataMode={false}>
      {children}
    </AppProviders>
  )
}

function render(ui: ReactElement, options?: { dataMode?: DataMode }) {
  return rtlRender(ui, {
    wrapper: ({ children }) => <TestProviders dataMode={options?.dataMode}>{children}</TestProviders>
  })
}

function getCarrackSection() {
  const section = screen.getByRole('button', { name: /collapse carrack/i }).closest('section')
  if (!section) throw new Error('Expected carrack section')
  return section
}

function getTomlSource() {
  const source = screen.getByRole('textbox', { name: /configuration toml source/i })
  if (!(source instanceof HTMLTextAreaElement)) throw new Error('Expected configuration TOML source')
  return source
}

async function openTomlOutput(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('tab', { name: 'TOML Output' }))
  expect(screen.getByRole('heading', { name: 'Generated TOML' })).toBeInTheDocument()
  return getTomlSource()
}

function countTomlOccurrences(value: string) {
  return getTomlSource().value.split(value).length - 1
}

async function dispatchShortcut(key: string, init: KeyboardEventInit = {}) {
  const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init })

  await act(async () => {
    window.dispatchEvent(event)
  })

  return event
}

function liveControlConfigData() {
  return {
    bootstrap: {
      enabled: true,
      local_only: true,
      requires_explicit_remote_endpoint: true,
      endpoint: 'control://owner'
    },
    snapshot: {
      revision: 7,
      config: {}
    }
  }
}

describe('ConfigurationPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useRealTimers()
    featureFlagMocks.integrationsEnabled = false
    featureFlagMocks.signingAttestationEnabled = false
    mockUseBlocker.mockImplementation(
      ({ shouldBlockFn }: { shouldBlockFn: (transition: typeof defaultBlockerTransition) => boolean }) =>
        shouldBlockFn(defaultBlockerTransition) ? blockedBlocker : idleBlocker
    )
  })

  it('renders the persistent header, shared tab bar, and defaults workspace first', () => {
    render(<ConfigurationPage enableNavigationBlocker={false} />)

    expect(screen.getByRole('heading', { name: 'Configuration' })).toBeInTheDocument()
    expect(screen.getByText('carrack.local')).toBeInTheDocument()
    expect(screen.getByText('Configuration Path')).toBeInTheDocument()
    expect(screen.getByText('~/.mesh-llm/config.toml')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /save config/i })).toBeDisabled()

    for (const label of ['Defaults', 'Model Deployment', 'TOML Output']) {
      expect(screen.getByRole('tab', { name: label })).toBeInTheDocument()
    }
    expect(screen.queryByRole('tab', { name: 'Signing / Attestation' })).not.toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: 'Integrations' })).not.toBeInTheDocument()

    expect(screen.getByRole('heading', { name: /inherited defaults/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /runtime/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /speculative decoding/i })).toBeInTheDocument()
    expect(screen.getByText('Default slots / parallel requests')).toBeInTheDocument()
    expect(screen.getByRole('complementary', { name: /\[defaults\]/i })).toBeInTheDocument()
    expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument()
  })

  it('shows temporary configuration sections only when their feature flags are enabled', async () => {
    const user = userEvent.setup()
    featureFlagMocks.integrationsEnabled = true
    featureFlagMocks.signingAttestationEnabled = true

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    const signingTab = screen.getByRole('tab', { name: 'Signing / Attestation' })
    const integrationsTab = screen.getByRole('tab', { name: 'Integrations' })
    expect(signingTab).toBeInTheDocument()
    expect(integrationsTab).toBeInTheDocument()

    await user.click(signingTab)
    expect(screen.getByRole('heading', { name: 'Signing / Attestation' })).toBeInTheDocument()
    expect(screen.getByText(/key binding, attestation receipts/i)).toBeInTheDocument()

    await user.click(integrationsTab)
    expect(screen.getByRole('heading', { name: 'Integrations' })).toBeInTheDocument()
    expect(screen.getByText(/plugin and external endpoint defaults/i)).toBeInTheDocument()
  })

  it('keeps a directly requested gated temporary section on the defaults workspace', () => {
    render(<ConfigurationPage initialTab="signing" enableNavigationBlocker={false} />)

    expect(screen.queryByRole('tab', { name: 'Signing / Attestation' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Signing / Attestation' })).not.toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Defaults' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('heading', { name: /inherited defaults/i })).toBeInTheDocument()
  })

  it('applies configuration section feature flags independently', () => {
    featureFlagMocks.signingAttestationEnabled = true
    featureFlagMocks.integrationsEnabled = false

    const { rerender } = render(<ConfigurationPage enableNavigationBlocker={false} />)

    expect(screen.getByRole('tab', { name: 'Signing / Attestation' })).toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: 'Integrations' })).not.toBeInTheDocument()

    featureFlagMocks.signingAttestationEnabled = false
    featureFlagMocks.integrationsEnabled = true
    rerender(<ConfigurationPage enableNavigationBlocker={false} />)

    expect(screen.queryByRole('tab', { name: 'Signing / Attestation' })).not.toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Integrations' })).toBeInTheDocument()
  })

  it('renders the Defaults sections and updates the active sidebar category', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    const memoryButton = screen.getByRole('button', { name: /memory/i })
    await user.click(memoryButton)

    expect(memoryButton).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('heading', { name: 'Runtime' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Backend' })).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Memory' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Speculative Decoding' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Request Defaults' })).toBeInTheDocument()
    expect(screen.getByText('Model Runtime')).toBeInTheDocument()
    expect(screen.getByText('KV cache policy')).toBeInTheDocument()
    expect(screen.getByText('Memory / safety margin')).toBeInTheDocument()
    expect(screen.getByText('Reasoning format')).toBeInTheDocument()
    expect(screen.getByText('Temperature')).toBeInTheDocument()
  })

  it('includes Defaults edits in dirty state, save, revert, and TOML review', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    const saveButton = screen.getByRole('button', { name: /save config/i })
    const defaultsTab = screen.getByRole('tab', { name: 'Defaults' })
    const tomlReviewTab = screen.getByRole('tab', { name: 'TOML Output' })
    expect(saveButton).toBeDisabled()
    expect(defaultsTab).not.toHaveAttribute('data-tab-dirty')

    await user.click(screen.getByRole('radio', { name: 'throughput' }))
    expect(saveButton).toBeEnabled()
    expect(defaultsTab).toHaveAttribute('data-tab-dirty', 'true')
    expect(tomlReviewTab).toHaveAttribute('data-tab-dirty', 'true')

    await user.click(tomlReviewTab)
    expect(screen.getByRole('heading', { name: 'Generated TOML' })).toBeInTheDocument()
    expect(screen.getByText('edits this node only')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Validation' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Effective launch summary' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Save' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /save config & sign/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /revert to disk/i })).not.toBeInTheDocument()
    expect(getTomlSource().value).toContain('tuning_profile = "throughput"')

    await user.click(saveButton)
    expect(defaultsTab).not.toHaveAttribute('data-tab-dirty')
    expect(tomlReviewTab).not.toHaveAttribute('data-tab-dirty')

    await user.click(screen.getByRole('tab', { name: 'Defaults' }))
    await user.click(screen.getAllByRole('radio', { name: 'saver' })[0])
    expect(defaultsTab).toHaveAttribute('data-tab-dirty', 'true')

    await user.click(screen.getByRole('button', { name: /revert/i }))
    expect(saveButton).toBeDisabled()
    expect(defaultsTab).not.toHaveAttribute('data-tab-dirty')
    await user.click(tomlReviewTab)
    expect(getTomlSource().value).toContain('tuning_profile = "throughput"')
  })

  it('saves live defaults through useConfigQuery.applyDefaults only when Save config is clicked', async () => {
    const applyDefaults = vi.fn().mockResolvedValue({
      success: true,
      current_revision: 8,
      config_hash: 'abc123',
      apply_mode: 'live'
    })
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: CONFIGURATION_HARNESS,
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: liveControlConfigData(),
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults
    })

    render(<ConfigurationPage enableNavigationBlocker={false} />, { dataMode: 'live' })

    const tuningProfileControl = within(screen.getByRole('radiogroup', { name: 'Default tuning profile' }))
    const saveButton = screen.getByRole('button', { name: /save config/i })

    fireEvent.click(tuningProfileControl.getByRole('radio', { name: 'throughput' }))
    expect(applyDefaults).not.toHaveBeenCalled()

    fireEvent.click(tuningProfileControl.getByRole('radio', { name: 'saver' }))
    expect(applyDefaults).not.toHaveBeenCalled()

    fireEvent.click(saveButton)

    await waitFor(() => expect(applyDefaults).toHaveBeenCalledTimes(1))
    expect(applyDefaults).toHaveBeenCalledWith(
      expect.objectContaining({
        'tuning-profile': 'saver'
      })
    )

    useConfigQuerySpy.mockRestore()
  })

  it('shows owner-control remediation and keeps dirty state when live saving is disabled', async () => {
    const applyDefaults = vi.fn()
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: CONFIGURATION_HARNESS,
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: {
          bootstrap: {
            enabled: false,
            local_only: true,
            requires_explicit_remote_endpoint: true,
            disabled_reason: 'missing_owner_identity',
            message: 'Configuration saving requires a local owner identity.',
            suggested_commands: [
              'mesh-llm auth status',
              'mesh-llm auth init --no-passphrase',
              'mesh-llm serve --owner-required'
            ]
          }
        },
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults
    })

    render(<ConfigurationPage enableNavigationBlocker={false} />, { dataMode: 'live' })

    const readOnlyHeading = screen.getByRole('heading', { name: 'Configuration UI is read-only' })
    const inheritedDefaultsHeading = screen.getByRole('heading', { name: /inherited defaults/i })
    const defaultsTab = screen.getByRole('tab', { name: 'Defaults' })
    expect(readOnlyHeading).toBeInTheDocument()
    expect(defaultsTab.compareDocumentPosition(readOnlyHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(
      inheritedDefaultsHeading.compareDocumentPosition(readOnlyHeading) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
    expect(
      screen.getByText('No owner-control identity on this node, run both commands to unlock saving.')
    ).toBeInTheDocument()
    expect(screen.getByText('missing owner identity')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /docs/i })).toHaveAttribute('href', 'https://docs.meshllm.cloud/')
    expect(screen.queryByRole('button', { name: /copy both/i })).not.toBeInTheDocument()
    expect(screen.getAllByText('mesh-llm')).toHaveLength(2)
    expect(screen.getByText('auth')).toBeInTheDocument()
    expect(screen.getByText('init')).toBeInTheDocument()
    expect(screen.getByText('serve')).toBeInTheDocument()
    expect(screen.getByText('--no-passphrase')).toBeInTheDocument()
    expect(screen.getByText('--owner-required')).toBeInTheDocument()
    expect(screen.getByText('Initialize owner identity (creates a local keypair)')).toBeInTheDocument()
    expect(screen.getByText('Restart the daemon so the new identity takes effect')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Copy mesh-llm auth init --no-passphrase' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Copy mesh-llm serve --owner-required' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('radio', { name: 'throughput' }))
    const saveButton = screen.getByRole('button', { name: /save config/i })
    expect(saveButton).toBeDisabled()
    expect(saveButton).toHaveAttribute('title', 'Runtime control is disabled: missing owner identity')
    expect(defaultsTab).toHaveAttribute('data-tab-dirty', 'true')

    await dispatchShortcut('s', { ctrlKey: true })

    expect(applyDefaults).not.toHaveBeenCalled()
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Config was not saved. Runtime control is disabled: missing owner identity.'
    )
    expect(defaultsTab).toHaveAttribute('data-tab-dirty', 'true')

    useConfigQuerySpy.mockRestore()
  })

  it('shows runtime-control apply errors without rewriting them as missing owner identity', async () => {
    const applyDefaults = vi.fn().mockResolvedValue({
      success: false,
      current_revision: 7,
      config_hash: 'abc123',
      apply_mode: 'unspecified',
      error: 'revision conflict: current revision is 9'
    })
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: CONFIGURATION_HARNESS,
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: liveControlConfigData(),
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults
    })

    render(<ConfigurationPage enableNavigationBlocker={false} />, { dataMode: 'live' })

    fireEvent.click(screen.getByRole('radio', { name: 'throughput' }))
    fireEvent.click(screen.getByRole('button', { name: /save config/i }))

    await waitFor(() => expect(applyDefaults).toHaveBeenCalledTimes(1))
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Config was not saved. Runtime control rejected the update: revision conflict: current revision is 9'
    )
    expect(screen.getByRole('alert')).not.toHaveTextContent('missing owner identity')
    expect(screen.getByRole('tab', { name: 'Defaults' })).toHaveAttribute('data-tab-dirty', 'true')

    useConfigQuerySpy.mockRestore()
  })

  it('shows a busy Save config button while live defaults are being written', async () => {
    let resolveApply: (value: {
      success: boolean
      current_revision: number
      config_hash: string
      apply_mode: string
    }) => void = () => undefined
    const applyDefaults = vi.fn(
      () =>
        new Promise<{
          success: boolean
          current_revision: number
          config_hash: string
          apply_mode: string
        }>((resolve) => {
          resolveApply = resolve
        })
    )
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: CONFIGURATION_HARNESS,
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: liveControlConfigData(),
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults
    })

    render(<ConfigurationPage enableNavigationBlocker={false} />, { dataMode: 'live' })

    fireEvent.click(screen.getByRole('radio', { name: 'throughput' }))
    fireEvent.click(screen.getByRole('button', { name: /save config/i }))

    const savingButton = screen.getByRole('button', { name: /saving config/i })
    expect(savingButton).toBeDisabled()
    expect(savingButton).toHaveAttribute('aria-busy', 'true')

    await act(async () => {
      resolveApply({ success: true, current_revision: 8, config_hash: 'abc123', apply_mode: 'live' })
    })

    expect(screen.getByRole('button', { name: /save config/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /save config/i })).not.toHaveAttribute('aria-busy', 'true')

    useConfigQuerySpy.mockRestore()
  })

  it('does not re-apply saved live defaults when the hook callback identity changes', async () => {
    const firstApplyDefaults = vi.fn().mockResolvedValue({
      success: true,
      current_revision: 8,
      config_hash: 'abc123',
      apply_mode: 'live'
    })
    const secondApplyDefaults = vi.fn().mockResolvedValue({
      success: true,
      current_revision: 9,
      config_hash: 'def456',
      apply_mode: 'live'
    })
    let currentApplyDefaults = firstApplyDefaults

    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockImplementation(() => ({
      data: CONFIGURATION_HARNESS,
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: liveControlConfigData(),
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults: currentApplyDefaults
    }))

    const { rerender } = render(<ConfigurationPage enableNavigationBlocker={false} />, { dataMode: 'live' })

    fireEvent.click(screen.getByRole('radio', { name: 'throughput' }))
    fireEvent.click(screen.getByRole('button', { name: /save config/i }))

    await waitFor(() => expect(firstApplyDefaults).toHaveBeenCalledTimes(1))

    currentApplyDefaults = secondApplyDefaults
    rerender(<ConfigurationPage enableNavigationBlocker={false} />)

    expect(firstApplyDefaults).toHaveBeenCalledTimes(1)
    expect(secondApplyDefaults).not.toHaveBeenCalled()

    useConfigQuerySpy.mockRestore()
  })

  it('shows hydrated live non-default defaults while omitting unchanged metadata sections', async () => {
    const user = userEvent.setup()
    const liveDefaults = {
      ...CONFIGURATION_HARNESS.defaults,
      settings: CONFIGURATION_HARNESS.defaults.settings.map((setting) =>
        setting.id === 'temperature'
          ? {
              ...setting,
              control: {
                ...setting.control,
                value: '0.8'
              }
            }
          : setting.id === 'server-alias'
            ? {
                ...setting,
                control: {
                  ...setting.control,
                  value: 'carrack-mesh'
                }
              }
            : setting.id === 'activation-wire-dtype'
              ? {
                  ...setting,
                  control: {
                    ...setting.control,
                    value: 'q8'
                  }
                }
              : setting.id === 'image-min-tokens'
                ? {
                    ...setting,
                    control: {
                      ...setting.control,
                      value: '64'
                    }
                  }
                : setting
      )
    }
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: { ...CONFIGURATION_HARNESS, defaults: liveDefaults },
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: null,
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults: vi.fn()
    })

    render(<ConfigurationPage enableNavigationBlocker={false} />, { dataMode: 'live' })

    const initialTomlSource = await openTomlOutput(user)
    expect(initialTomlSource.value).toContain('[defaults.request_defaults]')
    expect(initialTomlSource.value).toContain('temperature = 0.8')
    expect(initialTomlSource.value).toContain('[defaults.skippy]')
    expect(initialTomlSource.value).toContain('activation_wire_dtype = "q8"')
    expect(initialTomlSource.value).toContain('[defaults.multimodal]')
    expect(initialTomlSource.value).toContain('image_min_tokens = 64')
    expect(initialTomlSource.value).toContain('[defaults.advanced.server]')
    expect(initialTomlSource.value).toContain('alias = "carrack-mesh"')

    await user.click(screen.getByRole('tab', { name: 'Defaults' }))
    await user.click(screen.getByRole('button', { name: /request defaults/i }))
    expect(screen.getByRole('slider', { name: 'Temperature' })).toHaveValue('0.8')
    const skippyTransport = within(screen.getByRole('radiogroup', { name: 'Binary stage transport' }))
    const multimodalOffload = within(screen.getByRole('radiogroup', { name: 'MMProj offload' }))
    await user.click(skippyTransport.getByRole('radio', { name: 'on' }))
    await user.click(multimodalOffload.getByRole('radio', { name: 'on' }))
    const updatedTomlSource = await openTomlOutput(user)
    expect(updatedTomlSource.value).toContain('[defaults.skippy]')
    expect(updatedTomlSource.value).toContain('[defaults.multimodal]')

    useConfigQuerySpy.mockRestore()
  })

  it('does not call applyDefaults for defaults edits outside live mode', async () => {
    const user = userEvent.setup()
    const applyDefaults = vi.fn()
    const useConfigQuerySpy = vi.spyOn(configQueryModule, 'useConfigQuery').mockReturnValue({
      data: CONFIGURATION_HARNESS,
      isError: false,
      isFetching: false,
      isPending: false,
      statusQuery: { refetch: vi.fn() } as never,
      modelsQuery: { refetch: vi.fn() } as never,
      controlConfigQuery: {
        data: null,
        isError: false,
        isFetching: false,
        isPending: false
      } as never,
      applyDefaults
    })

    render(<ConfigurationPage enableNavigationBlocker={false} />, { dataMode: 'harness' })

    await user.click(screen.getByRole('radio', { name: 'throughput' }))

    await waitFor(() => expect(screen.getByRole('button', { name: /save config/i })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: /save config/i }))
    expect(applyDefaults).not.toHaveBeenCalled()

    useConfigQuerySpy.mockRestore()
  })

  it('uses an interactive slot meter instead of a Defaults slot slider', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    expect(screen.queryByRole('slider', { name: /default slots/i })).not.toBeInTheDocument()
    expect(screen.getByRole('radio', { name: '4 slots' })).toBeChecked()

    await user.click(screen.getByRole('radio', { name: '12 slots' }))

    expect(screen.getByRole('radio', { name: '12 slots' })).toBeChecked()
    expect(screen.getByText('3.6 GB · 12 × 0.30 GB')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /save config/i })).toBeEnabled()

    await user.click(screen.getByRole('tab', { name: 'TOML Output' }))
    expect(getTomlSource().value).toContain('[defaults.throughput]')
    expect(getTomlSource().value).toContain('parallel = 12')
  })

  it('supports dragging across the Defaults slot meter', () => {
    render(<ConfigurationPage enableNavigationBlocker={false} />)

    const slotMeter = screen.getByTestId('defaults-slot-meter')
    vi.spyOn(slotMeter, 'getBoundingClientRect').mockReturnValue({
      bottom: 16,
      height: 16,
      left: 10,
      right: 314,
      top: 0,
      width: 304,
      x: 10,
      y: 0,
      toJSON: () => ({})
    })

    fireEvent.pointerDown(slotMeter, { buttons: 1, clientX: 10, pointerId: 1 })
    expect(screen.getByRole('radio', { name: '1 slot' })).toBeChecked()

    fireEvent.pointerMove(slotMeter, { buttons: 1, clientX: 238, pointerId: 1 })
    expect(screen.getByRole('radio', { name: '12 slots' })).toBeChecked()
    expect(screen.getByText('3.6 GB · 12 × 0.30 GB')).toBeInTheDocument()
  })

  it('updates KV cache memory tiers from the selected policy', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    const policyControl = within(screen.getByRole('radiogroup', { name: 'KV cache policy' }))
    const tiers = () =>
      within(screen.getByRole('group', { name: 'KV cache memory tiers' }))
        .getAllByText(/^K /)
        .map((node) => node.closest('[data-kv-tier-active]'))

    expect(tiers().map((node) => node?.getAttribute('data-kv-tier-active'))).toEqual(['true', 'true', 'true'])

    await user.click(policyControl.getByRole('radio', { name: 'quality' }))
    expect(tiers().map((node) => node?.getAttribute('data-kv-tier-active'))).toEqual(['true', 'false', 'false'])

    await user.click(policyControl.getByRole('radio', { name: 'balanced' }))
    expect(tiers().map((node) => node?.getAttribute('data-kv-tier-active'))).toEqual(['false', 'true', 'false'])

    await user.click(policyControl.getByRole('radio', { name: 'saver' }))
    expect(tiers().map((node) => node?.getAttribute('data-kv-tier-active'))).toEqual(['false', 'false', 'true'])
  })

  it('renders speculative decoding defaults and writes them to TOML', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    await user.click(screen.getByRole('button', { name: /speculative decoding/i }))

    expect(screen.getByRole('heading', { name: 'Speculative Decoding' })).toBeInTheDocument()
    expect(screen.queryByText('Speculative decoding defaults')).not.toBeInTheDocument()
    expect(screen.queryByText('Compatibility & fallback')).not.toBeInTheDocument()
    expect(screen.queryByText('Performance defaults')).not.toBeInTheDocument()
    expect(screen.queryByText('Observability')).not.toBeInTheDocument()
    expect(screen.queryByText('Enable speculative decoding by default')).not.toBeInTheDocument()
    expect(screen.queryByText('Require compatibility check')).not.toBeInTheDocument()
    expect(screen.getByText('Incompatible pairing behavior')).toBeInTheDocument()

    const modeControl = within(screen.getByRole('radiogroup', { name: 'Default speculation mode' }))
    const draftPolicyControl = within(screen.getByRole('radiogroup', { name: 'Default draft selection policy' }))
    const pairingBehaviorControl = within(screen.getByRole('radiogroup', { name: 'Incompatible pairing behavior' }))
    const defaultsPreview = screen.getByRole('complementary', { name: /\[defaults\]/i })
    expect(modeControl.getByRole('radio', { name: 'auto' })).toBeChecked()
    expect(modeControl.getByRole('radio', { name: 'disabled' })).toBeInTheDocument()
    expect(defaultsPreview).not.toHaveTextContent('pairing_fault')
    expect(defaultsPreview).not.toHaveTextContent('draft_selection_policy')

    await user.click(modeControl.getByRole('radio', { name: 'draft' }))
    expect(draftPolicyControl.getByRole('radio', { name: 'auto' })).not.toBeDisabled()
    await user.click(pairingBehaviorControl.getByRole('radio', { name: 'Fail launch' }))
    expect(pairingBehaviorControl.getByRole('radio', { name: 'Fail launch' })).toBeChecked()
    expect(defaultsPreview).toHaveTextContent('pairing_fault = "fail_closed"')
    fireEvent.change(screen.getByRole('slider', { name: 'Default draft max tokens' }), { target: { value: '32' } })

    await user.click(screen.getByRole('tab', { name: 'TOML Output' }))
    expect(getTomlSource().value).toContain('[defaults.speculative]')
    expect(getTomlSource().value).not.toContain('enabled =')
    expect(getTomlSource().value).toContain('mode = "draft"')
    expect(getTomlSource().value).toContain('draft_max_tokens = 32')
    expect(getTomlSource().value).toContain('pairing_fault = "fail_closed"')
    expect(getTomlSource().value).not.toContain('draft_selection_policy = "auto"')
    expect(getTomlSource().value).not.toMatch(/^pairing_behavior =/m)
    expect(getTomlSource().value).not.toContain('incompatible_pairing_behavior')
    expect(getTomlSource().value).not.toContain('model_runtime = "cuda"')
    expect(getTomlSource().value).not.toContain('[defaults.request_defaults]')
    expect(getTomlSource().value).not.toContain('temperature = 0.70')
    expect(getTomlSource().value).not.toContain('reasoning_format = "auto"')
    expect(getTomlSource().value).not.toContain('llama_flavor')
    expect(getTomlSource().value).not.toContain('allow_cpu_speculation')
    expect(getTomlSource().value).not.toContain('diagnostics =')
  })

  it('disables draft speculative decoding controls unless mode is draft', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage enableNavigationBlocker={false} />)

    await user.click(screen.getByRole('button', { name: /speculative decoding/i }))

    const modeControl = within(screen.getByRole('radiogroup', { name: 'Default speculation mode' }))
    const draftPolicyControl = within(screen.getByRole('radiogroup', { name: 'Default draft selection policy' }))
    const pairingBehaviorControl = within(screen.getByRole('radiogroup', { name: 'Incompatible pairing behavior' }))

    expect(screen.queryByRole('combobox', { name: 'Default draft selection policy' })).not.toBeInTheDocument()
    expect(screen.queryByRole('combobox', { name: 'Incompatible pairing behavior' })).not.toBeInTheDocument()
    expect(draftPolicyControl.queryByRole('radio', { name: 'Catalog recommended' })).not.toBeInTheDocument()
    expect(draftPolicyControl.queryByRole('radio', { name: 'Auto-detect' })).not.toBeInTheDocument()
    expect(draftPolicyControl.getByRole('radio', { name: 'auto' })).toBeChecked()
    expect(pairingBehaviorControl.getByRole('radio', { name: 'Warn & Disable' })).toBeChecked()
    expect(pairingBehaviorControl.getByRole('radio', { name: 'Fail launch' })).toBeInTheDocument()

    await user.click(modeControl.getByRole('radio', { name: 'disabled' }))

    expect(modeControl.getByRole('radio', { name: 'disabled' })).toBeChecked()
    expect(modeControl.getByRole('radio', { name: 'draft' })).not.toBeDisabled()
    expect(draftPolicyControl.getByRole('radio', { name: 'auto' })).toBeDisabled()
    expect(pairingBehaviorControl.getByRole('radio', { name: 'Warn & Disable' })).toBeDisabled()
    expect(pairingBehaviorControl.getByRole('radio', { name: 'Fail launch' })).toBeDisabled()
    expect(screen.getByRole('slider', { name: 'Default draft max tokens' })).toBeDisabled()
    expect(screen.getByRole('slider', { name: 'Default draft minimum tokens' })).toBeDisabled()
    expect(screen.queryByRole('slider', { name: 'Default draft acceptance threshold' })).not.toBeInTheDocument()
    expect(screen.queryByRole('radiogroup', { name: 'Allow CPU speculation' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /show advanced/i }))
    expect(screen.getByRole('slider', { name: 'Default draft acceptance threshold' })).toBeDisabled()

    await user.click(modeControl.getByRole('radio', { name: 'n-gram' }))
    expect(draftPolicyControl.getByRole('radio', { name: 'auto' })).toBeDisabled()
    expect(pairingBehaviorControl.getByRole('radio', { name: 'Warn & Disable' })).toBeDisabled()
    expect(screen.getByRole('slider', { name: 'Default draft max tokens' })).toBeDisabled()

    await user.click(modeControl.getByRole('radio', { name: 'draft' }))
    expect(draftPolicyControl.getByRole('radio', { name: 'auto' })).not.toBeDisabled()
    expect(pairingBehaviorControl.getByRole('radio', { name: 'Warn & Disable' })).not.toBeDisabled()
    expect(screen.getByRole('slider', { name: 'Default draft max tokens' })).not.toBeDisabled()
  })

  it('renders remote nodes as read-only context and keeps TOML in the output tab', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    expect(screen.getByRole('heading', { name: 'perseus.local' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'triton.lab' })).toBeInTheDocument()
    expect(screen.getByText('Peers')).toBeInTheDocument()
    expect(screen.getByText('read-only')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add model to perseus.local' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Add model to triton.lab' })).toBeDisabled()
    expect(screen.queryByRole('textbox', { name: /configuration toml source/i })).not.toBeInTheDocument()

    const tomlSource = await openTomlOutput(user)
    expect(tomlSource.value).toContain('version = 1')
    expect(tomlSource.value).toContain('model = "GLM-4.7-Flash-Q4_K_M"')
    expect(tomlSource.value).not.toContain('perseus.local')
    expect(tomlSource.value).not.toContain('triton.lab')
  })

  it('blocks remote VRAM chip and slot interactions from editing placement', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const carrackGpu3Capacity = within(getCarrackSection()).getAllByRole('region', {
      name: /rtx 6000 pro capacity/i
    })[2]
    if (!carrackGpu3Capacity) throw new Error('Expected carrack GPU 3 capacity region')

    await user.click(carrackGpu3Capacity)
    expect(carrackGpu3Capacity.closest('[data-config-container-selected="true"]')).toBeInTheDocument()

    const perseusSection = screen.getByRole('heading', { name: 'perseus.local' }).closest('section')
    if (!perseusSection) throw new Error('Expected perseus.local section')

    const remoteCapacity = within(perseusSection).getByRole('region', { name: /unified memory capacity/i })
    const remoteChip = within(remoteCapacity).getByRole('button', { name: /qwen3\.5-27b-ud-q4_k_xl, .* read-only/i })
    const remoteReservedLane = within(remoteCapacity).getByRole('button', { name: /system reserved space/i })

    expect(remoteChip).toBeDisabled()
    expect(remoteChip).toHaveAttribute('draggable', 'false')
    expect(remoteReservedLane).toBeDisabled()

    await user.click(remoteCapacity)
    expect(remoteCapacity.closest('[data-config-container-selected="true"]')).not.toBeInTheDocument()
    expect(carrackGpu3Capacity.closest('[data-config-container-selected="true"]')).toBeInTheDocument()

    await user.click(within(getCarrackSection()).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 3/i })).toBeInTheDocument()
  })

  it('uses the clicked GPU container as the catalog add target', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const gpu3Capacity = within(getCarrackSection()).getAllByRole('region', { name: /rtx 6000 pro capacity/i })[2]
    if (!gpu3Capacity) throw new Error('Expected carrack GPU 3 capacity region')

    await user.click(gpu3Capacity)

    const selectedGpu3Container = gpu3Capacity.closest('[data-config-container-selected="true"]')
    if (!(selectedGpu3Container instanceof HTMLElement))
      throw new Error('Expected clicked GPU 3 container to be selected')

    await user.click(within(getCarrackSection()).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 3/i })).toBeInTheDocument()
    expect(within(gpu3Capacity).getByRole('button', { name: /phi-4-mini/i })).toBeInTheDocument()
  })

  it('uses the arrow-key selected empty GPU slot as the catalog add target', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const gpu3Capacity = within(getCarrackSection()).getAllByRole('region', { name: /rtx 6000 pro capacity/i })[2]
    if (!gpu3Capacity) throw new Error('Expected carrack GPU 3 capacity region')

    await user.keyboard('{ArrowDown}{ArrowDown}')

    const selectedGpu3Container = gpu3Capacity.closest('[data-config-container-selected="true"]')
    if (!(selectedGpu3Container instanceof HTMLElement)) throw new Error('Expected arrow-key selected GPU 3 container')

    const modelSelectionEvent = await dispatchShortcut('ArrowRight')
    expect(modelSelectionEvent.defaultPrevented).toBe(true)
    expect(gpu3Capacity.closest('[data-config-container-selected="true"]')).toBe(selectedGpu3Container)
    expect(screen.queryByRole('button', { name: /^remove /i })).not.toBeInTheDocument()

    await user.keyboard('a')
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 3/i })).toBeInTheDocument()
    expect(within(gpu3Capacity).getByRole('button', { name: /phi-4-mini/i })).toBeInTheDocument()
  })

  it('keeps the current model selected when Tab has no other editable node target', async () => {
    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    expect(screen.getByRole('button', { name: /remove llama-3\.3-70b-q4_k_m from gpu 1/i })).toBeInTheDocument()

    const tabEvent = await dispatchShortcut('Tab')

    expect(tabEvent.defaultPrevented).toBe(true)
    expect(screen.getByRole('button', { name: /remove llama-3\.3-70b-q4_k_m from gpu 1/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /remove glm-4\.7-flash-q4_k_m from gpu 0/i })).not.toBeInTheDocument()
  })

  it('keeps keyboard edits scoped to the local node when remote assignments exist', async () => {
    const user = userEvent.setup()
    const data = {
      ...CONFIGURATION_HARNESS,
      assigns: [
        ...CONFIGURATION_HARNESS.assigns,
        { id: 'a6', modelId: 'phi4', nodeId: 'node-b', containerIdx: 0, ctx: 4096 }
      ],
      preferredAssignId: 'a2'
    }

    render(<ConfigurationPage initialTab="local-deployment" data={data} enableNavigationBlocker={false} />)

    await user.keyboard('{ArrowDown}')
    const contextEvent = await dispatchShortcut('ArrowRight', { altKey: true })

    expect(contextEvent.defaultPrevented).toBe(true)
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '17,408 ctx'
    )

    const tomlSource = await openTomlOutput(user)
    expect(tomlSource.value).not.toContain('phi-4-mini')
  })

  it('selects models within the current GPU slot with left and right arrows', async () => {
    const user = userEvent.setup()
    const data = {
      ...CONFIGURATION_HARNESS,
      assigns: [
        ...CONFIGURATION_HARNESS.assigns,
        { id: 'a6', modelId: 'phi4', nodeId: 'node-a', containerIdx: 2, ctx: 4096 }
      ],
      preferredAssignId: 'a3'
    }

    render(<ConfigurationPage initialTab="local-deployment" data={data} enableNavigationBlocker={false} />)

    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()

    await user.keyboard('{ArrowRight}')
    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 2/i })).toBeInTheDocument()

    await user.keyboard('{ArrowLeft}')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()

    await user.keyboard('{Shift>}{ArrowRight}{/Shift}')
    expect(screen.getByRole('button', { name: /remove phi-4-mini from gpu 2/i })).toBeInTheDocument()

    await user.keyboard('{Shift>}{ArrowLeft}{/Shift}')
    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()
  })

  it('selects models from reserved lane selection within the current GPU slot', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const gpu2Capacity = within(getCarrackSection()).getAllByRole('region', { name: /rtx 6000 pro capacity/i })[1]
    if (!gpu2Capacity) throw new Error('Expected carrack GPU 2 capacity region')

    await user.click(
      within(gpu2Capacity).getByRole('button', { name: /system reserved space, .* reserved on rtx 6000 pro/i })
    )
    expect(screen.queryByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).not.toBeInTheDocument()

    await user.keyboard('{ArrowRight}')

    expect(screen.getByRole('button', { name: /remove qwen3\.5-27b-q4_k_m from gpu 2/i })).toBeInTheDocument()
  })

  it('does not expose remote pooled placements as editable model buttons', () => {
    const data = {
      ...CONFIGURATION_HARNESS,
      assigns: [
        ...CONFIGURATION_HARNESS.assigns,
        { id: 'a6', modelId: 'phi4', nodeId: 'node-b', containerIdx: 0, ctx: 4096 }
      ],
      preferredAssignId: 'a5'
    }

    render(<ConfigurationPage initialTab="local-deployment" data={data} enableNavigationBlocker={false} />)

    expect(
      screen.queryByRole('button', { name: /remove qwen3\.5-27b-ud-q4_k_xl from perseus\.local pool/i })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: /remove phi-4-mini from perseus\.local pool/i })
    ).not.toBeInTheDocument()
    expect(screen.getByText('Qwen3.5-27B-UD-Q4_K_XL')).toBeInTheDocument()
    expect(screen.getByText('phi-4-mini')).toBeInTheDocument()
  })

  it('restores separate GPU assignments after previewing pooled placement', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const gpu2Capacity = within(getCarrackSection()).getAllByRole('region', { name: /rtx 6000 pro capacity/i })[1]
    const rtx3080Capacity = within(getCarrackSection()).getByRole('region', { name: /rtx 3080 capacity/i })
    if (!gpu2Capacity) throw new Error('Expected carrack GPU 2 capacity region')

    expect(within(gpu2Capacity).getByRole('button', { name: /qwen3\.5-27b-q4_k_m/i })).toBeInTheDocument()
    expect(within(rtx3080Capacity).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'separate' }))

    const restoredGpu2Capacity = within(getCarrackSection()).getAllByRole('region', {
      name: /rtx 6000 pro capacity/i
    })[1]
    const restoredRtx3080Capacity = within(getCarrackSection()).getByRole('region', { name: /rtx 3080 capacity/i })
    if (!restoredGpu2Capacity) throw new Error('Expected restored carrack GPU 2 capacity region')

    expect(within(restoredGpu2Capacity).getByRole('button', { name: /qwen3\.5-27b-q4_k_m/i })).toBeInTheDocument()
    expect(within(restoredRtx3080Capacity).getByRole('button', { name: /qwen3-4b-q4_k_m/i })).toBeInTheDocument()
  })

  it('saves dirty changes and reverts back to the last saved configuration', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const saveButton = screen.getByRole('button', { name: /save config/i })
    const revertButton = screen.getByRole('button', { name: /revert/i })

    expect(saveButton).toHaveAttribute('aria-keyshortcuts', 'Control+S')
    expect(revertButton).toHaveAttribute('aria-keyshortcuts', 'Control+X')
    expect(saveButton).toBeDisabled()
    expect(saveButton).toHaveAttribute('title', 'No changes to save')

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    expect(saveButton).toBeEnabled()

    const saveEvent = await dispatchShortcut('s', { ctrlKey: true })
    expect(saveEvent.defaultPrevented).toBe(true)
    expect(saveButton).toBeDisabled()
    expect(saveButton).toHaveAttribute('title', 'No changes to save')

    await user.click(within(getCarrackSection()).getByRole('button', { name: 'Add model to carrack' }))
    await user.type(screen.getByRole('textbox', { name: 'Command bar search' }), 'phi')
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1))
    await user.keyboard('{Enter}')
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Model catalog' })).not.toBeInTheDocument())

    expect(screen.getByRole('button', { name: /phi-4-mini, .* weights/i })).toBeInTheDocument()
    expect(saveButton).toBeEnabled()
    await openTomlOutput(user)
    expect(countTomlOccurrences('[models.hardware]')).toBe(0)

    const revertEvent = await dispatchShortcut('x', { ctrlKey: true })
    expect(revertEvent.defaultPrevented).toBe(true)
    expect(saveButton).toBeDisabled()
    expect(countTomlOccurrences('[models.hardware]')).toBe(0)
    expect(screen.queryByRole('button', { name: /phi-4-mini, .* weights/i })).not.toBeInTheDocument()
  })

  it('tracks configuration history with Ctrl+Z and Ctrl+R', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const undoButton = screen.getByRole('button', { name: /undo/i })
    const redoButton = screen.getByRole('button', { name: /redo/i })

    expect(undoButton).toHaveAttribute('aria-keyshortcuts', 'Control+Z')
    expect(redoButton).toHaveAttribute('aria-keyshortcuts', 'Control+R')

    await user.keyboard('{ArrowDown}')
    const contextEvent = await dispatchShortcut('ArrowRight', { altKey: true })
    expect(contextEvent.defaultPrevented).toBe(true)
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '17,408 ctx'
    )
    expect(undoButton).toBeEnabled()

    const undoEvent = await dispatchShortcut('z', { ctrlKey: true })
    expect(undoEvent.defaultPrevented).toBe(true)
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '16,384 ctx'
    )
    expect(redoButton).toBeEnabled()

    const redoEvent = await dispatchShortcut('r', { ctrlKey: true })
    expect(redoEvent.defaultPrevented).toBe(true)
    expect(screen.getByRole('button', { name: /qwen3\.5-27b-q4_k_m, 17\.4 gb weights/i })).toHaveTextContent(
      '17,408 ctx'
    )

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))
    await openTomlOutput(user)
    expect(countTomlOccurrences('[models.hardware]')).toBe(0)

    await dispatchShortcut('z', { ctrlKey: true })
    expect(countTomlOccurrences('[models.hardware]')).toBe(4)

    await dispatchShortcut('r', { ctrlKey: true })
    expect(countTomlOccurrences('[models.hardware]')).toBe(0)
  })

  it('does not consume plain s when the selected node is already separate', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    const shortcutEvent = await dispatchShortcut('s')

    expect(shortcutEvent.defaultPrevented).toBe(false)
    await openTomlOutput(user)
    expect(countTomlOccurrences('[models.hardware]')).toBe(4)
  })

  it('shows the navigation blocker only when enabled and there are unsaved changes', async () => {
    const user = userEvent.setup()

    render(<ConfigurationPage initialTab="local-deployment" />)

    expect(mockUseBlocker).toHaveBeenCalled()
    expect(screen.queryByRole('dialog', { name: 'Unsaved configuration' })).not.toBeInTheDocument()

    await user.click(within(getCarrackSection()).getByRole('radio', { name: 'pooled' }))

    expect(screen.getByRole('dialog', { name: 'Unsaved configuration' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Stay' }))
    expect(blockedBlocker.reset).toHaveBeenCalled()
  })

  it('skips navigation blocking when disabled', () => {
    render(<ConfigurationPage initialTab="local-deployment" enableNavigationBlocker={false} />)

    expect(mockUseBlocker).not.toHaveBeenCalled()
    expect(screen.queryByRole('dialog', { name: 'Unsaved configuration' })).not.toBeInTheDocument()
  })
})
