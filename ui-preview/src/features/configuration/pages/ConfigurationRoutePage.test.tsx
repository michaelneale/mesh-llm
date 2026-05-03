import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ConfigurationTabId } from '@/features/configuration/components/configuration-tab-ids'

const routerMocks = vi.hoisted(() => {
  const navigate = vi.fn()
  return {
    navigate,
    useNavigate: vi.fn(() => navigate),
    useParams: vi.fn(() => ({})),
  }
})

const featureFlagMocks = vi.hoisted(() => ({
  integrationsEnabled: false,
  newConfigurationPageEnabled: true,
  signingAttestationEnabled: false,
}))

vi.mock('@tanstack/react-router', () => ({
  Navigate: ({ params, replace, to }: { params: { configurationTab: ConfigurationTabId }; replace?: boolean; to: string }) => (
    <div data-replace={replace ? 'true' : 'false'} data-testid="redirect" data-to={to}>
      {params.configurationTab}
    </div>
  ),
  useNavigate: routerMocks.useNavigate,
  useParams: routerMocks.useParams,
}))

vi.mock('@/features/configuration/pages/ConfigurationPage', () => ({
  ConfigurationPageContent: ({ activeTab, onTabChange }: { activeTab: ConfigurationTabId; onTabChange: (tab: ConfigurationTabId) => void }) => (
    <button type="button" onClick={() => onTabChange('toml-review')}>
      Active tab: {activeTab}
    </button>
  ),
}))

vi.mock('@/lib/feature-flags', () => ({
  useBooleanFeatureFlag: vi.fn((path: string) => {
    if (path === 'configuration/integrations') return featureFlagMocks.integrationsEnabled
    if (path === 'configuration/signingAttestation') return featureFlagMocks.signingAttestationEnabled
    return featureFlagMocks.newConfigurationPageEnabled
  }),
}))

import { ConfigurationRoutePage } from '@/features/configuration/pages/ConfigurationRoutePage'

describe('ConfigurationRoutePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    featureFlagMocks.integrationsEnabled = false
    featureFlagMocks.newConfigurationPageEnabled = true
    featureFlagMocks.signingAttestationEnabled = false
    routerMocks.useNavigate.mockReturnValue(routerMocks.navigate)
    routerMocks.useParams.mockReturnValue({})
  })

  it('shows a gated message when the configuration feature is disabled', () => {
    featureFlagMocks.newConfigurationPageEnabled = false
    routerMocks.useParams.mockReturnValue({ configurationTab: 'defaults' })

    render(<ConfigurationRoutePage />)

    expect(screen.getByRole('heading', { name: 'Configuration is gated' })).toBeInTheDocument()
    expect(screen.getByText(/global\/newConfigurationPage/i)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /active tab/i })).not.toBeInTheDocument()
  })

  it('redirects the bare configuration route to the default tab path', () => {
    render(<ConfigurationRoutePage />)

    expect(screen.getByTestId('redirect')).toHaveAttribute('data-to', '/configuration/$configurationTab')
    expect(screen.getByTestId('redirect')).toHaveAttribute('data-replace', 'true')
    expect(screen.getByTestId('redirect')).toHaveTextContent('defaults')
  })

  it('restores a valid tab from the path segment', () => {
    routerMocks.useParams.mockReturnValue({ configurationTab: 'local-deployment' })

    render(<ConfigurationRoutePage />)

    expect(screen.getByRole('button', { name: 'Active tab: local-deployment' })).toBeInTheDocument()
  })

  it('navigates tab changes to the matching tab path', async () => {
    const user = userEvent.setup()
    routerMocks.useParams.mockReturnValue({ configurationTab: 'defaults' })

    render(<ConfigurationRoutePage />)

    await user.click(screen.getByRole('button', { name: 'Active tab: defaults' }))

    expect(routerMocks.navigate).toHaveBeenCalledWith({
      params: { configurationTab: 'toml-review' },
      replace: true,
      to: '/configuration/$configurationTab',
    })
  })

  it('redirects gated temporary section paths back to defaults', () => {
    routerMocks.useParams.mockReturnValue({ configurationTab: 'signing' })

    const { rerender } = render(<ConfigurationRoutePage />)

    expect(screen.getByTestId('redirect')).toHaveTextContent('defaults')

    routerMocks.useParams.mockReturnValue({ configurationTab: 'integrations' })
    rerender(<ConfigurationRoutePage />)

    expect(screen.getByTestId('redirect')).toHaveTextContent('defaults')
  })

  it('restores gated temporary section paths when their flags are enabled', () => {
    featureFlagMocks.integrationsEnabled = true
    featureFlagMocks.signingAttestationEnabled = true
    routerMocks.useParams.mockReturnValue({ configurationTab: 'signing' })

    const { rerender } = render(<ConfigurationRoutePage />)

    expect(screen.getByRole('button', { name: 'Active tab: signing' })).toBeInTheDocument()

    routerMocks.useParams.mockReturnValue({ configurationTab: 'integrations' })
    rerender(<ConfigurationRoutePage />)

    expect(screen.getByRole('button', { name: 'Active tab: integrations' })).toBeInTheDocument()
  })

  it('redirects unknown tab paths back to the default tab path', () => {
    routerMocks.useParams.mockReturnValue({ configurationTab: 'missing-tab' })

    render(<ConfigurationRoutePage />)

    expect(screen.getByTestId('redirect')).toHaveTextContent('defaults')
  })
})
