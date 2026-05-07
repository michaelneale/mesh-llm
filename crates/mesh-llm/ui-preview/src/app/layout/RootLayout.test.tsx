import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AppProviders } from '@/app/providers/AppProviders'
import { RootLayout } from '@/app/layout/RootLayout'

const routerState = vi.hoisted(() => ({ pathname: '/' }))
const navigateSpy = vi.hoisted(() => vi.fn())
const useStatusStreamSpy = vi.hoisted(() => vi.fn())
const useStatusQuerySpy = vi.hoisted(() => vi.fn())
const topNavSpy = vi.hoisted(() => vi.fn())
const EventSourceStub = vi.hoisted(() => vi.fn())

vi.mock('@tanstack/react-router', () => ({
  HeadContent: () => null,
  Outlet: () => <div>Route outlet</div>,
  useRouter: () => ({ navigate: navigateSpy }),
  useRouterState: ({ select }: { select: (state: { location: { pathname: string } }) => string }) =>
    select({ location: { pathname: routerState.pathname } })
}))

vi.mock('@/features/network/api/use-status-stream', () => ({
  useStatusStream: useStatusStreamSpy
}))

vi.mock('@/features/network/api/use-status-query', () => ({
  useStatusQuery: useStatusQuerySpy
}))

vi.mock('@/features/shell/components/TopNav', () => ({
  TopNav: (props: unknown) => {
    topNavSpy(props)
    return <div>Top nav</div>
  }
}))

vi.mock('@/features/shell/components/PreferencesPanel', () => ({
  PreferencesPanel: () => null
}))

vi.mock('@/features/shell/components/Footer', () => ({
  Footer: () => <div>Footer</div>
}))

vi.mock('@/features/shell/hooks/useUiPreferences', () => ({
  useUIPreferences: () => ({
    theme: 'dark',
    accent: 'blue',
    density: 'comfortable',
    panelStyle: 'solid',
    setTheme: vi.fn(),
    setAccent: vi.fn(),
    setDensity: vi.fn(),
    setPanelStyle: vi.fn()
  })
}))

vi.mock('@/lib/feature-flags', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/feature-flags')>()

  return {
    ...actual,
    useBooleanFeatureFlag: () => true
  }
})

function renderRootLayout(initialDataMode: 'harness' | 'live') {
  render(
    <AppProviders initialDataMode={initialDataMode} persistDataMode={false}>
      <RootLayout />
    </AppProviders>
  )
}

describe('RootLayout', () => {
  beforeEach(() => {
    routerState.pathname = '/'
    navigateSpy.mockReset()
    useStatusStreamSpy.mockReset()
    useStatusQuerySpy.mockReset()
    topNavSpy.mockReset()
    useStatusQuerySpy.mockReturnValue({ data: undefined })
    vi.stubGlobal('EventSource', EventSourceStub)
  })

  it('does not start the live status stream in harness mode', () => {
    renderRootLayout('harness')

    expect(screen.getByText('Top nav')).toBeInTheDocument()
    expect(useStatusStreamSpy).toHaveBeenCalledWith({ enabled: false })
  })

  it('starts the shared live status stream in live mode', () => {
    renderRootLayout('live')

    expect(useStatusStreamSpy).toHaveBeenCalledWith({ enabled: true })
  })

  it('passes live status-backed invite rows while keeping the configured API target', () => {
    useStatusQuerySpy.mockReturnValue({
      data: {
        node_id: 'node-1',
        node_state: 'serving',
        model_name: 'Qwen-Test',
        peers: [],
        models: [],
        my_vram_gb: 24,
        api_port: 3131,
        gpus: [],
        serving_models: [],
        hostname: 'mesh.local',
        token: 'invite-token-123'
      }
    })

    renderRootLayout('live')

    expect(topNavSpy).toHaveBeenCalled()
    expect(topNavSpy.mock.calls.at(-1)?.[0]).toEqual(
      expect.objectContaining({
        apiUrl: 'http://127.0.0.1:3131/v1',
        apiTargetLiveness: 'live',
        joinCommands: expect.arrayContaining([
          expect.objectContaining({ label: 'Invite token', value: 'invite-token-123' }),
          expect.objectContaining({
            label: 'Auto join and serve command',
            value: 'mesh-llm --auto --join invite-token-123'
          }),
          expect.objectContaining({
            label: 'Client-only join command',
            value: 'mesh-llm client --join invite-token-123'
          })
        ])
      })
    )
  })

  it('does not replace the configured API target with a public mesh node id', () => {
    useStatusQuerySpy.mockReturnValue({
      data: {
        node_id: '16ce0bb4de',
        node_state: 'client',
        model_name: '(client)',
        peers: [],
        models: [],
        my_vram_gb: 0,
        api_port: 9337,
        gpus: [],
        serving_models: [],
        my_hostname: '6834941b7eede8',
        token: 'invite-token-123'
      }
    })

    renderRootLayout('live')

    expect(topNavSpy.mock.calls.at(-1)?.[0]).toEqual(
      expect.objectContaining({
        apiUrl: 'http://127.0.0.1:9337/v1',
        apiTargetLiveness: 'live'
      })
    )
  })

  it('falls back to the placeholder invite token when live status has not reported one yet', () => {
    useStatusQuerySpy.mockReturnValue({
      data: {
        node_id: 'node-1',
        node_state: 'serving',
        model_name: 'Qwen-Test',
        peers: [],
        models: [],
        my_vram_gb: 24,
        api_port: 3131,
        gpus: [],
        serving_models: [],
        hostname: 'mesh.local'
      }
    })

    renderRootLayout('live')

    expect(topNavSpy.mock.calls.at(-1)?.[0]).toEqual(
      expect.objectContaining({
        apiUrl: 'http://127.0.0.1:3131/v1',
        apiTargetLiveness: 'live',
        joinCommands: expect.arrayContaining([
          expect.objectContaining({ label: 'Invite token', value: 'Invite token unavailable', disabled: true }),
          expect.objectContaining({
            label: 'Auto join and serve command',
            value: 'Auto join command unavailable',
            disabled: true
          }),
          expect.objectContaining({
            label: 'Client-only join command',
            value: 'Client-only join command unavailable',
            disabled: true
          })
        ])
      })
    )
  })

  it('passes unavailable API target liveness when live status cannot be fetched', () => {
    useStatusQuerySpy.mockReturnValue({ data: undefined, isError: true })

    renderRootLayout('live')

    expect(topNavSpy.mock.calls.at(-1)?.[0]).toEqual(
      expect.objectContaining({
        apiTargetLiveness: 'unavailable'
      })
    )
  })
})
