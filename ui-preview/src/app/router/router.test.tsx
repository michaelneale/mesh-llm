import { RouterProvider, createMemoryHistory, createRouter } from '@tanstack/react-router'
import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { ConfigurationTabId } from '@/features/configuration/components/configuration-tab-ids'
import { routeTree } from '@/app/router/router'

vi.mock('@/features/developer/pages/DeveloperPlaygroundPage', async () => {
  const router = await vi.importActual<typeof import('@tanstack/react-router')>('@tanstack/react-router')

  return {
    DeveloperPlaygroundPage: () => {
      const { tab } = router.useSearch({ from: '/__playground' })

      return <div>Active developer route tab: {tab}</div>
    },
  }
})

vi.mock('@/features/configuration/pages/ConfigurationPage', () => ({
  ConfigurationPageContent: ({ activeTab }: { activeTab: ConfigurationTabId }) => <div>Active route tab: {activeTab}</div>,
}))

vi.mock('@/features/network/pages/DashboardPage', () => ({
  DashboardPageSurface: () => <div>Dashboard route</div>,
}))

vi.mock('@/features/chat/pages/ChatPage', () => ({
  ChatPageContent: () => <div>Chat route</div>,
}))

vi.mock('@/lib/feature-flags', () => ({
  useBooleanFeatureFlag: () => true,
}))

function renderRouterAt(pathname: string) {
  return renderRouterWithHistory(createMemoryHistory({ initialEntries: [pathname] }))
}

function renderRouterWithHistory(history: ReturnType<typeof createMemoryHistory>) {
  const testRouter = createRouter({
    history,
    routeTree,
  })

  render(<RouterProvider router={testRouter} />)

  return testRouter
}

describe('app router routes', () => {
  it.each([
    ['/', 'MeshLLM - Dashboard', 'Dashboard route'],
    ['/chat', 'MeshLLM - Chat', 'Chat route'],
    ['/configuration/defaults', 'MeshLLM - Configuration', 'Active route tab: defaults'],
    ['/__playground?tab=shell-controls', 'MeshLLM - Developer Playground', 'Active developer route tab: shell-controls'],
  ])('sets the document title for %s', async (pathname, title, routeText) => {
    renderRouterAt(pathname)

    await screen.findByText(routeText)
    await waitFor(() => expect(document.title).toBe(title))
  })

  it('canonicalizes the bare configuration route to the default tab path', async () => {
    const testRouter = renderRouterAt('/configuration')

    await screen.findByText('Active route tab: defaults')
    await waitFor(() => expect(testRouter.state.location.pathname).toBe('/configuration/defaults'))
  })

  it('restores a configuration tab from the path segment on initial load', async () => {
    const testRouter = renderRouterAt('/configuration/local-deployment')

    await screen.findByText('Active route tab: local-deployment')
    expect(testRouter.state.location.pathname).toBe('/configuration/local-deployment')
  })

  it('restores a developer playground tab from the search params on initial load', async () => {
    const testRouter = renderRouterAt('/__playground?tab=data-display')

    await screen.findByText('Active developer route tab: data-display')
    expect(testRouter.state.location.pathname).toBe('/__playground')
    expect(testRouter.state.location.search).toMatchObject({ tab: 'data-display' })
  })

  it('falls back to the default developer playground tab for unknown search params', async () => {
    const testRouter = renderRouterAt('/__playground?tab=missing-tab')

    await screen.findByText('Active developer route tab: shell-controls')
    expect(testRouter.state.location.search).toMatchObject({ tab: 'shell-controls' })
  })

  it('preserves the developer playground tab when browser back returns to the page', async () => {
    const history = createMemoryHistory({
      initialEntries: ['/', '/__playground?tab=chat-components', '/configuration/defaults'],
      initialIndex: 2,
    })
    const testRouter = renderRouterWithHistory(history)

    await screen.findByText('Active route tab: defaults')

    history.back()

    await screen.findByText('Active developer route tab: chat-components')
    await waitFor(() => expect(testRouter.state.location.pathname).toBe('/__playground'))
    expect(testRouter.state.location.search).toMatchObject({ tab: 'chat-components' })
  })
})
