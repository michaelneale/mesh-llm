import { useEffect } from 'react'
import { QueryClient, useQueryClient } from '@tanstack/react-query'
import {
  HeadContent,
  Outlet,
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter
} from '@tanstack/react-router'
import { act, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { AppProviders } from '@/app/providers/AppProviders'
import type { ConfigurationTabId } from '@/features/configuration/components/configuration-tab-ids'
import { ConfigurationRoutePage } from '@/features/configuration/pages/ConfigurationRoutePage'
import { ChatPageContent } from '@/features/chat/pages/ChatPage'
import { DeveloperPlaygroundPage } from '@/features/developer/pages/DeveloperPlaygroundPage'
import { DashboardPageSurface } from '@/features/network/pages/DashboardPage'
import { parseDeveloperPlaygroundSearch } from '@/features/developer/playground/developer-playground-tabs'
import { ReservesPageContent } from '@/features/reserves/pages/ReservesPage'
import { statusKeys } from '@/lib/query/query-keys'

const routeCacheProbe = vi.hoisted(() => ({
  dashboardClient: undefined as QueryClient | undefined,
  chatClient: undefined as QueryClient | undefined
}))

vi.mock('@/features/reserves/pages/ReservesPage', () => ({
  ReservesPageContent: () => <div>Reserves route</div>
}))

vi.mock('@/features/developer/pages/DeveloperPlaygroundPage', async () => {
  const router = await vi.importActual<typeof import('@tanstack/react-router')>('@tanstack/react-router')

  return {
    DeveloperPlaygroundPage: () => {
      const { tab } = router.useSearch({ from: '/__playground' })

      return <div>Active developer route tab: {tab}</div>
    }
  }
})

vi.mock('@/features/configuration/pages/ConfigurationPage', () => ({
  ConfigurationPageContent: ({ activeTab }: { activeTab: ConfigurationTabId }) => (
    <div>Active route tab: {activeTab}</div>
  )
}))

vi.mock('@/features/network/pages/DashboardPage', () => ({
  DashboardPageSurface: () => {
    const queryClient = useQueryClient()

    useEffect(() => {
      routeCacheProbe.dashboardClient = queryClient
      queryClient.setQueryData(statusKeys.detail(), { source: 'dashboard-cache' })
    }, [queryClient])

    return <div>Dashboard route</div>
  }
}))

vi.mock('@/features/chat/pages/ChatPage', () => ({
  ChatPageContent: () => {
    const queryClient = useQueryClient()
    routeCacheProbe.chatClient = queryClient
    const cachedStatus = queryClient.getQueryData<{ source: string }>(statusKeys.detail())

    return <div>Chat route cache: {cachedStatus?.source ?? 'missing'}</div>
  }
}))

vi.mock('@/lib/feature-flags', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/feature-flags')>()

  return {
    ...actual,
    useBooleanFeatureFlag: () => true
  }
})

function TestRootLayout() {
  return (
    <>
      <HeadContent />
      <Outlet />
    </>
  )
}

const rootRoute = createRootRoute({ component: TestRootLayout })
const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  head: () => ({ meta: [{ title: 'MeshLLM - Dashboard' }] }),
  component: DashboardPageSurface
})
const chatRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/chat',
  head: () => ({ meta: [{ title: 'MeshLLM - Chat' }] }),
  component: ChatPageContent
})
const reservesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/reserves',
  head: () => ({ meta: [{ title: 'MeshLLM - Reserves' }] }),
  component: ReservesPageContent
})
const configurationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/configuration',
  head: () => ({ meta: [{ title: 'MeshLLM - Configuration' }] }),
  component: ConfigurationRoutePage
})
const configurationTabRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/configuration/$configurationTab',
  head: () => ({ meta: [{ title: 'MeshLLM - Configuration' }] }),
  component: ConfigurationRoutePage
})
const developerPlaygroundRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/__playground',
  head: () => ({ meta: [{ title: 'MeshLLM - Developer Playground' }] }),
  validateSearch: parseDeveloperPlaygroundSearch,
  component: DeveloperPlaygroundPage
})
const testRouteTree = rootRoute.addChildren([
  indexRoute,
  reservesRoute,
  chatRoute,
  configurationRoute,
  configurationTabRoute,
  developerPlaygroundRoute
])

function renderRouterAt(pathname: string, queryClient = new QueryClient()) {
  return renderRouterWithHistory(createMemoryHistory({ initialEntries: [pathname] }), queryClient)
}

function renderRouterWithHistory(history: ReturnType<typeof createMemoryHistory>, queryClient = new QueryClient()) {
  const testRouter = createRouter({
    history,
    routeTree: testRouteTree
  })

  render(
    <AppProviders initialDataMode="harness" persistDataMode={false} queryClient={queryClient}>
      <RouterProvider router={testRouter} />
    </AppProviders>
  )

  return testRouter
}

describe('app router routes', () => {
  it.each([
    ['/', 'MeshLLM - Dashboard', 'Dashboard route'],
    ['/reserves', 'MeshLLM - Reserves', 'Reserves route'],
    ['/chat', 'MeshLLM - Chat', 'Chat route cache: missing'],
    ['/configuration/defaults', 'MeshLLM - Configuration', 'Active route tab: defaults'],
    ['/__playground?tab=shell-controls', 'MeshLLM - Developer Playground', 'Active developer route tab: shell-controls']
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
      initialIndex: 2
    })
    const testRouter = renderRouterWithHistory(history)

    await screen.findByText('Active route tab: defaults')

    await act(async () => {
      history.back()
    })

    await screen.findByText('Active developer route tab: chat-components')
    await waitFor(() => expect(testRouter.state.location.pathname).toBe('/__playground'))
    expect(testRouter.state.location.search).toMatchObject({ tab: 'chat-components' })
  })

  it('reuses the shared query cache when navigating from dashboard to chat', async () => {
    const testRouter = renderRouterAt('/')

    await screen.findByText('Dashboard route')

    await act(async () => {
      await testRouter.navigate({ to: '/chat' })
    })

    await screen.findByText('Chat route cache: dashboard-cache')
    expect(routeCacheProbe.chatClient).toBe(routeCacheProbe.dashboardClient)
  })
})
