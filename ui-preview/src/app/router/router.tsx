import { createRootRoute, createRoute, createRouter, lazyRouteComponent } from '@tanstack/react-router'
import { AppErrorBoundary, NotFoundRoute } from '@/app/error-boundaries/AppErrorBoundary'
import { FeatureErrorBoundary } from '@/app/error-boundaries/FeatureErrorBoundary'
import { RootLayout } from '@/app/layout/RootLayout'
import { parseDeveloperPlaygroundSearch } from '@/features/developer/playground/developer-playground-tabs'
import { env } from '@/lib/env'

const enableMeshVizPerfRoute = import.meta.env.DEV || import.meta.env.VITE_ENABLE_PERF_ROUTE === 'true'

const rootRoute = createRootRoute({
  component: RootLayout,
  errorComponent: AppErrorBoundary,
  notFoundComponent: NotFoundRoute,
})
const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  head: () => ({ meta: [{ title: 'MeshLLM - Dashboard' }] }),
  component: lazyRouteComponent(() => import('@/features/network/pages/DashboardPage'), 'DashboardPageSurface'),
  errorComponent: FeatureErrorBoundary,
})
const chatRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/chat',
  head: () => ({ meta: [{ title: 'MeshLLM - Chat' }] }),
  component: lazyRouteComponent(() => import('@/features/chat/pages/ChatPage'), 'ChatPageContent'),
  errorComponent: FeatureErrorBoundary,
})
const configurationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/configuration',
  head: () => ({ meta: [{ title: 'MeshLLM - Configuration' }] }),
  component: lazyRouteComponent(() => import('@/features/configuration/pages/ConfigurationRoutePage'), 'ConfigurationRoutePage'),
  errorComponent: FeatureErrorBoundary,
})
const configurationTabRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/configuration/$configurationTab',
  head: () => ({ meta: [{ title: 'MeshLLM - Configuration' }] }),
  component: lazyRouteComponent(() => import('@/features/configuration/pages/ConfigurationRoutePage'), 'ConfigurationRoutePage'),
  errorComponent: FeatureErrorBoundary,
})
const developerPlaygroundRoute = import.meta.env.DEV
  ? createRoute({
      getParentRoute: () => rootRoute,
      path: '/__playground',
      head: () => ({ meta: [{ title: 'MeshLLM - Developer Playground' }] }),
      validateSearch: parseDeveloperPlaygroundSearch,
      component: lazyRouteComponent(() => import('@/features/developer/pages/DeveloperPlaygroundPage'), 'DeveloperPlaygroundPage'),
    })
  : null
const meshVizPerfRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/__meshviz-perf',
  component: lazyRouteComponent(() => import('@/features/network/pages/MeshVizPerfPage'), 'MeshVizPerfPage'),
})
export const routeTree = rootRoute.addChildren([
  indexRoute,
  chatRoute,
  configurationRoute,
  configurationTabRoute,
  ...(developerPlaygroundRoute ? [developerPlaygroundRoute] : []),
  ...(enableMeshVizPerfRoute ? [meshVizPerfRoute] : []),
])
export const router = createRouter({ routeTree, basepath: env.routerBasePath })
declare module '@tanstack/react-router' { interface Register { router: typeof router } }
