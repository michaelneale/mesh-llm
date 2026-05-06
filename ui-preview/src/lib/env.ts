export function normalizeRouterBasePath(value: string | undefined) {
  const trimmed = value?.trim()
  if (!trimmed || trimmed === '.' || trimmed === './') return '/'

  const withLeadingSlash = trimmed.startsWith('/') ? trimmed : `/${trimmed}`
  const withoutTrailingSlash = withLeadingSlash.replace(/\/+$/, '')
  return withoutTrailingSlash === '' ? '/' : withoutTrailingSlash
}

export function hrefWithBasePath(path: string, basePath = env.routerBasePath) {
  const normalizedBasePath = normalizeRouterBasePath(basePath)
  const normalizedPath = path.startsWith('/') ? path : `/${path}`

  if (normalizedBasePath === '/') return normalizedPath
  if (normalizedPath === '/') return `${normalizedBasePath}/`
  return `${normalizedBasePath}${normalizedPath}`
}

export function stripBasePath(pathname: string, basePath = env.routerBasePath) {
  const normalizedBasePath = normalizeRouterBasePath(basePath)
  if (normalizedBasePath === '/' || !pathname.startsWith(normalizedBasePath)) return pathname

  const strippedPathname = pathname.slice(normalizedBasePath.length)
  if (!strippedPathname) return '/'
  return strippedPathname.startsWith('/') ? strippedPathname : `/${strippedPathname}`
}

export const env = {
  appVersion: import.meta.env.VITE_APP_VERSION ?? 'dev',
  apiUrl: import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:9337',
  routerBasePath: normalizeRouterBasePath(import.meta.env.VITE_ROUTER_BASE_PATH ?? import.meta.env.BASE_URL),
  storageNamespace: import.meta.env.VITE_STORAGE_NAMESPACE ?? 'mesh-llm-ui-preview',
  isDevelopment: import.meta.env.DEV
}

export function isDevelopmentMode() {
  return env.isDevelopment
}
