import { readFileSync } from 'node:fs'
import path from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'mesh-llm:preferences:v1'
const bootstrapScript = extractBootstrapScript()

function extractBootstrapScript(): string {
  const indexHtml = readFileSync(path.join(process.cwd(), 'index.html'), 'utf8')
  const match = indexHtml.match(/<script>\s*([\s\S]*?)\s*<\/script>/)
  if (!match) throw new Error('Unable to find the inline preference bootstrap script')
  return match[1]
}

function installMatchMediaMock(prefersDark: boolean) {
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    writable: true,
    value: vi.fn((query: string) => ({
      matches: query === '(prefers-color-scheme: dark)' ? prefersDark : false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

function runBootstrap({ prefersDark, storedPreferences }: { prefersDark: boolean; storedPreferences?: object }) {
  document.head.innerHTML = ''
  document.documentElement.removeAttribute('data-theme')
  document.documentElement.removeAttribute('data-theme-preference')
  document.documentElement.removeAttribute('data-accent')
  document.documentElement.removeAttribute('data-density')
  document.documentElement.removeAttribute('data-panel-style')
  window.localStorage.clear()
  installMatchMediaMock(prefersDark)

  if (storedPreferences) {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(storedPreferences))
  }

  Function(bootstrapScript)()
}

function runtimeThemeColorMeta() {
  return document.querySelector<HTMLMetaElement>('meta[name="theme-color"][data-mesh-theme-color="runtime"]')
}

describe('pre-hydration preferences bootstrap', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('defaults auto dark systems to soft panel edges before React hydrates', () => {
    runBootstrap({ prefersDark: true })

    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(document.documentElement.dataset.themePreference).toBe('auto')
    expect(document.documentElement.dataset.panelStyle).toBe('soft')
    expect(document.documentElement.style.getPropertyValue('--initial-color-scheme')).toBe('dark')
    expect(runtimeThemeColorMeta()?.content).toBe('#171b24')
  })

  it('defaults auto light systems to solid panel edges before React hydrates', () => {
    runBootstrap({ prefersDark: false })

    expect(document.documentElement.dataset.theme).toBe('light')
    expect(document.documentElement.dataset.themePreference).toBe('auto')
    expect(document.documentElement.dataset.panelStyle).toBe('solid')
    expect(document.documentElement.style.getPropertyValue('--initial-color-scheme')).toBe('light')
    expect(runtimeThemeColorMeta()?.content).toBe('#fbfaf7')
  })

  it('treats legacy solid panel preferences as theme defaults', () => {
    runBootstrap({
      prefersDark: true,
      storedPreferences: { theme: 'auto', accent: 'blue', density: 'normal', panelStyle: 'solid' }
    })

    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(document.documentElement.dataset.themePreference).toBe('auto')
    expect(document.documentElement.dataset.panelStyle).toBe('soft')
  })

  it('preserves legacy soft panel preferences as explicit overrides', () => {
    runBootstrap({
      prefersDark: false,
      storedPreferences: { theme: 'auto', accent: 'blue', density: 'normal', panelStyle: 'soft' }
    })

    expect(document.documentElement.dataset.theme).toBe('light')
    expect(document.documentElement.dataset.themePreference).toBe('auto')
    expect(document.documentElement.dataset.panelStyle).toBe('soft')
  })

  it('honors current non-overridden light preferences as solid panel edges', () => {
    runBootstrap({
      prefersDark: true,
      storedPreferences: {
        theme: 'light',
        accent: 'blue',
        density: 'normal',
        panelStyle: 'soft',
        panelStyleOverride: false
      }
    })

    expect(document.documentElement.dataset.theme).toBe('light')
    expect(document.documentElement.dataset.themePreference).toBe('light')
    expect(document.documentElement.dataset.panelStyle).toBe('solid')
  })

  it('persists current explicit panel overrides across resolved themes', () => {
    runBootstrap({
      prefersDark: true,
      storedPreferences: {
        theme: 'auto',
        accent: 'blue',
        density: 'normal',
        panelStyle: 'solid',
        panelStyleOverride: true
      }
    })

    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(document.documentElement.dataset.themePreference).toBe('auto')
    expect(document.documentElement.dataset.panelStyle).toBe('solid')
  })
})
