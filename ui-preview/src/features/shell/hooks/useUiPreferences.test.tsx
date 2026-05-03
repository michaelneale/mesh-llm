import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  DEFAULT_UI_PREFERENCES,
  UI_PREFERENCES_STORAGE_KEY,
  UI_THEME_COLORS,
  useUIPreferences,
} from '@/features/shell/hooks/useUiPreferences'
import { env } from '@/lib/env'

const SYSTEM_THEME_QUERY = '(prefers-color-scheme: dark)'

type MatchMediaController = {
  setMatches: (matches: boolean) => void
}

function installMatchMediaMock(initialMatches: boolean): MatchMediaController {
  let matches = initialMatches
  const listeners = new Set<EventListenerOrEventListenerObject>()
  const legacyListeners = new Set<(this: MediaQueryList, event: MediaQueryListEvent) => void>()

  const mediaQueryList = {
    get matches() {
      return matches
    },
    media: SYSTEM_THEME_QUERY,
    onchange: null,
    addEventListener: (type: string, listener: EventListenerOrEventListenerObject | null) => {
      if (type === 'change' && listener) listeners.add(listener)
    },
    removeEventListener: (type: string, listener: EventListenerOrEventListenerObject | null) => {
      if (type === 'change' && listener) listeners.delete(listener)
    },
    addListener: (listener: (this: MediaQueryList, event: MediaQueryListEvent) => void) => {
      legacyListeners.add(listener)
    },
    removeListener: (listener: (this: MediaQueryList, event: MediaQueryListEvent) => void) => {
      legacyListeners.delete(listener)
    },
    dispatchEvent: (event: Event) => {
      Object.defineProperties(event, {
        matches: { value: matches },
        media: { value: SYSTEM_THEME_QUERY },
      })
      listeners.forEach((listener) => {
        if (typeof listener === 'function') {
          listener.call(mediaQueryList, event)
          return
        }
        listener.handleEvent(event)
      })
      legacyListeners.forEach((listener) => {
        listener.call(mediaQueryList, event as MediaQueryListEvent)
      })
      return true
    },
  } satisfies MediaQueryList

  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn((query: string) => {
      if (query !== SYSTEM_THEME_QUERY) throw new Error(`Unexpected media query: ${query}`)
      return mediaQueryList
    }),
  })

  return {
    setMatches(nextMatches: boolean) {
      matches = nextMatches
      mediaQueryList.dispatchEvent(new Event('change'))
    },
  }
}

function clearUiPreferenceAttributes() {
  document.documentElement.removeAttribute('data-theme')
  document.documentElement.removeAttribute('data-theme-preference')
  document.documentElement.removeAttribute('data-accent')
  document.documentElement.removeAttribute('data-density')
  document.documentElement.removeAttribute('data-panel-style')
  document.documentElement.removeAttribute('data-config-panel-style')
  document.querySelector('meta[name="theme-color"][data-mesh-theme-color="runtime"]')?.remove()
}

function runtimeThemeColorMeta() {
  return document.querySelector<HTMLMetaElement>('meta[name="theme-color"][data-mesh-theme-color="runtime"]')
}

describe('useUIPreferences', () => {
  let systemTheme: MatchMediaController

  beforeEach(() => {
    systemTheme = installMatchMediaMock(true)
    window.localStorage.clear()
    clearUiPreferenceAttributes()
  })

  it('falls back to defaults and writes them to the document', async () => {
    const { result } = renderHook(() => useUIPreferences())

    expect(result.current.theme).toBe(DEFAULT_UI_PREFERENCES.theme)
    expect(result.current.accent).toBe(DEFAULT_UI_PREFERENCES.accent)
    expect(result.current.density).toBe(DEFAULT_UI_PREFERENCES.density)
    expect(result.current.panelStyle).toBe(DEFAULT_UI_PREFERENCES.panelStyle)

    await waitFor(() => {
      expect(document.documentElement.dataset.theme).toBe('dark')
      expect(document.documentElement.dataset.themePreference).toBe(DEFAULT_UI_PREFERENCES.theme)
      expect(document.documentElement.dataset.panelStyle).toBe(DEFAULT_UI_PREFERENCES.panelStyle)
      expect(runtimeThemeColorMeta()?.content).toBe(UI_THEME_COLORS.dark)
      expect(window.localStorage.getItem(UI_PREFERENCES_STORAGE_KEY)).toBe(JSON.stringify(DEFAULT_UI_PREFERENCES))
    })
  })

  it('resolves auto from the system preference and tracks system changes', async () => {
    const { result } = renderHook(() => useUIPreferences())

    await waitFor(() => {
      expect(result.current.theme).toBe('auto')
      expect(result.current.resolvedTheme).toBe('dark')
      expect(document.documentElement.dataset.theme).toBe('dark')
      expect(runtimeThemeColorMeta()?.content).toBe(UI_THEME_COLORS.dark)
    })

    act(() => {
      systemTheme.setMatches(false)
    })

    await waitFor(() => {
      expect(result.current.theme).toBe('auto')
      expect(result.current.resolvedTheme).toBe('light')
      expect(document.documentElement.dataset.theme).toBe('light')
      expect(document.documentElement.dataset.themePreference).toBe('auto')
      expect(runtimeThemeColorMeta()?.content).toBe(UI_THEME_COLORS.light)
    })
  })

  it('keeps explicit theme choices when the system preference changes', async () => {
    const { result } = renderHook(() => useUIPreferences())

    act(() => {
      result.current.setTheme('light')
    })

    await waitFor(() => {
      expect(result.current.theme).toBe('light')
      expect(result.current.resolvedTheme).toBe('light')
      expect(document.documentElement.dataset.theme).toBe('light')
    })

    act(() => {
      systemTheme.setMatches(true)
    })

    await waitFor(() => {
      expect(result.current.theme).toBe('light')
      expect(result.current.resolvedTheme).toBe('light')
      expect(document.documentElement.dataset.theme).toBe('light')
      expect(document.documentElement.dataset.themePreference).toBe('light')
      expect(runtimeThemeColorMeta()?.content).toBe(UI_THEME_COLORS.light)
    })
  })

  it('uses an app-namespaced preferences storage key', () => {
    expect(UI_PREFERENCES_STORAGE_KEY).toBe(`${env.storageNamespace}:preferences:v1`)
  })

  it('hydrates from valid saved preferences', async () => {
    const savedPreferences = { theme: 'light', accent: 'pink', density: 'sparse', panelStyle: 'soft' }
    window.localStorage.setItem(UI_PREFERENCES_STORAGE_KEY, JSON.stringify(savedPreferences))

    const { result } = renderHook(() => useUIPreferences())

    expect(result.current.theme).toBe(savedPreferences.theme)
    expect(result.current.accent).toBe(savedPreferences.accent)
    expect(result.current.density).toBe(savedPreferences.density)
    expect(result.current.panelStyle).toBe(savedPreferences.panelStyle)

    await waitFor(() => {
      expect(document.documentElement.dataset.theme).toBe(savedPreferences.theme)
      expect(document.documentElement.dataset.themePreference).toBe(savedPreferences.theme)
      expect(document.documentElement.dataset.accent).toBe(savedPreferences.accent)
      expect(document.documentElement.dataset.density).toBe(savedPreferences.density)
      expect(document.documentElement.dataset.panelStyle).toBe(savedPreferences.panelStyle)
      expect(runtimeThemeColorMeta()?.content).toBe(UI_THEME_COLORS.light)
    })
  })

  it('ignores invalid saved preferences', () => {
    window.localStorage.setItem(UI_PREFERENCES_STORAGE_KEY, JSON.stringify({ theme: 'solarized', accent: 'blue', density: 'normal' }))

    const { result } = renderHook(() => useUIPreferences())

    expect(result.current.theme).toBe(DEFAULT_UI_PREFERENCES.theme)
    expect(result.current.accent).toBe('blue')
    expect(result.current.density).toBe('normal')
    expect(result.current.panelStyle).toBe(DEFAULT_UI_PREFERENCES.panelStyle)
  })

  it('preserves valid fields when saved preferences are partially invalid', () => {
    window.localStorage.setItem(UI_PREFERENCES_STORAGE_KEY, JSON.stringify({ theme: 'light', accent: 'unknown', density: 'sparse', panelStyle: 'soft' }))

    const { result } = renderHook(() => useUIPreferences())

    expect(result.current.theme).toBe('light')
    expect(result.current.accent).toBe(DEFAULT_UI_PREFERENCES.accent)
    expect(result.current.density).toBe('sparse')
    expect(result.current.panelStyle).toBe('soft')
  })

  it('fills missing saved preference fields from defaults', () => {
    window.localStorage.setItem(UI_PREFERENCES_STORAGE_KEY, JSON.stringify({ theme: 'light' }))

    const { result } = renderHook(() => useUIPreferences())

    expect(result.current.theme).toBe('light')
    expect(result.current.accent).toBe(DEFAULT_UI_PREFERENCES.accent)
    expect(result.current.density).toBe(DEFAULT_UI_PREFERENCES.density)
    expect(result.current.panelStyle).toBe(DEFAULT_UI_PREFERENCES.panelStyle)
  })

  it('persists preference updates', async () => {
    const { result } = renderHook(() => useUIPreferences())

    act(() => {
      result.current.setTheme('light')
      result.current.setAccent('violet')
      result.current.setDensity('compact')
      result.current.setPanelStyle('soft')
    })

    await waitFor(() => {
      expect(document.documentElement.dataset.theme).toBe('light')
      expect(document.documentElement.dataset.themePreference).toBe('light')
      expect(document.documentElement.dataset.accent).toBe('violet')
      expect(document.documentElement.dataset.density).toBe('compact')
      expect(document.documentElement.dataset.panelStyle).toBe('soft')
      expect(runtimeThemeColorMeta()?.content).toBe(UI_THEME_COLORS.light)
      expect(window.localStorage.getItem(UI_PREFERENCES_STORAGE_KEY)).toBe(JSON.stringify({ theme: 'light', accent: 'violet', density: 'compact', panelStyle: 'soft' }))
    })
  })

  it('syncs valid fields from storage events and defaults invalid fields', async () => {
    const { result } = renderHook(() => useUIPreferences())

    act(() => {
      window.dispatchEvent(new StorageEvent('storage', {
        key: UI_PREFERENCES_STORAGE_KEY,
        newValue: JSON.stringify({ theme: 'light', accent: 'unknown', density: 'compact', panelStyle: 'soft' }),
        storageArea: window.localStorage,
      }))
    })

    await waitFor(() => {
      expect(result.current.theme).toBe('light')
      expect(result.current.accent).toBe(DEFAULT_UI_PREFERENCES.accent)
      expect(result.current.density).toBe('compact')
      expect(result.current.panelStyle).toBe('soft')
      expect(document.documentElement.dataset.theme).toBe('light')
      expect(document.documentElement.dataset.themePreference).toBe('light')
      expect(document.documentElement.dataset.accent).toBe(DEFAULT_UI_PREFERENCES.accent)
      expect(document.documentElement.dataset.density).toBe('compact')
      expect(document.documentElement.dataset.panelStyle).toBe('soft')
      expect(runtimeThemeColorMeta()?.content).toBe(UI_THEME_COLORS.light)
    })
  })

  it('hydrates temporary configuration panel style preferences into the global panel style', () => {
    window.localStorage.setItem(UI_PREFERENCES_STORAGE_KEY, JSON.stringify({ theme: 'dark', accent: 'blue', density: 'normal', configPanelStyle: 'soft' }))

    const { result } = renderHook(() => useUIPreferences())

    expect(result.current.panelStyle).toBe('soft')
  })
})
