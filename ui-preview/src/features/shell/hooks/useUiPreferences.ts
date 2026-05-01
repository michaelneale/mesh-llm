import { useCallback, useEffect, useState } from 'react'
import type { Accent, Density, PanelStyle, Theme } from '@/features/app-tabs/types'
import { APP_STORAGE_KEYS } from '@/features/app-tabs/data'

export type UiPreferences = {
  theme: Theme
  accent: Accent
  density: Density
  panelStyle: PanelStyle
}

export const UI_PREFERENCES_STORAGE_KEY = APP_STORAGE_KEYS.preferences
const UI_THEME_COLOR_META_SELECTOR = 'meta[name="theme-color"][data-mesh-theme-color="runtime"]'

export const UI_THEME_COLORS: Record<Theme, string> = {
  dark: '#171b24',
  light: '#fbfaf7',
}

export const DEFAULT_UI_PREFERENCES: UiPreferences = {
  theme: 'dark',
  accent: 'blue',
  density: 'normal',
  panelStyle: 'solid',
}

function isTheme(value: unknown): value is Theme {
  return value === 'light' || value === 'dark'
}

function isAccent(value: unknown): value is Accent {
  return value === 'blue' || value === 'cyan' || value === 'violet' || value === 'green' || value === 'amber' || value === 'pink'
}

function isDensity(value: unknown): value is Density {
  return value === 'compact' || value === 'normal' || value === 'sparse'
}

function isPanelStyle(value: unknown): value is PanelStyle {
  return value === 'solid' || value === 'soft'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function normalizeUiPreferences(value: unknown): UiPreferences {
  if (!isRecord(value)) return DEFAULT_UI_PREFERENCES
  return {
    theme: isTheme(value.theme) ? value.theme : DEFAULT_UI_PREFERENCES.theme,
    accent: isAccent(value.accent) ? value.accent : DEFAULT_UI_PREFERENCES.accent,
    density: isDensity(value.density) ? value.density : DEFAULT_UI_PREFERENCES.density,
    panelStyle: isPanelStyle(value.panelStyle)
      ? value.panelStyle
      : isPanelStyle(value.configPanelStyle) ? value.configPanelStyle : DEFAULT_UI_PREFERENCES.panelStyle,
  }
}

export function readStoredUiPreferences(): UiPreferences {
  if (typeof window === 'undefined') return DEFAULT_UI_PREFERENCES

  try {
    const storedValue = window.localStorage.getItem(UI_PREFERENCES_STORAGE_KEY)
    if (storedValue) return normalizeUiPreferences(JSON.parse(storedValue))

    return DEFAULT_UI_PREFERENCES
  } catch {
    return DEFAULT_UI_PREFERENCES
  }
}

function writeStoredUiPreferences(preferences: UiPreferences): void {
  if (typeof window === 'undefined') return

  try {
    window.localStorage.setItem(UI_PREFERENCES_STORAGE_KEY, JSON.stringify(preferences))
  } catch {
    return
  }
}

function applyThemeColor(theme: Theme): void {
  if (typeof document === 'undefined') return

  const existingMeta = document.querySelector<HTMLMetaElement>(UI_THEME_COLOR_META_SELECTOR)
  const themeColorMeta = existingMeta ?? document.createElement('meta')

  if (!existingMeta) {
    themeColorMeta.name = 'theme-color'
    themeColorMeta.dataset.meshThemeColor = 'runtime'
    document.head.appendChild(themeColorMeta)
  }

  themeColorMeta.content = UI_THEME_COLORS[theme]
}

function applyUiPreferencesToDocument(preferences: UiPreferences): void {
  if (typeof document === 'undefined') return

  const root = document.documentElement
  root.dataset.theme = preferences.theme
  root.dataset.accent = preferences.accent
  root.dataset.density = preferences.density
  root.dataset.panelStyle = preferences.panelStyle
  delete root.dataset.configPanelStyle
  applyThemeColor(preferences.theme)
}

export function useUIPreferences() {
  const [preferences, setPreferences] = useState<UiPreferences>(() => readStoredUiPreferences())

  useEffect(() => {
    applyUiPreferencesToDocument(preferences)
    writeStoredUiPreferences(preferences)
  }, [preferences])

  useEffect(() => {
    const handleStorage = (event: StorageEvent) => {
      if (event.key !== UI_PREFERENCES_STORAGE_KEY || event.storageArea !== window.localStorage) return
      if (!event.newValue) {
        setPreferences(DEFAULT_UI_PREFERENCES)
        return
      }

      try {
        setPreferences(normalizeUiPreferences(JSON.parse(event.newValue)))
      } catch {
        setPreferences(DEFAULT_UI_PREFERENCES)
      }
    }

    window.addEventListener('storage', handleStorage)
    return () => window.removeEventListener('storage', handleStorage)
  }, [])

  const setTheme = useCallback((theme: Theme) => {
    setPreferences((current) => ({ ...current, theme }))
  }, [])

  const toggleTheme = useCallback(() => {
    setPreferences((current) => ({ ...current, theme: current.theme === 'dark' ? 'light' : 'dark' }))
  }, [])

  const setAccent = useCallback((accent: Accent) => {
    setPreferences((current) => ({ ...current, accent }))
  }, [])

  const setDensity = useCallback((density: Density) => {
    setPreferences((current) => ({ ...current, density }))
  }, [])

  const setPanelStyle = useCallback((panelStyle: PanelStyle) => {
    setPreferences((current) => ({ ...current, panelStyle }))
  }, [])

  return {
    ...preferences,
    setTheme,
    toggleTheme,
    setAccent,
    setDensity,
    setPanelStyle,
  }
}
