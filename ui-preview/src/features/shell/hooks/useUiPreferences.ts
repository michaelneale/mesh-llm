import { useCallback, useEffect, useState } from 'react'
import type { Accent, Density, PanelStyle, ResolvedTheme, Theme } from '@/features/app-tabs/types'
import { APP_STORAGE_KEYS } from '@/features/app-tabs/data'

export type UiPreferences = {
  theme: Theme
  accent: Accent
  density: Density
  panelStyle: PanelStyle
}

export const UI_PREFERENCES_STORAGE_KEY = APP_STORAGE_KEYS.preferences
const UI_THEME_COLOR_META_SELECTOR = 'meta[name="theme-color"][data-mesh-theme-color="runtime"]'

export const UI_THEME_COLORS: Record<ResolvedTheme, string> = {
  dark: '#171b24',
  light: '#fbfaf7'
}

export const DEFAULT_UI_PREFERENCES: UiPreferences = {
  theme: 'auto',
  accent: 'blue',
  density: 'normal',
  panelStyle: 'solid'
}

type LegacyMediaQueryList = {
  addListener: (listener: () => void) => void
  removeListener: (listener: () => void) => void
}

function isTheme(value: unknown): value is Theme {
  return value === 'auto' || value === 'light' || value === 'dark'
}

function isAccent(value: unknown): value is Accent {
  return (
    value === 'blue' ||
    value === 'cyan' ||
    value === 'violet' ||
    value === 'green' ||
    value === 'amber' ||
    value === 'pink'
  )
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
      : isPanelStyle(value.configPanelStyle)
        ? value.configPanelStyle
        : DEFAULT_UI_PREFERENCES.panelStyle
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

function readSystemTheme(): ResolvedTheme {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return 'light'
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function resolveThemePreference(theme: Theme, systemTheme: ResolvedTheme = readSystemTheme()): ResolvedTheme {
  return theme === 'auto' ? systemTheme : theme
}

function addSystemThemeListener(mediaQuery: MediaQueryList, listener: () => void): () => void {
  if (typeof mediaQuery.addEventListener === 'function') {
    mediaQuery.addEventListener('change', listener)
    return () => mediaQuery.removeEventListener('change', listener)
  }

  const legacyMediaQuery: LegacyMediaQueryList = mediaQuery
  legacyMediaQuery.addListener(listener)
  return () => legacyMediaQuery.removeListener(listener)
}

function applyThemeColor(theme: ResolvedTheme): void {
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

function applyUiPreferencesToDocument(preferences: UiPreferences, resolvedTheme: ResolvedTheme): void {
  if (typeof document === 'undefined') return

  const root = document.documentElement
  root.dataset.theme = resolvedTheme
  root.dataset.themePreference = preferences.theme
  root.dataset.accent = preferences.accent
  root.dataset.density = preferences.density
  root.dataset.panelStyle = preferences.panelStyle
  delete root.dataset.configPanelStyle
  applyThemeColor(resolvedTheme)
}

export function useUIPreferences() {
  const [preferences, setPreferences] = useState<UiPreferences>(() => readStoredUiPreferences())
  const [systemTheme, setSystemTheme] = useState<ResolvedTheme>(() => readSystemTheme())
  const resolvedTheme = resolveThemePreference(preferences.theme, systemTheme)

  useEffect(() => {
    applyUiPreferencesToDocument(preferences, resolvedTheme)
    writeStoredUiPreferences(preferences)
  }, [preferences, resolvedTheme])

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return undefined

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const syncSystemTheme = () => setSystemTheme(mediaQuery.matches ? 'dark' : 'light')

    syncSystemTheme()
    return addSystemThemeListener(mediaQuery, syncSystemTheme)
  }, [])

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
    resolvedTheme,
    setTheme,
    setAccent,
    setDensity,
    setPanelStyle
  }
}
