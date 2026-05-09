import { useEffect, useMemo, useState } from 'react'

export type ResolvedTheme = 'light' | 'dark'
export type ThemeMode = ResolvedTheme | 'auto'

export function readResolvedTheme(mode: ThemeMode = 'auto'): ResolvedTheme {
  if (mode === 'light' || mode === 'dark') return mode
  if (typeof document === 'undefined') return 'light'
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

export function useResolvedTheme(mode: ThemeMode = 'auto') {
  const [resolvedTheme, setResolvedTheme] = useState<ResolvedTheme>(() => readResolvedTheme(mode))

  useEffect(() => {
    if (typeof document === 'undefined') return

    const root = document.documentElement
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const updateTheme = () => setResolvedTheme(readResolvedTheme(mode))

    // Initial sync when effect runs - safe as this is subscription setup
    requestAnimationFrame(updateTheme)
    const observer = new MutationObserver(updateTheme)
    observer.observe(root, { attributes: true, attributeFilter: ['class'] })
    media.addEventListener('change', updateTheme)

    return () => {
      observer.disconnect()
      media.removeEventListener('change', updateTheme)
    }
  }, [mode])

  const explicitMode = useMemo(() => (mode === 'light' || mode === 'dark' ? mode : null), [mode])
  return explicitMode ?? resolvedTheme
}
