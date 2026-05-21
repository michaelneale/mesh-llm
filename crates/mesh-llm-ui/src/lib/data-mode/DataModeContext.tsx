import { useEffect, useState, type ReactNode, type SetStateAction } from 'react'
import { env } from '@/lib/env'
import { DataModeContext, type DataMode } from '@/lib/data-mode/data-mode-context'

export const DATA_MODE_STORAGE_KEY = `${env.storageNamespace}:data-mode:v1`

export type DataModeProviderProps = {
  children: ReactNode
  initialMode?: DataMode
  persist?: boolean
  storageKey?: string
}

function isDataMode(value: unknown): value is DataMode {
  return value === 'live' || value === 'harness'
}

function readStoredDataMode(storageKey: string, fallbackMode: DataMode, persist: boolean): DataMode {
  if (!persist || typeof window === 'undefined') return fallbackMode

  try {
    const storedValue = window.localStorage.getItem(storageKey)
    return isDataMode(storedValue) ? storedValue : fallbackMode
  } catch {
    return fallbackMode
  }
}

function writeStoredDataMode(storageKey: string, mode: DataMode, persist: boolean): void {
  if (!persist || typeof window === 'undefined') return

  try {
    window.localStorage.setItem(storageKey, mode)
  } catch {
    return
  }
}

// Default data mode resolution.
//
// Production bundles (fly app, embedded UI in the shipped `mesh-llm` binary,
// `just build` / `just bundle`, anything built with `vite build`) default to
// `'live'` so a fresh visitor sees real mesh state — not fixture providers
// labelled "Vast.ai" / "RunPod" on the Reserves page or harness data on
// Chat / Configuration. That regression (#615) was caused by the previous
// hard-coded `'harness'` default leaking into production after the
// swap-ui-preview commit.
//
// Dev builds (`npm run dev` via vite) still default to `'harness'` so
// designers iterating on mockups don't have to flip the in-app toggle each
// reload. The toggle is persisted in localStorage per-origin, so any user
// who explicitly picks a mode keeps it.
//
// Tests and the developer playground can pin a specific mode by passing
// `initialMode="harness"` (or `"live"`) explicitly.
function defaultInitialMode(): DataMode {
  return env.isDevelopment ? 'harness' : 'live'
}

export function DataModeProvider({
  children,
  initialMode = defaultInitialMode(),
  persist = true,
  storageKey = DATA_MODE_STORAGE_KEY
}: DataModeProviderProps) {
  const [mode, setModeState] = useState<DataMode>(() => readStoredDataMode(storageKey, initialMode, persist))

  useEffect(() => {
    writeStoredDataMode(storageKey, mode, persist)
  }, [mode, persist, storageKey])

  const setMode = (nextMode: SetStateAction<DataMode>) => {
    setModeState((currentMode) => {
      const resolvedMode = typeof nextMode === 'function' ? nextMode(currentMode) : nextMode
      return isDataMode(resolvedMode) ? resolvedMode : currentMode
    })
  }

  return <DataModeContext.Provider value={{ mode, setMode }}>{children}</DataModeContext.Provider>
}
