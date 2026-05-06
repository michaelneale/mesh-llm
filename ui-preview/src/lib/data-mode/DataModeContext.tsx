import { useEffect, useState, type ReactNode, type SetStateAction } from 'react'
import { env } from '@/lib/env'
import { DataModeContext, type DataMode } from './data-mode-context'

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

export function DataModeProvider({
  children,
  initialMode = 'harness',
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
