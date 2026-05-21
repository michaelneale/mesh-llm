import { createContext } from 'react'
import type { SetStateAction } from 'react'

export type DataMode = 'live' | 'harness'

export type DataModeContextValue = {
  mode: DataMode
  setMode: (mode: SetStateAction<DataMode>) => void
}

// Fallback used only if a component reads `useDataMode` outside a
// `DataModeProvider` (production always mounts the provider). Match the
// provider default so the un-wrapped fallback doesn't silently flip
// surfaces into harness/fixture mode.
export const DataModeContext = createContext<DataModeContextValue>({
  mode: 'live',
  setMode: () => undefined
})
