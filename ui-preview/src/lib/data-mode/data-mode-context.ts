import { createContext } from 'react'
import type { SetStateAction } from 'react'

export type DataMode = 'live' | 'harness'

export type DataModeContextValue = {
  mode: DataMode
  setMode: (mode: SetStateAction<DataMode>) => void
}

export const DataModeContext = createContext<DataModeContextValue>({
  mode: 'harness',
  setMode: () => undefined,
})
