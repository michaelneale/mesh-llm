import { useContext } from 'react'
import { DataModeContext, type DataModeContextValue } from './data-mode-context'

export function useDataMode(): DataModeContextValue {
  return useContext(DataModeContext)
}
