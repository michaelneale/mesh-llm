import { useContext } from 'react'
import { DataModeContext, type DataModeContextValue } from '@/lib/data-mode/data-mode-context'

export function useDataMode(): DataModeContextValue {
  return useContext(DataModeContext)
}
