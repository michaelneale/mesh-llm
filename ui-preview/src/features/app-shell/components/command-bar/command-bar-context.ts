import { createContext, type Dispatch, type SetStateAction } from 'react'

export type CommandBarContextValue = {
  isOpen: boolean
  query: string
  setQuery: Dispatch<SetStateAction<string>>
  activeModeId: string | null
  setActiveModeId: Dispatch<SetStateAction<string | null>>
  activeIndex: number
  setActiveIndex: Dispatch<SetStateAction<number>>
  selectionError: string | null
  setSelectionError: Dispatch<SetStateAction<string | null>>
  returnFocusElement: HTMLElement | null
  openCommandBar: (modeId?: string) => void
  closeCommandBar: () => void
  toggleCommandBar: (modeId?: string) => void
}

export const CommandBarContext = createContext<CommandBarContextValue | null>(null)
