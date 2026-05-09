import { useContext } from 'react'
import { CommandBarContext } from '@/features/app-shell/components/command-bar/command-bar-context'

export function useCommandBar() {
  const context = useContext(CommandBarContext)

  if (!context) {
    throw new Error('useCommandBar must be used within a CommandBarProvider.')
  }

  return context
}
