import { useContext } from 'react';

import { CommandBarContext } from './CommandBarProvider';

export function useCommandBar() {
  const context = useContext(CommandBarContext);

  if (!context) {
    throw new Error('useCommandBar must be used within a CommandBarProvider.');
  }

  return context;
}
