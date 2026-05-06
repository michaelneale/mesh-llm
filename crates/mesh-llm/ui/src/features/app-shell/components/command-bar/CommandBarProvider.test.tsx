// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { CommandBarProvider } from './CommandBarProvider';
import { useCommandBar } from './useCommandBar';

function HookConsumerOutsideProvider() {
  useCommandBar();
  return null;
}

function CommandBarTestConsumer() {
  const { isOpen, activeModeId, openCommandBar, closeCommandBar } = useCommandBar();

  return (
    <div>
      <div data-testid="command-bar-state">{isOpen ? 'open' : 'closed'}</div>
      <div data-testid="command-bar-mode">{activeModeId ?? 'none'}</div>
      <button type="button" onClick={() => openCommandBar('models')}>
        Open models
      </button>
      <button type="button" onClick={closeCommandBar}>
        Close
      </button>
      <button type="button">Primary trigger</button>
      <button type="button">Secondary target</button>
      <input aria-label="Editable target" />
    </div>
  );
}

function renderProvider() {
  return render(
    <CommandBarProvider>
      <CommandBarTestConsumer />
    </CommandBarProvider>,
  );
}

  describe('CommandBarProvider', () => {
   const originalUserAgent = navigator.userAgent;

   beforeEach(() => {
     Object.defineProperty(navigator, 'userAgent', {
       configurable: true,
       value: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0)',
     });
   });

   afterEach(() => {
     Object.defineProperty(navigator, 'userAgent', {
       configurable: true,
       value: originalUserAgent,
     });
     cleanup();
   });

   it('throws a clear error when the hook is used outside the provider', () => {
     const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

     expect(() => render(<HookConsumerOutsideProvider />)).toThrow(
       'useCommandBar must be used within a CommandBarProvider.',
     );

     consoleError.mockRestore();
   });

   it('closes on Escape and restores focus to the previously focused element', () => {
    renderProvider();

    const primaryTrigger = screen.getByRole('button', { name: 'Primary trigger' });
    const secondaryTarget = screen.getByRole('button', { name: 'Secondary target' });
    const openButton = screen.getByRole('button', { name: 'Open models' });

    primaryTrigger.focus();
    fireEvent.click(openButton);
    secondaryTarget.focus();
    fireEvent.keyDown(window, { key: 'Escape' });

    expect(screen.getByTestId('command-bar-state')).toHaveTextContent('closed');
    expect(screen.getByTestId('command-bar-mode')).toHaveTextContent('none');
    expect(primaryTrigger).toHaveFocus();
  });
});
