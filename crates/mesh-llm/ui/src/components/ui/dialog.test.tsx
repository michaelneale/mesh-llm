import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { Dialog, DialogContent, DialogDescription, DialogTitle } from './dialog';

describe('Dialog', () => {
  it('renders a centered dialog with accessible title and description', () => {
    render(
      <Dialog open>
        <DialogContent>
          <DialogTitle>Search commands</DialogTitle>
          <DialogDescription>Find actions quickly.</DialogDescription>
          <button type="button">Run</button>
        </DialogContent>
      </Dialog>,
    );

    const dialog = screen.getByRole('dialog', { name: 'Search commands' });

    expect(dialog).toBeInTheDocument();
    expect(screen.getByText('Find actions quickly.')).toBeInTheDocument();
    expect(document.querySelector('[class*="backdrop-blur-sm"]')).toBeInTheDocument();
    expect(dialog).toHaveClass('left-1/2', 'top-1/2', 'rounded-xl', 'border-border/70', 'bg-card');
  });
});
