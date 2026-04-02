import React, { useEffect, useRef } from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { ConfigErrorBoundary, useAsyncError } from '../ConfigErrorBoundary';

function ThrowingChild(): JSX.Element {
  throw new Error('Test render explosion');
}

function AsyncRejecter(): JSX.Element {
  const throwAsyncError = useAsyncError();
  const triggerRef = useRef<() => void>();

  useEffect(() => {
    triggerRef.current = () => {
      throwAsyncError(new Error('Async promise rejection'));
    };
  }, [throwAsyncError]);

  useEffect(() => {
    const timer = setTimeout(() => {
      triggerRef.current?.();
    }, 100);
    return () => clearTimeout(timer);
  }, []);

  return <div data-testid="async-child">Async child</div>;
}

describe('ConfigErrorBoundary', () => {
  it('catches render error from child and shows recovery UI', () => {
    // Suppress React's console.error for the expected error boundary log
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      <ConfigErrorBoundary>
        <ThrowingChild />
      </ConfigErrorBoundary>,
    );

    expect(screen.getByTestId('error-boundary-message')).toBeVisible();
    expect(screen.getByText('Something went wrong')).toBeVisible();

    consoleSpy.mockRestore();
  });

  it('shows "Something went wrong" text with reload button', () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      <ConfigErrorBoundary>
        <ThrowingChild />
      </ConfigErrorBoundary>,
    );

    expect(screen.getByText('Something went wrong')).toBeVisible();
    expect(screen.getByRole('button', { name: /reload/i })).toBeVisible();

    consoleSpy.mockRestore();
  });

  it('renders children normally when no error occurs', () => {
    render(
      <ConfigErrorBoundary>
        <div data-testid="happy-child">All good</div>
      </ConfigErrorBoundary>,
    );

    expect(screen.getByTestId('happy-child')).toBeVisible();
    expect(screen.queryByTestId('error-boundary-message')).toBeNull();
  });

  it('catches unhandledrejection event with a non-Error reason', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      <ConfigErrorBoundary>
        <div data-testid="happy-child">All good</div>
      </ConfigErrorBoundary>,
    );

    const event = new PromiseRejectionEvent('unhandledrejection', {
      promise: Promise.resolve(),
      reason: 'raw string rejection',
    });
    await act(async () => {
      window.dispatchEvent(event);
    });

    await waitFor(
      () => {
        expect(screen.getByTestId('error-boundary-message')).toBeVisible();
      },
      { timeout: 3000 },
    );

    expect(screen.getByText('Something went wrong')).toBeVisible();

    consoleSpy.mockRestore();
  });

  it('catches async Promise rejection via useAsyncError hook', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      <ConfigErrorBoundary>
        <AsyncRejecter />
      </ConfigErrorBoundary>,
    );

    await waitFor(
      () => {
        expect(screen.getByTestId('error-boundary-message')).toBeVisible();
      },
      { timeout: 3000 },
    );

    expect(screen.getByText('Something went wrong')).toBeVisible();
    expect(screen.getByRole('button', { name: /reload/i })).toBeVisible();

    consoleSpy.mockRestore();
  });
});
