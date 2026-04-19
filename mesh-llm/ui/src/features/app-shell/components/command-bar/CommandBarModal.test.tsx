// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { CommandBarMode, CommandBarResultContainerProps, CommandBarResultItemProps } from './command-bar-types';
import { CommandBarModal } from './CommandBarModal';
import { CommandBarProvider } from './CommandBarProvider';
import { useCommandBar } from './useCommandBar';

type Item = { id: string; name: string };

function ModelsIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" {...props}>
      <circle cx="8" cy="8" r="7" />
    </svg>
  );
}

function NodesIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" {...props}>
      <rect x="2" y="2" width="12" height="12" />
    </svg>
  );
}

function OpenCommandBarButton() {
  const { openCommandBar } = useCommandBar();

  return (
    <button type="button" onClick={() => openCommandBar()}>
      Open command bar
    </button>
  );
}

function OpenExplicitModeButton({ label, modeId }: { label: string; modeId: string }) {
  const { openCommandBar } = useCommandBar();

  return (
    <button type="button" onClick={() => openCommandBar(modeId)}>
      {label}
    </button>
  );
}

function createMode(
  id: string,
  label: string,
  source: CommandBarMode<Item>['source'],
  overrides: Partial<CommandBarMode<Item>> = {},
): CommandBarMode<Item> {
  return {
    id,
    label,
    leadingIcon: id === 'models' ? ModelsIcon : NodesIcon,
    source,
    getItemKey: (item) => item.id,
    getSearchText: (item) => item.name,
    onSelect: () => {},
    ...overrides,
  };
}

function ScrollableResultContainer({ children }: CommandBarResultContainerProps<Item>) {
  return <div className="max-h-24 overflow-y-auto">{children}</div>;
}

function renderModal({
  behavior = 'distinct',
  defaultModeId = 'models',
  modes = [
    createMode('models', 'Models', [{ id: 'alpha', name: 'Alpha model' }]),
    createMode('nodes', 'Nodes', [{ id: 'beta', name: 'Beta node' }]),
  ],
  interstitial,
}: {
  behavior?: 'distinct' | 'combined';
  defaultModeId?: string | null;
  modes?: readonly CommandBarMode<Item>[];
  interstitial?: React.ReactNode;
} = {}) {
  return render(
    <CommandBarProvider>
      <OpenCommandBarButton />
      <OpenExplicitModeButton label="Open models command bar" modeId="models" />
      <CommandBarModal modes={modes} behavior={behavior} defaultModeId={defaultModeId} interstitial={interstitial} />
    </CommandBarProvider>,
  );
}

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;

  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });

  return { promise, resolve, reject };
}

function createRequestKey(modeId: string, query: string) {
  return `${modeId}:${query}`;
}

async function flushAsyncWork() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function runAllTimers() {
  await act(async () => {
    await vi.runAllTimersAsync();
  });
}

async function advanceTimersByTime(milliseconds: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(milliseconds);
  });
}

describe('CommandBarModal', () => {
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
    vi.useRealTimers();
    cleanup();
  });

  it('focuses the textbox immediately on open and shows a bounded mode shortcut hint', async () => {
    renderModal({
      behavior: 'distinct',
      defaultModeId: 'models',
    });

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    await screen.findByRole('dialog');

    const input = screen.getByRole('textbox', { name: 'Command bar search' });

    await waitFor(() => expect(input).toHaveFocus());
    expect(screen.getByTestId('command-bar-leading-icon')).toHaveAttribute('data-mode-id', 'models');
    expect(screen.getByText('Ctrl+1-2')).toBeInTheDocument();
  });

  it('renders a clear button inside the search field when query text is present and clears on click', async () => {
    renderModal({
      behavior: 'distinct',
      defaultModeId: 'models',
    });

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    const input = await screen.findByRole('textbox', { name: 'Command bar search' });
    fireEvent.change(input, { target: { value: 'qwen' } });

    const clearButton = screen.getByRole('button', { name: 'Clear search' });
    expect(clearButton).toBeInTheDocument();

    fireEvent.click(clearButton);

    expect(input).toHaveValue('');
    expect(input).toHaveFocus();
    expect(screen.queryByRole('button', { name: 'Clear search' })).not.toBeInTheDocument();
  });

  it('renders an optional interstitial slot between the header block and results and omits it by default', async () => {
    renderModal();

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    await screen.findByRole('dialog');
    expect(screen.queryByTestId('command-bar-interstitial')).not.toBeInTheDocument();

    cleanup();

    renderModal({ interstitial: <div>Filter controls</div> });

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    await screen.findByRole('dialog');

    const header = screen.getByTestId('command-bar-header');
    const interstitial = screen.getByTestId('command-bar-interstitial');
    const results = screen.getByTestId('command-bar-results');

    expect(interstitial).toHaveTextContent('Filter controls');
    expect(header.compareDocumentPosition(interstitial) & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0);
    expect(interstitial.compareDocumentPosition(results) & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0);
  });

  it('shows mode chips and shortcut hint only for distinct multi-mode configs', async () => {
    renderModal({
      behavior: 'distinct',
      modes: [createMode('models', 'Models', [{ id: 'alpha', name: 'Alpha model' }])],
    });

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    const dialog = await screen.findByRole('dialog');
    expect(within(dialog).queryByRole('button', { name: /Models Ctrl\+1/i })).not.toBeInTheDocument();
    expect(within(dialog).queryByText('Modes')).not.toBeInTheDocument();
    expect(within(dialog).queryByText('Ctrl+1-1')).not.toBeInTheDocument();

    cleanup();

    renderModal({
      behavior: 'distinct',
      modes: [
        createMode('models', 'Models', [{ id: 'alpha', name: 'Alpha model' }]),
        createMode('nodes', 'Nodes', [{ id: 'beta', name: 'Beta node' }]),
      ],
    });

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    const multiModeDialog = await screen.findByRole('dialog');
    const modelsChip = within(multiModeDialog).getByRole('button', { name: /Models Ctrl\+1/i });
    const nodesChip = within(multiModeDialog).getByRole('button', { name: /Nodes Ctrl\+2/i });

    expect(modelsChip).toHaveAttribute('aria-pressed', 'true');
    expect(nodesChip).toHaveAttribute('aria-pressed', 'false');
    expect(within(multiModeDialog).getByText('Ctrl+1-2')).toBeInTheDocument();
  });

  it('renders combined rows with per-row mode badges inside a shell-owned listbox', async () => {
    function CustomResultItem({ item, selected, query, modeLabel }: CommandBarResultItemProps<Item>) {
      return (
        <div data-testid={`custom-item-${item.id}`} data-selected={selected} data-query={query} data-mode={modeLabel}>
          {item.name}
        </div>
      );
    }

    renderModal({
      behavior: 'combined',
      modes: [
        createMode(
          'models',
          'Models',
          [
            { id: 'alpha', name: 'Alpha model' },
            { id: 'gamma', name: 'Gamma model' },
          ],
          { ResultItem: CustomResultItem },
        ),
        createMode('nodes', 'Nodes', [{ id: 'beta', name: 'Beta node' }], { ResultItem: CustomResultItem }),
      ],
    });

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    const listbox = await screen.findByRole('listbox', { name: 'Command bar results' });
    const options = within(listbox).getAllByRole('option');

    expect(options).toHaveLength(3);
    expect(screen.queryByRole('button', { name: /Models Ctrl\+1/i })).not.toBeInTheDocument();
    expect(within(listbox).getByTestId('custom-item-alpha')).toHaveAttribute('data-mode', 'Models');
    expect(within(listbox).getByTestId('custom-item-gamma')).toHaveAttribute('data-query', '');
    expect(within(listbox).getByTestId('custom-item-beta')).toHaveAttribute('data-selected', 'false');
    expect(within(options[0]).getByText('Models')).toBeInTheDocument();
    expect(within(options[1]).getByText('Models')).toBeInTheDocument();
    expect(within(options[2]).getByText('Nodes')).toBeInTheDocument();
  });

  it('supports keyboard-only navigation with clamped arrows, hover selection, and enter-driven success close', async () => {
    const onSelect = vi.fn();

    renderModal({
      behavior: 'distinct',
      modes: [
        createMode(
          'models',
          'Models',
          [
            { id: 'alpha', name: 'Alpha model' },
            { id: 'beta', name: 'Beta model' },
            { id: 'gamma', name: 'Gamma model' },
        ],
        { ResultContainer: ScrollableResultContainer, onSelect },
          ),
        ],
      });

      fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

      const input = await screen.findByRole('textbox', { name: 'Command bar search' });
      const listbox = screen.getByRole('listbox', { name: 'Command bar results' });

    await waitFor(() => expect(input).toHaveFocus());

    const getOptions = () => within(listbox).getAllByRole('option');

    expect(getOptions()[0]).toHaveAttribute('aria-selected', 'true');

    fireEvent.keyDown(input, { key: 'ArrowDown' });
    expect(getOptions()[1]).toHaveAttribute('aria-selected', 'true');

    fireEvent.keyDown(input, { key: 'ArrowDown' });
    expect(getOptions()[2]).toHaveAttribute('aria-selected', 'true');

    fireEvent.keyDown(input, { key: 'ArrowDown' });
    expect(getOptions()[2]).toHaveAttribute('aria-selected', 'true');

    fireEvent.keyDown(input, { key: 'ArrowUp' });
    expect(getOptions()[1]).toHaveAttribute('aria-selected', 'true');

    fireEvent.keyDown(input, { key: 'ArrowUp' });
    expect(getOptions()[0]).toHaveAttribute('aria-selected', 'true');

    fireEvent.keyDown(input, { key: 'ArrowUp' });
    expect(getOptions()[0]).toHaveAttribute('aria-selected', 'true');

    fireEvent.pointerMove(getOptions()[1]);
    expect(getOptions()[1]).toHaveAttribute('aria-selected', 'true');

    fireEvent.change(input, { target: { value: 'beta' } });
    await waitFor(() => expect(within(listbox).getAllByRole('option')).toHaveLength(1));

    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => expect(onSelect).toHaveBeenCalledTimes(1));
    expect(onSelect).toHaveBeenCalledWith({ id: 'beta', name: 'Beta model' });
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    const reopenedInput = await screen.findByRole('textbox', { name: 'Command bar search' });
    expect(screen.getByTestId('command-bar-leading-icon')).toHaveAttribute('data-mode-id', 'models');
    expect(reopenedInput).toHaveValue('');
  });

  it('scrolls the keyboard-active option into view when arrow navigation reaches an item below the visible list area', async () => {
    const prototypeScrollIntoView = HTMLElement.prototype.scrollIntoView;
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: vi.fn(),
    });

    try {
      renderModal({
        behavior: 'distinct',
        modes: [
          createMode(
            'models',
            'Models',
            Array.from({ length: 12 }, (_, index) => ({
              id: `item-${index + 1}`,
              name: `Item ${String(index + 1).padStart(2, '0')}`,
            })),
            { ResultContainer: ScrollableResultContainer },
          ),
        ],
      });

      fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

      const input = await screen.findByRole('textbox', { name: 'Command bar search' });
      const listbox = screen.getByRole('listbox', { name: 'Command bar results' });

      await waitFor(() => expect(input).toHaveFocus());

      const targetOption = within(listbox).getByRole('option', { name: 'Item 08' });
      const targetScrollIntoView = vi.fn();

      Object.defineProperty(targetOption, 'scrollIntoView', {
        configurable: true,
        value: targetScrollIntoView,
      });

      for (let index = 0; index < 7; index += 1) {
        fireEvent.keyDown(input, { key: 'ArrowDown' });
      }

      expect(targetOption).toHaveAttribute('aria-selected', 'true');
      expect(targetScrollIntoView).toHaveBeenCalledWith({ block: 'nearest' });
    } finally {
      if (prototypeScrollIntoView) {
        Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
          configurable: true,
          value: prototypeScrollIntoView,
        });
      } else {
        Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView');
      }
    }
  });

  it('opens an explicitly requested modeId when provided', async () => {
    renderModal();

    fireEvent.click(screen.getByRole('button', { name: 'Open models command bar' }));

    const input = await screen.findByRole('textbox', { name: 'Command bar search' });

    expect(screen.getByTestId('command-bar-leading-icon')).toHaveAttribute('data-mode-id', 'models');
    expect(input).toHaveFocus();
  });

  it('switches distinct modes with Ctrl digit shortcuts and ignores missing targets', async () => {
    renderModal({
      behavior: 'distinct',
      modes: [
        createMode('models', 'Models', [{ id: 'alpha', name: 'Alpha model' }]),
        createMode('nodes', 'Nodes', [{ id: 'beta', name: 'Beta node' }]),
      ],
    });

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    const input = await screen.findByRole('textbox', { name: 'Command bar search' });
    const dialog = screen.getByRole('dialog');

    fireEvent.keyDown(input, { key: '2', ctrlKey: true });

    await waitFor(() => {
      expect(within(dialog).getByRole('button', { name: /Nodes Ctrl\+2/i })).toHaveAttribute('aria-pressed', 'true');
    });
    expect(screen.getByTestId('command-bar-leading-icon')).toHaveAttribute('data-mode-id', 'nodes');
    expect(screen.getByText('Beta node')).toBeInTheDocument();
    expect(screen.getByRole('dialog')).toBeInTheDocument();

    fireEvent.keyDown(input, { key: '9', ctrlKey: true });

    expect(within(dialog).getByRole('button', { name: /Nodes Ctrl\+2/i })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it('closes on Escape through the provider flow and restores focus', async () => {
    renderModal();

    const openButton = screen.getByRole('button', { name: 'Open command bar' });
    openButton.focus();

    fireEvent.click(openButton);

    const input = await screen.findByRole('textbox', { name: 'Command bar search' });
    await waitFor(() => expect(input).toHaveFocus());

    fireEvent.keyDown(input, { key: 'Escape' });

    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(openButton).toHaveFocus();
  });

  it('keeps the modal open and shows an inline destructive error when selection rejects', async () => {
    const onSelect = vi.fn(async () => {
      throw new Error('Selection failed.');
    });

    renderModal({
      behavior: 'distinct',
      modes: [
        createMode(
          'models',
          'Models',
          [{ id: 'alpha', name: 'Alpha model' }],
          { onSelect },
        ),
      ],
    });

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    const input = await screen.findByRole('textbox', { name: 'Command bar search' });
    fireEvent.change(input, { target: { value: 'alpha' } });

    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => expect(onSelect).toHaveBeenCalledTimes(1));
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(input).toHaveValue('alpha');

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Action failed');
    expect(alert).toHaveTextContent('Selection failed.');
  });

  it('loads async results immediately on open and debounces query refetches by 150ms', async () => {
    vi.useFakeTimers();

    const source = vi
      .fn<(query: string, signal: AbortSignal) => Promise<readonly Item[]>>()
      .mockImplementationOnce(async () => [{ id: 'alpha', name: 'Alpha model' }])
      .mockImplementationOnce(async (query) => [{ id: `${query}-1`, name: `${query} result` }]);

    renderModal({
      behavior: 'distinct',
      modes: [createMode('models', 'Models', source)],
    });

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    expect(screen.getByRole('dialog')).toBeInTheDocument();

    expect(source).toHaveBeenCalledTimes(1);
    expect(source).toHaveBeenLastCalledWith('', expect.any(AbortSignal));
    expect(screen.getByText('Loading results')).toBeInTheDocument();

    await runAllTimers();
    await flushAsyncWork();

    expect(screen.getByText('Alpha model')).toBeInTheDocument();

    const input = screen.getByRole('textbox', { name: 'Command bar search' });
    fireEvent.change(input, { target: { value: 'beta' } });

    expect(source).toHaveBeenCalledTimes(1);

    await advanceTimersByTime(149);
    expect(source).toHaveBeenCalledTimes(1);

    await advanceTimersByTime(1);
    expect(source).toHaveBeenCalledTimes(2);
    expect(source).toHaveBeenLastCalledWith('beta', expect.any(AbortSignal));

    await runAllTimers();
    await flushAsyncWork();

    expect(screen.getByText('beta result')).toBeInTheDocument();
  });

  it('aborts in-flight async requests when the modal closes or the distinct mode changes', async () => {
    const firstSignals: AbortSignal[] = [];
    const secondSignals: AbortSignal[] = [];

    renderModal({
      behavior: 'distinct',
      modes: [
        createMode('models', 'Models', (_query, signal) => {
          firstSignals.push(signal);
          return new Promise<readonly Item[]>(() => {});
        }),
        createMode('nodes', 'Nodes', (_query, signal) => {
          secondSignals.push(signal);
          return new Promise<readonly Item[]>(() => {});
        }),
      ],
    });

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    const input = await screen.findByRole('textbox', { name: 'Command bar search' });
    await waitFor(() => expect(firstSignals).toHaveLength(1));
    expect(firstSignals[0]?.aborted).toBe(false);

    fireEvent.keyDown(input, { key: '2', ctrlKey: true });

    await waitFor(() => expect(secondSignals).toHaveLength(1));
    expect(firstSignals[0]?.aborted).toBe(true);
    expect(secondSignals[0]?.aborted).toBe(false);

    fireEvent.keyDown(input, { key: 'Escape' });

    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(secondSignals[0]?.aborted).toBe(true);
  });

  it('ignores stale async responses and resets selection errors on query and mode changes', async () => {
    vi.useFakeTimers();

    const requests = new Map<string, ReturnType<typeof createDeferred<readonly Item[]>>>();
    const source = vi.fn<(query: string, signal: AbortSignal) => Promise<readonly Item[]>>((query) => {
      const deferred = createDeferred<readonly Item[]>();
      requests.set(query, deferred);
      return deferred.promise;
    });
    const rejectingSelect = vi.fn().mockRejectedValue(new Error('Selection failed.'));

    renderModal({
      behavior: 'distinct',
      modes: [
        createMode('models', 'Models', source, { onSelect: rejectingSelect }),
        createMode('nodes', 'Nodes', [{ id: 'node-1', name: 'Fast node' }]),
      ],
    });

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    const input = screen.getByRole('textbox', { name: 'Command bar search' });
    expect(source).toHaveBeenCalledWith('', expect.any(AbortSignal));

    requests.get('')?.resolve([{ id: 'alpha', name: 'Alpha model' }]);
    await runAllTimers();
    await flushAsyncWork();
    fireEvent.keyDown(input, { key: 'Enter' });
    await flushAsyncWork();

    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent('Selection failed.');

    fireEvent.change(input, { target: { value: 'slow' } });
    await advanceTimersByTime(150);
    expect(source).toHaveBeenCalledWith('slow', expect.any(AbortSignal));

    fireEvent.change(input, { target: { value: 'fast' } });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    await advanceTimersByTime(150);
    expect(source).toHaveBeenCalledWith('fast', expect.any(AbortSignal));

    requests.get('fast')?.resolve([{ id: 'fast', name: 'Fast result' }]);
    await runAllTimers();
    await flushAsyncWork();
    expect(screen.getByText('Fast result')).toBeInTheDocument();

    requests.get('slow')?.resolve([{ id: 'slow', name: 'Slow result' }]);
    await runAllTimers();
    await flushAsyncWork();
    expect(screen.queryByText('Slow result')).not.toBeInTheDocument();
    expect(screen.getByText('Fast result')).toBeInTheDocument();

    fireEvent.keyDown(input, { key: '2', ctrlKey: true });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByText('Fast node')).toBeInTheDocument();
  });

  it('keeps current combined async results from multiple modes and ignores stale older per-mode responses', async () => {
    vi.useFakeTimers();

    const requests = new Map<string, ReturnType<typeof createDeferred<readonly Item[]>>>();
    const createAsyncSource = (modeId: string) =>
      vi.fn<(query: string, signal: AbortSignal) => Promise<readonly Item[]>>((query) => {
        const deferred = createDeferred<readonly Item[]>();
        requests.set(createRequestKey(modeId, query), deferred);
        return deferred.promise;
      });

    const modelsSource = createAsyncSource('models');
    const nodesSource = createAsyncSource('nodes');

    renderModal({
      behavior: 'combined',
      modes: [
        createMode('models', 'Models', modelsSource),
        createMode('nodes', 'Nodes', nodesSource),
      ],
    });

    fireEvent.click(screen.getByRole('button', { name: 'Open command bar' }));

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(modelsSource).toHaveBeenCalledWith('', expect.any(AbortSignal));
    expect(nodesSource).toHaveBeenCalledWith('', expect.any(AbortSignal));

    requests.get(createRequestKey('nodes', ''))?.resolve([{ id: 'node-initial', name: 'Node initial' }]);
    await runAllTimers();
    await flushAsyncWork();
    expect(screen.getByText('Node initial')).toBeInTheDocument();
    expect(screen.queryByText('Model initial')).not.toBeInTheDocument();

    requests.get(createRequestKey('models', ''))?.resolve([{ id: 'model-initial', name: 'Model initial' }]);
    await runAllTimers();
    await flushAsyncWork();
    expect(screen.getByText('Node initial')).toBeInTheDocument();
    expect(screen.getByText('Model initial')).toBeInTheDocument();

    const input = screen.getByRole('textbox', { name: 'Command bar search' });
    fireEvent.change(input, { target: { value: 'slow' } });
    await advanceTimersByTime(150);
    expect(modelsSource).toHaveBeenCalledWith('slow', expect.any(AbortSignal));
    expect(nodesSource).toHaveBeenCalledWith('slow', expect.any(AbortSignal));

    fireEvent.change(input, { target: { value: 'fast' } });
    await advanceTimersByTime(150);
    expect(modelsSource).toHaveBeenCalledWith('fast', expect.any(AbortSignal));
    expect(nodesSource).toHaveBeenCalledWith('fast', expect.any(AbortSignal));

    requests.get(createRequestKey('nodes', 'fast'))?.resolve([{ id: 'node-fast', name: 'Node fast' }]);
    await runAllTimers();
    await flushAsyncWork();
    expect(screen.getByText('Node fast')).toBeInTheDocument();
    expect(screen.queryByText('Model fast')).not.toBeInTheDocument();

    requests.get(createRequestKey('models', 'fast'))?.resolve([{ id: 'model-fast', name: 'Model fast' }]);
    await runAllTimers();
    await flushAsyncWork();
    expect(screen.getByText('Node fast')).toBeInTheDocument();
    expect(screen.getByText('Model fast')).toBeInTheDocument();

    requests.get(createRequestKey('models', 'slow'))?.resolve([{ id: 'model-slow', name: 'Model slow' }]);
    requests.get(createRequestKey('nodes', 'slow'))?.resolve([{ id: 'node-slow', name: 'Node slow' }]);
    await runAllTimers();
    await flushAsyncWork();

    expect(screen.queryByText('Model slow')).not.toBeInTheDocument();
    expect(screen.queryByText('Node slow')).not.toBeInTheDocument();
    expect(screen.getByText('Node fast')).toBeInTheDocument();
    expect(screen.getByText('Model fast')).toBeInTheDocument();
  });
});
