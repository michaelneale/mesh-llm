import { Loader2, Search, X } from 'lucide-react';
import { type ElementType, type KeyboardEvent as ReactKeyboardEvent, type ReactNode, useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';

import { Alert, AlertDescription, AlertTitle } from '../../../../components/ui/alert';
import { Badge } from '../../../../components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '../../../../components/ui/dialog';
import { Input } from '../../../../components/ui/input';
import { cn } from '../../../../lib/utils';
import {
  filterCommandBarResults,
  resolveCommandBarActiveIndex,
  resolveCommandBarOpenModeId,
  type CommandBarResolvedMode,
} from './command-bar-helpers';
import type {
  CommandBarBehavior,
  CommandBarMode,
  CommandBarNormalizedResult,
  CommandBarResultContainerProps,
} from './command-bar-types';
import { useCommandBar } from './useCommandBar';

export interface CommandBarModalProps<T> {
  modes: readonly CommandBarMode<T>[];
  behavior: CommandBarBehavior;
  defaultModeId?: string | null;
  fallbackIcon?: ElementType;
  title?: string;
  description?: string;
  placeholder?: string;
  emptyMessage?: string;
  interstitial?: ReactNode;
}

function DefaultResultContainer<T>({ children }: CommandBarResultContainerProps<T>) {
  return <div className="divide-y divide-border/60">{children}</div>;
}

function getCommandBarOptionId(listboxId: string, compositeKey: string) {
  return `${listboxId}-${compositeKey}`;
}

function getCommandBarSelectionErrorMessage(error: unknown) {
  if (error instanceof Error && error.message.trim().length > 0) return error.message;
  if (typeof error === 'string' && error.trim().length > 0) return error;
  return 'Could not complete that action.';
}

function isCommandBarAsyncSource<T>(source: CommandBarMode<T>['source']): source is Exclude<CommandBarMode<T>['source'], readonly T[]> {
  return typeof source === 'function';
}

function resolveCommandBarModeSource<T>(
  mode: CommandBarMode<T>,
  asyncItemsByModeId: Record<string, readonly T[]>,
): readonly T[] {
  if (isCommandBarAsyncSource(mode.source)) {
    return asyncItemsByModeId[mode.id] ?? [];
  }

  return mode.source;
}

function CommandBarOptionRow<T>({
  behavior,
  isActive,
  listboxId,
  onClick,
  onPointerMove,
  optionRef,
  result,
  renderItem,
}: {
  behavior: CommandBarBehavior;
  isActive: boolean;
  listboxId: string;
  onClick: () => void;
  onPointerMove: () => void;
  optionRef?: (node: HTMLDivElement | null) => void;
  result: CommandBarNormalizedResult<T>;
  renderItem: (result: CommandBarNormalizedResult<T>, isActive: boolean) => React.ReactNode;
}) {
  return (
    <div
      ref={optionRef}
      id={getCommandBarOptionId(listboxId, result.compositeKey)}
      role="option"
      tabIndex={-1}
      aria-selected={isActive}
      onClick={onClick}
      onKeyDown={(event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        onClick();
      }}
      onPointerMove={onPointerMove}
      className={cn(
        'flex min-h-11 cursor-pointer items-center gap-3 px-3 py-2.5 transition-colors',
        isActive ? 'bg-muted/60' : 'bg-card hover:bg-muted/35',
      )}
    >
      <div className="min-w-0 flex-1">{renderItem(result, isActive)}</div>
      {behavior === 'combined' ? (
        <Badge className="shrink-0 rounded-md border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] uppercase tracking-[0.08em] text-muted-foreground">
          {result.modeLabel}
        </Badge>
      ) : null}
    </div>
  );
}

export function CommandBarModal<T>({
  modes,
  behavior,
  defaultModeId,
  fallbackIcon = Search,
  title = 'Command bar',
  description = 'Search available results.',
  placeholder,
  emptyMessage = 'No matching results.',
  interstitial,
}: CommandBarModalProps<T>) {
  const {
    activeIndex,
    activeModeId,
    closeCommandBar,
    isOpen,
    query,
    returnFocusElement,
    selectionError,
    setActiveIndex,
    setActiveModeId,
    setQuery,
    setSelectionError,
  } = useCommandBar();
  const inputRef = useRef<HTMLInputElement>(null);
  const abortControllersRef = useRef(new Map<string, AbortController>());
  const activeOptionElementsRef = useRef(new Map<string, HTMLDivElement>());
  const requestTokensByModeIdRef = useRef(new Map<string, number>());
  const previousIsOpenRef = useRef(false);
  const previousQueryRef = useRef(query);
  const previousAsyncModeIdsRef = useRef<string[]>([]);
  const shouldScrollActiveOptionIntoViewRef = useRef(false);
  const listboxId = useId();
  const modeById = useMemo(() => new Map(modes.map((mode) => [mode.id, mode] as const)), [modes]);
  const [asyncItemsByModeId, setAsyncItemsByModeId] = useState<Record<string, readonly T[]>>({});
  const [asyncLoadingByModeId, setAsyncLoadingByModeId] = useState<Record<string, boolean>>({});
  const [asyncErrorByModeId, setAsyncErrorByModeId] = useState<Record<string, string | null>>({});
  const shortcutPrefix = 'Ctrl+';
  const resolvedModes = useMemo<CommandBarResolvedMode<T>[]>(() => {
    return modes.map((mode) => ({
      ...mode,
      source: resolveCommandBarModeSource(mode, asyncItemsByModeId),
    }));
  }, [asyncItemsByModeId, modes]);
  const resolvedDistinctModeId = useMemo(() => {
    if (behavior !== 'distinct') return null;
    if (activeModeId && modeById.has(activeModeId)) return activeModeId;
    return resolveCommandBarOpenModeId(resolvedModes, defaultModeId);
  }, [activeModeId, behavior, defaultModeId, modeById, resolvedModes]);
  const activeMode = resolvedDistinctModeId ? modeById.get(resolvedDistinctModeId) ?? null : null;
  const LeadingIcon = activeMode?.leadingIcon ?? fallbackIcon;
  const results = useMemo(
    () => filterCommandBarResults({
      modes: resolvedModes,
      behavior,
      query,
      activeModeId: resolvedDistinctModeId,
      defaultModeId,
    }),
    [behavior, defaultModeId, query, resolvedDistinctModeId, resolvedModes],
  );
  const activeOptionId = activeIndex >= 0 ? getCommandBarOptionId(listboxId, results[activeIndex]?.compositeKey ?? '') : undefined;
  const ResultContainer = behavior === 'distinct' ? activeMode?.ResultContainer ?? DefaultResultContainer : DefaultResultContainer;
  const showModeChips = behavior === 'distinct' && modes.length > 1;
  const showModeShortcutHint = behavior === 'distinct' && modes.length > 1;
  const modeShortcutHint = `${shortcutPrefix}1-${modes.length}`;
  const inputPlaceholder = placeholder ?? (activeMode ? `Search ${activeMode.label.toLowerCase()}` : 'Search');
  const listProps = useMemo<React.HTMLAttributes<HTMLDivElement>>(() => ({
    id: listboxId,
    role: 'listbox',
    tabIndex: 0,
    'aria-label': 'Command bar results',
    'aria-activedescendant': activeOptionId,
    className: 'overflow-hidden rounded-lg border border-border/70 bg-card shadow-soft',
  }), [activeOptionId, listboxId]);
  const activeAsyncModeIds = useMemo(() => {
    if (behavior === 'distinct') {
      if (!resolvedDistinctModeId) return [];
      const mode = modeById.get(resolvedDistinctModeId);
      return mode && isCommandBarAsyncSource(mode.source) ? [mode.id] : [];
    }

    return modes.filter((mode) => isCommandBarAsyncSource(mode.source)).map((mode) => mode.id);
  }, [behavior, modeById, modes, resolvedDistinctModeId]);
  const isLoading = activeAsyncModeIds.some((modeId) => asyncLoadingByModeId[modeId]);
  const asyncErrorMessage = activeAsyncModeIds.map((modeId) => asyncErrorByModeId[modeId]).find((message): message is string => Boolean(message));

  const abortInFlightRequests = useCallback((modeIds?: readonly string[]) => {
    const targetModeIds = modeIds ?? Array.from(abortControllersRef.current.keys());

    targetModeIds.forEach((modeId) => {
      const controller = abortControllersRef.current.get(modeId);
      if (!controller) return;
      controller.abort();
      abortControllersRef.current.delete(modeId);
    });
  }, []);

  const fetchAsyncResults = useCallback((modeIds: readonly string[], nextQuery: string) => {
    if (modeIds.length === 0) return;

    modeIds.forEach((modeId) => {
      const mode = modeById.get(modeId);
      if (!mode || !isCommandBarAsyncSource(mode.source)) return;

      abortInFlightRequests([modeId]);

      const controller = new AbortController();
      const requestToken = (requestTokensByModeIdRef.current.get(modeId) ?? 0) + 1;

      requestTokensByModeIdRef.current.set(modeId, requestToken);

      abortControllersRef.current.set(modeId, controller);
      setAsyncItemsByModeId((current) => ({ ...current, [modeId]: [] }));
      setAsyncLoadingByModeId((current) => ({ ...current, [modeId]: true }));
      setAsyncErrorByModeId((current) => ({ ...current, [modeId]: null }));

      mode.source(nextQuery, controller.signal)
        .then((items) => {
          if (controller.signal.aborted) return;
          if (requestToken !== requestTokensByModeIdRef.current.get(modeId)) return;

          abortControllersRef.current.delete(modeId);
          setAsyncItemsByModeId((current) => ({ ...current, [modeId]: items }));
          setAsyncErrorByModeId((current) => ({ ...current, [modeId]: null }));
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted) return;
          if (requestToken !== requestTokensByModeIdRef.current.get(modeId)) return;

          abortControllersRef.current.delete(modeId);
          setAsyncItemsByModeId((current) => ({ ...current, [modeId]: [] }));
          setAsyncErrorByModeId((current) => ({
            ...current,
            [modeId]: getCommandBarSelectionErrorMessage(error),
          }));
        })
        .finally(() => {
          if (controller.signal.aborted) return;
          if (requestToken !== requestTokensByModeIdRef.current.get(modeId)) return;

          setAsyncLoadingByModeId((current) => ({ ...current, [modeId]: false }));
        });
    });
  }, [abortInFlightRequests, modeById]);

  const handleModeSwitch = useCallback((modeId: string) => {
    setSelectionError(null);
    setActiveModeId(modeId);
  }, [setActiveModeId, setSelectionError]);

  const handleSelectResult = useCallback(async (result: CommandBarNormalizedResult<T>) => {
    const mode = modeById.get(result.modeId);
    if (!mode) return;

    setSelectionError(null);

    try {
      await Promise.resolve(mode.onSelect(result.item));
      closeCommandBar();
    } catch (error) {
      setSelectionError(getCommandBarSelectionErrorMessage(error));
    }
  }, [closeCommandBar, modeById, setSelectionError]);

  const handleKeyDown = useCallback((event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'ArrowDown') {
      if (results.length === 0) return;
      event.preventDefault();
      shouldScrollActiveOptionIntoViewRef.current = true;
      setActiveIndex((currentIndex) => {
        if (currentIndex < 0) return 0;
        return Math.min(currentIndex + 1, results.length - 1);
      });
      return;
    }

    if (event.key === 'ArrowUp') {
      if (results.length === 0) return;
      event.preventDefault();
      shouldScrollActiveOptionIntoViewRef.current = true;
      setActiveIndex((currentIndex) => {
        if (currentIndex < 0) return 0;
        return Math.max(currentIndex - 1, 0);
      });
      return;
    }

    if (event.key === 'Enter') {
      const activeResult = results[activeIndex];
      if (!activeResult) return;
      event.preventDefault();
      void handleSelectResult(activeResult);
      return;
    }

    if (behavior !== 'distinct') return;
    if (event.altKey || event.shiftKey) return;
    if (!/^\d$/.test(event.key) || event.key === '0') return;

    if (!event.ctrlKey) return;

    const nextMode = modes[Number(event.key) - 1];
    if (!nextMode) return;

    event.preventDefault();
    handleModeSwitch(nextMode.id);
  }, [activeIndex, behavior, handleModeSwitch, handleSelectResult, modes, results, setActiveIndex]);

  useEffect(() => {
    if (!isOpen || behavior !== 'distinct' || !resolvedDistinctModeId || activeModeId === resolvedDistinctModeId) {
      return;
    }

    setActiveModeId(resolvedDistinctModeId);
  }, [activeModeId, behavior, isOpen, resolvedDistinctModeId, setActiveModeId]);

  useEffect(() => {
    if (!isOpen) return;
    setActiveIndex(resolveCommandBarActiveIndex(results));
  }, [isOpen, results, setActiveIndex]);

  useEffect(() => {
    if (!isOpen) {
      shouldScrollActiveOptionIntoViewRef.current = false;
      return;
    }

    if (!shouldScrollActiveOptionIntoViewRef.current) return;

    const activeResult = results[activeIndex];

    if (!activeResult) {
      shouldScrollActiveOptionIntoViewRef.current = false;
      return;
    }

    activeOptionElementsRef.current.get(activeResult.compositeKey)?.scrollIntoView?.({ block: 'nearest' });
    shouldScrollActiveOptionIntoViewRef.current = false;
  }, [activeIndex, isOpen, results]);

  useEffect(() => {
    const previouslyOpen = previousIsOpenRef.current;
    const previousQuery = previousQueryRef.current;
    const previousAsyncModeIds = previousAsyncModeIdsRef.current;
    const modesChanged =
      previousAsyncModeIds.length !== activeAsyncModeIds.length
      || previousAsyncModeIds.some((modeId, index) => modeId !== activeAsyncModeIds[index]);

    previousIsOpenRef.current = isOpen;
    previousQueryRef.current = query;
    previousAsyncModeIdsRef.current = [...activeAsyncModeIds];

    if (!isOpen) {
      abortInFlightRequests();
      requestTokensByModeIdRef.current.clear();
      setAsyncLoadingByModeId({});
      setAsyncErrorByModeId({});
      setAsyncItemsByModeId({});
      return;
    }

    if (activeAsyncModeIds.length === 0) return;

    if (!previouslyOpen || modesChanged) {
      fetchAsyncResults(activeAsyncModeIds, query);
      return () => {
        abortInFlightRequests(activeAsyncModeIds);
      };
    }

    if (query === previousQuery) {
      return () => {
        abortInFlightRequests(activeAsyncModeIds);
      };
    }

    const timeoutId = window.setTimeout(() => {
      fetchAsyncResults(activeAsyncModeIds, query);
    }, 150);

    return () => {
      window.clearTimeout(timeoutId);
      abortInFlightRequests(activeAsyncModeIds);
    };
  }, [abortInFlightRequests, activeAsyncModeIds, fetchAsyncResults, isOpen, query]);

  const showLoadingState = isLoading && results.length === 0;
  const showErrorState = !showLoadingState && Boolean(asyncErrorMessage) && results.length === 0;
  return (
    <Dialog open={isOpen} onOpenChange={(nextOpen) => !nextOpen && closeCommandBar()}>
      <DialogContent
        className="gap-0 overflow-hidden p-0"
        onCloseAutoFocus={(event) => event.preventDefault()}
        onEscapeKeyDown={(event) => {
          event.preventDefault();
          const focusTarget = returnFocusElement;
          closeCommandBar();
          if (focusTarget?.isConnected) {
            queueMicrotask(() => {
              if (focusTarget.isConnected) focusTarget.focus();
            });
          }
        }}
        onKeyDown={handleKeyDown}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          inputRef.current?.focus();
        }}
      >
        <DialogTitle className="sr-only">{title}</DialogTitle>
        <DialogDescription className="sr-only">{description}</DialogDescription>

        <div data-testid="command-bar-header" className="border-b border-border/70 bg-muted/20 p-3 sm:p-4">
          <div className="flex items-center gap-3">
            <div className="relative min-w-0 flex-1">
              <span
                data-testid="command-bar-leading-icon"
                data-mode-id={activeMode?.id ?? 'default'}
                className="pointer-events-none absolute left-3 top-1/2 z-10 -translate-y-1/2 text-muted-foreground"
              >
                <LeadingIcon className="h-4 w-4" aria-hidden="true" />
              </span>
              <Input
                ref={inputRef}
                value={query}
                onChange={(event) => {
                  setSelectionError(null);
                  setQuery(event.target.value);
                }}
                aria-label="Command bar search"
                aria-controls={listboxId}
                className="h-11 rounded-lg border-border/70 bg-background pl-10 pr-10"
                placeholder={inputPlaceholder}
              />
              {query.length > 0 ? (
                <button
                  type="button"
                  aria-label="Clear search"
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => {
                    setSelectionError(null);
                    setQuery('');
                    inputRef.current?.focus();
                  }}
                  className="absolute right-3 top-1/2 z-10 inline-flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <X className="h-4 w-4" aria-hidden="true" />
                </button>
              ) : null}
            </div>
            {showModeShortcutHint ? (
              <div className="shrink-0 text-right">
                <div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Modes</div>
                <div className="text-xs font-medium text-foreground/80">{modeShortcutHint}</div>
              </div>
            ) : null}
          </div>

          {showModeChips ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {modes.map((mode, index) => {
                const isSelected = mode.id === resolvedDistinctModeId;

                return (
                  <button
                    key={mode.id}
                    type="button"
                    aria-pressed={isSelected}
                    onClick={() => handleModeSwitch(mode.id)}
                    className={cn(
                      'inline-flex h-8 items-center gap-2 rounded-md border px-2.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                      isSelected
                        ? 'border-border bg-card text-foreground shadow-soft'
                        : 'border-border/60 bg-muted/35 text-muted-foreground hover:bg-muted/55',
                    )}
                  >
                    <span>{mode.label}</span>
                      <span className="text-[10px] uppercase tracking-[0.08em] text-muted-foreground">
                        Ctrl+{index + 1}
                      </span>
                  </button>
                );
              })}
            </div>
          ) : null}
        </div>

        {interstitial ? (
          <div data-testid="command-bar-interstitial" className="border-b border-border/70 bg-card px-3 py-2.5 sm:px-4">
            {interstitial}
          </div>
        ) : null}

        <div data-testid="command-bar-results" className="bg-card p-2 sm:p-3">
          {selectionError ? (
            <Alert variant="destructive" className="mb-2 border-destructive/40 px-3 py-2.5">
              <AlertTitle>Action failed</AlertTitle>
              <AlertDescription>{selectionError}</AlertDescription>
            </Alert>
          ) : null}
          <div {...listProps}>
            {results.length > 0 ? (
              <ResultContainer
                listProps={listProps}
                query={query}
                modeId={behavior === 'distinct' ? resolvedDistinctModeId : activeModeId}
                activeIndex={activeIndex}
                results={results}
              >
                {results.map((result, index) => {
                  const mode = modeById.get(result.modeId);
                  const ResultItem = mode?.ResultItem;

                  return (
                    <CommandBarOptionRow
                      key={result.compositeKey}
                      behavior={behavior}
                      isActive={index === activeIndex}
                      listboxId={listboxId}
                      optionRef={(node) => {
                        if (node) {
                          activeOptionElementsRef.current.set(result.compositeKey, node);
                          return;
                        }

                        activeOptionElementsRef.current.delete(result.compositeKey);
                      }}
                      result={result}
                      onClick={() => {
                        void handleSelectResult(result);
                      }}
                      onPointerMove={() => {
                        if (index === activeIndex) return;
                        setActiveIndex(index);
                      }}
                      renderItem={(currentResult, isActive) => {
                        if (ResultItem) {
                          return (
                            <ResultItem
                              item={currentResult.item}
                              selected={isActive}
                              query={query}
                              modeLabel={currentResult.modeLabel}
                            />
                          );
                        }

                        return (
                          <div className="truncate text-sm font-medium text-foreground">
                            {currentResult.searchText}
                          </div>
                        );
                      }}
                    />
                  );
                })}
              </ResultContainer>
            ) : showLoadingState ? (
              <div className="flex min-h-28 items-center gap-3 px-3 py-6 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                <div>
                  <div className="font-medium text-foreground">Loading results</div>
                  <div className="text-xs text-muted-foreground">Checking the latest matches.</div>
                </div>
              </div>
            ) : showErrorState ? (
              <div className="px-3 py-6 text-sm text-muted-foreground">
                <div className="font-medium text-foreground">Could not load results</div>
                <div className="text-xs text-muted-foreground">{asyncErrorMessage}</div>
              </div>
            ) : (
              <div className="px-3 py-6 text-sm text-muted-foreground">{emptyMessage}</div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
