import type { CommandBarBehavior, CommandBarMode, CommandBarNormalizedResult } from './command-bar-types';

export const createCommandBarCompositeKey = (modeId: string, itemKey: string): string => `${modeId}:${itemKey}`;

export type CommandBarResolvedMode<T> = Omit<CommandBarMode<T>, 'source'> & { source: readonly T[] };

type CommandBarSearchCandidate<T> = {
  result: CommandBarNormalizedResult<T>;
  modeIndex: number;
  itemIndex: number;
  prefixMatch: boolean;
  matchIndex: number;
};

export interface CommandBarResultFilterOptions<T> {
  modes: readonly CommandBarResolvedMode<T>[];
  behavior: CommandBarBehavior;
  query: string;
  activeModeId?: string | null;
  defaultModeId?: string | null;
}

const normalizeCommandBarSearchText = (value: string): string => value.trim().toLowerCase();

const getCommandBarSearchTerms = <T>(mode: CommandBarMode<T>, item: T): readonly string[] => {
  const terms = [mode.getSearchText(item), ...(mode.getKeywords?.(item) ?? [])];

  return terms.map(normalizeCommandBarSearchText).filter((term) => term.length > 0);
};

const findCommandBarSearchMatch = (terms: readonly string[], query: string): { prefixMatch: boolean; matchIndex: number } | null => {
  if (!query) return { prefixMatch: true, matchIndex: 0 };

  let matchIndex = Number.POSITIVE_INFINITY;

  for (const term of terms) {
    const index = term.indexOf(query);
    if (index < 0) continue;
    if (index === 0) return { prefixMatch: true, matchIndex: 0 };
    if (index < matchIndex) matchIndex = index;
  }

  if (matchIndex === Number.POSITIVE_INFINITY) return null;

  return { prefixMatch: false, matchIndex };
};

const compareCommandBarSearchCandidates = <T>(a: CommandBarSearchCandidate<T>, b: CommandBarSearchCandidate<T>): number => {
  if (a.prefixMatch !== b.prefixMatch) return a.prefixMatch ? -1 : 1;
  if (a.matchIndex !== b.matchIndex) return a.matchIndex - b.matchIndex;
  if (a.modeIndex !== b.modeIndex) return a.modeIndex - b.modeIndex;
  if (a.itemIndex !== b.itemIndex) return a.itemIndex - b.itemIndex;
  return a.result.compositeKey.localeCompare(b.result.compositeKey);
};

export const normalizeCommandBarResult = <T>(mode: CommandBarMode<T>, item: T): CommandBarNormalizedResult<T> => {
  const itemKey = mode.getItemKey(item);

  return {
    item,
    modeId: mode.id,
    modeLabel: mode.label,
    itemKey,
    compositeKey: createCommandBarCompositeKey(mode.id, itemKey),
    searchText: mode.getSearchText(item),
  };
};

export const normalizeCommandBarResults = <T>(mode: CommandBarMode<T>, items: readonly T[]): CommandBarNormalizedResult<T>[] =>
  items.map((item) => normalizeCommandBarResult(mode, item));

export const resolveCommandBarOpenModeId = <T>(modes: readonly CommandBarResolvedMode<T>[], defaultModeId?: string | null): string | null => {
  if (defaultModeId) {
    const defaultMode = modes.find((mode) => mode.id === defaultModeId);
    if (defaultMode) return defaultMode.id;
  }

  return modes[0]?.id ?? null;
};

export const resolveCommandBarActiveIndex = <T>(results: readonly T[]): number => (results.length > 0 ? 0 : -1);

export const filterCommandBarResults = <T>({
  modes,
  behavior,
  query,
  activeModeId,
  defaultModeId,
}: CommandBarResultFilterOptions<T>): CommandBarNormalizedResult<T>[] => {
  const normalizedQuery = normalizeCommandBarSearchText(query);
  const resolvedActiveModeId = behavior === 'distinct' ? activeModeId ?? resolveCommandBarOpenModeId(modes, defaultModeId) : null;

  const searchCandidates: CommandBarSearchCandidate<T>[] = [];

  modes.forEach((mode, modeIndex) => {
    if (behavior === 'distinct' && mode.id !== resolvedActiveModeId) return;

    mode.source.forEach((item, itemIndex) => {
      const result = normalizeCommandBarResult(mode, item);
      const match = findCommandBarSearchMatch(getCommandBarSearchTerms(mode, item), normalizedQuery);

      if (!match) return;

      searchCandidates.push({
        result,
        modeIndex,
        itemIndex,
        prefixMatch: match.prefixMatch,
        matchIndex: match.matchIndex,
      });
    });
  });

  return searchCandidates.sort(compareCommandBarSearchCandidates).map((candidate) => candidate.result);
};
