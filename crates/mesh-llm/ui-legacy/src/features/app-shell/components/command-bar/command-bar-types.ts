import type * as React from 'react';

export type CommandBarBehavior = 'distinct' | 'combined';

export type CommandBarAsyncSource<T> = (query: string, signal: AbortSignal) => Promise<readonly T[]>;

export type CommandBarSource<T> = readonly T[] | CommandBarAsyncSource<T>;

export type CommandBarSelectHandler<T> = (item: T) => void | Promise<void>;

export interface CommandBarNormalizedResult<T> {
  item: T;
  modeId: string;
  modeLabel: string;
  itemKey: string;
  compositeKey: string;
  searchText: string;
}

export interface CommandBarResultContainerProps<T> {
  children: React.ReactNode;
  listProps: React.HTMLAttributes<HTMLDivElement>;
  query: string;
  modeId: string | null;
  activeIndex: number;
  results: readonly CommandBarNormalizedResult<T>[];
}

export interface CommandBarResultItemProps<T> {
  item: T;
  selected: boolean;
  query: string;
  modeLabel: string;
}

export interface CommandBarMode<T> {
  id: string;
  label: string;
  leadingIcon: React.ElementType;
  source: CommandBarSource<T>;
  getItemKey: (item: T) => string;
  getSearchText: (item: T) => string;
  getKeywords?: (item: T) => readonly string[];
  ResultContainer?: React.ComponentType<CommandBarResultContainerProps<T>>;
  ResultItem?: React.ComponentType<CommandBarResultItemProps<T>>;
  onSelect: CommandBarSelectHandler<T>;
}
