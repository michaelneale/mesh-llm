import { describe, expect, it } from 'vitest';

import {
  createCommandBarCompositeKey,
  filterCommandBarResults,
  normalizeCommandBarResult,
  resolveCommandBarActiveIndex,
  resolveCommandBarOpenModeId,
  type CommandBarResolvedMode,
} from './command-bar-helpers';

type Item = { id: string; name: string; keywords?: readonly string[] };

const createMode = (id: string, label: string, source: readonly Item[]): CommandBarResolvedMode<Item> => ({
  id,
  label,
  leadingIcon: 'span',
  source,
  getItemKey: (item) => item.id,
  getSearchText: (item) => item.name,
  getKeywords: (item) => item.keywords ?? [],
  onSelect: () => {},
});

describe('command-bar helpers', () => {
  it('normalizes result metadata and search text', () => {
    const mode = createMode('models', 'Models', []);
    const result = normalizeCommandBarResult(mode, { id: 'foo', name: 'Foo' });

    expect(result).toEqual({
      item: { id: 'foo', name: 'Foo' },
      modeId: 'models',
      modeLabel: 'Models',
      itemKey: 'foo',
      compositeKey: 'models:foo',
      searchText: 'Foo',
    });
  });

  it('keeps composite keys distinct across modes', () => {
    expect(createCommandBarCompositeKey('models', 'foo')).toBe('models:foo');
    expect(createCommandBarCompositeKey('nodes', 'foo')).toBe('nodes:foo');
    expect(createCommandBarCompositeKey('models', 'foo')).not.toBe(createCommandBarCompositeKey('nodes', 'foo'));
  });

  it('filters distinct results to the active mode without leaking other modes', () => {
    const modes = [
      createMode('models', 'Models', [
        { id: 'shared', name: 'Shared model' },
        { id: 'local', name: 'Local model' },
      ]),
      createMode('nodes', 'Nodes', [{ id: 'shared', name: 'Shared node' }]),
    ];

    const results = filterCommandBarResults({
      modes,
      behavior: 'distinct',
      query: '',
      activeModeId: 'nodes',
      defaultModeId: 'models',
    });

    expect(results).toHaveLength(1);
    expect(results).toEqual([
      expect.objectContaining({
        modeId: 'nodes',
        modeLabel: 'Nodes',
        compositeKey: 'nodes:shared',
      }),
    ]);
  });

  it('flattens combined results while preserving mode metadata and composite keys', () => {
    const modes = [
      createMode('models', 'Models', [
        { id: 'shared', name: 'Shared model' },
        { id: 'unique-model', name: 'Unique model' },
      ]),
      createMode('nodes', 'Nodes', [{ id: 'shared', name: 'Shared node' }]),
    ];

    const results = filterCommandBarResults({
      modes,
      behavior: 'combined',
      query: '',
    });

    expect(results).toEqual([
      expect.objectContaining({ modeId: 'models', modeLabel: 'Models', compositeKey: 'models:shared' }),
      expect.objectContaining({ modeId: 'models', modeLabel: 'Models', compositeKey: 'models:unique-model' }),
      expect.objectContaining({ modeId: 'nodes', modeLabel: 'Nodes', compositeKey: 'nodes:shared' }),
    ]);
  });

  it('matches search text and keywords case-insensitively', () => {
    const modes = [createMode('models', 'Models', [{ id: 'keyword', name: 'Hidden name', keywords: ['Alpha Beta'] }])];

    const results = filterCommandBarResults({
      modes,
      behavior: 'combined',
      query: 'beTa',
    });

    expect(results).toHaveLength(1);
    expect(results[0]).toEqual(expect.objectContaining({ compositeKey: 'models:keyword' }));
  });

  it('orders matches by prefix, then earlier substring index, then mode order, then item order', () => {
    const modes = [
      createMode('first', 'First', [
        { id: 'first-late', name: 'x alpha' },
        { id: 'first-early', name: 'xalpha' },
      ]),
      createMode('second', 'Second', [
        { id: 'second-prefix', name: 'alpha root' },
        { id: 'second-late', name: 'x alpha' },
      ]),
    ];

    const results = filterCommandBarResults({
      modes,
      behavior: 'combined',
      query: 'alpha',
    });

    expect(results.map((result) => result.compositeKey)).toEqual([
      'second:second-prefix',
      'first:first-early',
      'first:first-late',
      'second:second-late',
    ]);
  });

  it('resolves the open mode from the default id or falls back to the first mode', () => {
    const modes = [createMode('models', 'Models', []), createMode('nodes', 'Nodes', [])];

    expect(resolveCommandBarOpenModeId(modes, 'nodes')).toBe('nodes');
    expect(resolveCommandBarOpenModeId(modes, 'missing')).toBe('models');
    expect(resolveCommandBarOpenModeId(modes)).toBe('models');
    expect(resolveCommandBarOpenModeId([], 'nodes')).toBeNull();
  });

  it('resets the active index to zero for non-empty results and minus one when empty', () => {
    expect(resolveCommandBarActiveIndex([{ id: 'one' }])).toBe(0);
    expect(resolveCommandBarActiveIndex([])).toBe(-1);
  });
});
