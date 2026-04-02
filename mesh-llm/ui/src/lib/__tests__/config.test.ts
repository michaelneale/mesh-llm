import { describe, expect, it } from 'vitest';

import { deepEqual, parseConfig, serializeConfig, validateSplits } from '../config';
import type { MeshConfig, NodeConfig } from '../../types/config';

describe('mesh config parse/serialize', () => {
  it('round-trips full mesh config shape', () => {
    const input: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          hostname: 'alpha.local',
          placement_mode: 'separate',
          models: [
            {
              name: 'Qwen3-30B-A3B-Q4_K_M',
              model_key: 'mk-qwen3-30b',
              split: { start: 0, end: 21, total: 33 },
              path: '/Users/test/.models/Qwen3-30B-A3B-Q4_K_M.gguf',
              ctx_size: 8192,
              moe_experts: 24,
              gpu_index: 0,
            },
            {
              name: 'Qwen2.5-Coder-7B-Q4_K_M',
              model_key: 'mk-qwen2.5-coder-7b',
              split: { start: 21, end: 33, total: 33 },
              ctx_size: 4096,
              gpu_index: 1,
            },
          ],
        },
        {
          node_id: 'node-b',
          placement_mode: 'pooled',
          models: [
            {
              name: 'GLM-4.7-Flash-Q4_K_M',
              ctx_size: 4096,
            },
          ],
        },
      ],
    };

    const serialized = serializeConfig(input);
    const parsed = parseConfig(serialized);

    expect(parsed).toEqual(input);
  });

  it('returns null for invalid toml', () => {
    expect(parseConfig('invalid = [broken toml')).toBeNull();
  });

  it('requires explicit authored schema version', () => {
    expect(parseConfig('')).toBeNull();
    expect(parseConfig('[[nodes]]\nnode_id = "node-a"\n')).toBeNull();
    expect(parseConfig('version = 1\n')).toEqual({ version: 1, nodes: [] });
    expect(parseConfig('version = 2\n')).toBeNull();
    expect(parseConfig('version = 3\n')).toBeNull();
  });

  it('defaults missing nodes/models to empty arrays for valid versioned config', () => {
    expect(parseConfig('version = 1\n')).toEqual({ version: 1, nodes: [] });
    expect(parseConfig('version = 1\n\n[[nodes]]\nnode_id = "node-a"\n')).toEqual({
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          placement_mode: 'pooled',
          models: [],
        },
      ],
    });
  });

  it('rejects malformed split objects instead of silently accepting them', () => {
    const malformed = `
version = 1

[[nodes]]
node_id = "node-a"

[[nodes.models]]
name = "Qwen3-30B-A3B-Q4_K_M"
model_key = "mk-qwen3-30b"
split = { start = 0, end = "20", total = 33 }
`;

    expect(parseConfig(malformed)).toBeNull();
  });

  it('rejects zero-layer splits (start >= end)', () => {
    const zeroLayers = `
version = 1

[[nodes]]
node_id = "node-a"

[[nodes.models]]
name = "Qwen3-30B-A3B-Q4_K_M"
model_key = "mk-qwen3-30b"
split = { start = 0, end = 0, total = 33 }
`;

    expect(parseConfig(zeroLayers)).toBeNull();

    const reversed = `
version = 1

[[nodes]]
node_id = "node-a"

[[nodes.models]]
name = "Qwen3-30B-A3B-Q4_K_M"
model_key = "mk-qwen3-30b"
split = { start = 10, end = 5, total = 33 }
`;

    expect(parseConfig(reversed)).toBeNull();
  });

  it('accepts valid single-node full-model split', () => {
    const fullModel = `
version = 1

[[nodes]]
node_id = "node-a"

[[nodes.models]]
name = "Qwen3-30B-A3B-Q4_K_M"
model_key = "mk-qwen3-30b"
split = { start = 0, end = 33, total = 33 }
`;

    const parsed = parseConfig(fullModel);
    expect(parsed).not.toBeNull();
    expect(parsed?.version).toBe(1);
    expect(parsed?.nodes[0].models[0].split).toEqual({ start: 0, end: 33, total: 33 });
    expect(parsed?.nodes[0].placement_mode).toBe('pooled');
  });

  it('serializes parsed schema v1 config back as schema v1', () => {
    const parsed = parseConfig('version = 1\n');
    expect(parsed).toEqual({ version: 1, nodes: [] });
    const serialized = serializeConfig(parsed as MeshConfig);
    expect(serialized).toContain('version = 1');
  });

  it('rejects invalid placement_mode values', () => {
    const invalidPlacementMode = `
version = 1

[[nodes]]
node_id = "node-a"
placement_mode = "invalid"
`;

    expect(parseConfig(invalidPlacementMode)).toBeNull();
  });

  it('rejects invalid gpu_index values', () => {
    const negativeGpuIndex = `
version = 1

[[nodes]]
node_id = "node-a"
placement_mode = "separate"

[[nodes.models]]
name = "Qwen3-30B-A3B-Q4_K_M"
gpu_index = -1
`;

    const nonIntegerGpuIndex = `
version = 1

[[nodes]]
node_id = "node-a"
placement_mode = "separate"

[[nodes.models]]
name = "Qwen3-30B-A3B-Q4_K_M"
gpu_index = 1.5
`;

    const nonNumericGpuIndex = `
version = 1

[[nodes]]
node_id = "node-a"
placement_mode = "separate"

[[nodes.models]]
name = "Qwen3-30B-A3B-Q4_K_M"
gpu_index = "0"
`;

    expect(parseConfig(negativeGpuIndex)).toBeNull();
    expect(parseConfig(nonIntegerGpuIndex)).toBeNull();
    expect(parseConfig(nonNumericGpuIndex)).toBeNull();
  });
});

describe('deepEqual', () => {
  it('returns true for identical primitives', () => {
    expect(deepEqual(1, 1)).toBe(true);
    expect(deepEqual('test', 'test')).toBe(true);
    expect(deepEqual(true, true)).toBe(true);
  });

  it('returns false for different primitives', () => {
    expect(deepEqual(1, 2)).toBe(false);
    expect(deepEqual('test', 'other')).toBe(false);
    expect(deepEqual(true, false)).toBe(false);
  });

  it('returns true for identical objects regardless of key order', () => {
    expect(deepEqual({ a: 1, b: 2 }, { b: 2, a: 1 })).toBe(true);
  });

  it('returns false for objects with different values', () => {
    expect(deepEqual({ split: { start: 0, end: 5 } }, { split: { start: 0, end: 6 } })).toBe(false);
  });

  it('treats undefined values and missing keys as equivalent', () => {
    expect(deepEqual({ a: 1, b: undefined }, { a: 1 })).toBe(true);
    expect(deepEqual({ a: 1 }, { a: 1, b: undefined })).toBe(true);
  });

  it('returns true for identical arrays', () => {
    expect(deepEqual([1, 2, 3], [1, 2, 3])).toBe(true);
  });

  it('returns false for arrays with different lengths', () => {
    expect(deepEqual([1, 2], [1, 2, 3])).toBe(false);
  });

  it('returns false for arrays with different values', () => {
    expect(deepEqual([1, 2, 3], [1, 2, 4])).toBe(false);
  });

  it('returns true for null === null', () => {
    expect(deepEqual(null, null)).toBe(true);
  });

  it('returns false for null vs undefined', () => {
    expect(deepEqual(null, undefined)).toBe(false);
  });

  it('returns true for nested objects with same structure', () => {
    const obj1: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          placement_mode: 'pooled',
          models: [
            {
              name: 'Qwen3-30B',
              ctx_size: 4096,
            },
          ],
        },
      ],
    };

    const obj2: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          placement_mode: 'pooled',
          models: [
            {
              name: 'Qwen3-30B',
              ctx_size: 4096,
            },
          ],
        },
      ],
    };

    expect(deepEqual(obj1, obj2)).toBe(true);
  });

  it('returns false for nested objects with different values', () => {
    const obj1: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          placement_mode: 'pooled',
          models: [
            {
              name: 'Qwen3-30B',
              ctx_size: 4096,
            },
          ],
        },
      ],
    };

    const obj2: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          placement_mode: 'pooled',
          models: [
            {
              name: 'Qwen3-30B',
              ctx_size: 8192,
            },
          ],
        },
      ],
    };

    expect(deepEqual(obj1, obj2)).toBe(false);
  });
});

describe('validateSplits', () => {
  function makeNode(splits: Array<{ start: number; end: number; total: number }>): NodeConfig {
    return {
      node_id: 'test-node',
      placement_mode: 'pooled',
      models: splits.map((split) => ({
        name: 'TestModel',
        model_key: 'key123',
        split,
      })),
    };
  }

  it('returns split_gap error when consecutive splits have a gap', () => {
    const nodes = [makeNode([
      { start: 0, end: 5, total: 15 },
      { start: 10, end: 15, total: 15 },
    ])];
    const errors = validateSplits(nodes);
    expect(errors).toHaveLength(1);
    expect(errors[0].code).toBe('split_gap');
  });

  it('returns overlapping_split_ranges error when splits overlap', () => {
    const nodes = [makeNode([
      { start: 0, end: 10, total: 15 },
      { start: 5, end: 15, total: 15 },
    ])];
    const errors = validateSplits(nodes);
    expect(errors.some((e) => e.code === 'overlapping_split_ranges')).toBe(true);
  });

  it('returns incomplete_split_coverage error when splits do not reach total', () => {
    const nodes = [makeNode([{ start: 0, end: 5, total: 10 }])];
    const errors = validateSplits(nodes);
    expect(errors).toHaveLength(1);
    expect(errors[0].code).toBe('incomplete_split_coverage');
  });

  it('returns no errors for valid contiguous splits that cover total', () => {
    const nodes = [makeNode([
      { start: 0, end: 5, total: 10 },
      { start: 5, end: 10, total: 10 },
    ])];
    const errors = validateSplits(nodes);
    expect(errors).toHaveLength(0);
  });
});
