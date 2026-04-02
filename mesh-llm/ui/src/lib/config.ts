import { parse, stringify } from 'smol-toml';

import type { ConfigValidationError } from './api';
import type {
  MeshConfig,
  ModelAssignment,
  ModelSplit,
  NodeConfig,
  PlacementMode,
} from '../types/config';

export const AUTHORED_CONFIG_VERSION = 1;

type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function toStringOrUndefined(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function toNumberOrUndefined(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function toNonNegativeIntegerOrUndefined(value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
    return undefined;
  }
  return value;
}

function normalizeSplit(value: unknown): ModelSplit | null {
  if (!isRecord(value)) {
    return null;
  }

  const start = toNonNegativeIntegerOrUndefined(value.start);
  const end = toNonNegativeIntegerOrUndefined(value.end);
  const total = toNonNegativeIntegerOrUndefined(value.total);
  if (
    start === undefined ||
    end === undefined ||
    total === undefined ||
    total === 0 ||
    start >= end ||
    end > total
  ) {
    return null;
  }

  return { start, end, total };
}

function normalizeModel(value: unknown): { model: ModelAssignment | null; invalid: boolean } {
  if (!isRecord(value)) {
    return { model: null, invalid: false };
  }

  const name = toStringOrUndefined(value.name);
  if (!name) {
    return { model: null, invalid: false };
  }

  const model: ModelAssignment = { name };
  const modelKeyRaw = value.model_key;
  if (modelKeyRaw !== undefined) {
    const modelKey = toStringOrUndefined(modelKeyRaw);
    if (!modelKey || modelKey.trim().length === 0) {
      return { model: null, invalid: true };
    }
    model.model_key = modelKey;
  }

  const splitRaw = value.split;
  if (splitRaw !== undefined) {
    const split = normalizeSplit(splitRaw);
    if (!split) {
      return { model: null, invalid: true };
    }
    model.split = split;
  }

  const path = toStringOrUndefined(value.path);
  const ctxSize = toNumberOrUndefined(value.ctx_size);
  const moeExperts = toNumberOrUndefined(value.moe_experts);

  if (path !== undefined) {
    model.path = path;
  }
  model.ctx_size = ctxSize ?? 4096;
  if (moeExperts !== undefined) {
    model.moe_experts = moeExperts;
  }

  const gpuIndexRaw = value.gpu_index;
  if (gpuIndexRaw !== undefined) {
    const gpuIndex = toNonNegativeIntegerOrUndefined(gpuIndexRaw);
    if (gpuIndex === undefined) {
      return { model: null, invalid: true };
    }
    model.gpu_index = gpuIndex;
  }

  return { model, invalid: false };
}

function normalizePlacementMode(value: unknown): PlacementMode | null {
  if (value === undefined) {
    return 'pooled';
  }
  return value === 'pooled' || value === 'separate' ? value : null;
}

export function deepEqual(a: unknown, b: unknown): boolean {
  // Handle primitives and null
  if (a === b) return true;
  if (a === null || b === null) return a === b;
  if (typeof a !== 'object' || typeof b !== 'object') return false;

  // Handle arrays
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!deepEqual(a[i], b[i])) return false;
    }
    return true;
  }

  // Handle objects
  if (Array.isArray(a) || Array.isArray(b)) return false;

  const aObj = a as UnknownRecord;
  const bObj = b as UnknownRecord;
  const aKeys = Object.keys(aObj).filter((k) => aObj[k] !== undefined);
  const bKeys = Object.keys(bObj).filter((k) => bObj[k] !== undefined);

  aKeys.sort();
  bKeys.sort();

  if (aKeys.length !== bKeys.length) return false;

  for (let i = 0; i < aKeys.length; i++) {
    if (aKeys[i] !== bKeys[i]) return false;
  }

  for (const key of aKeys) {
    if (!deepEqual(aObj[key], bObj[key])) {
      return false;
    }
  }

  return true;
}

function normalizeNode(value: unknown): { node: NodeConfig | null; invalid: boolean } {
  if (!isRecord(value)) {
    return { node: null, invalid: false };
  }

  const nodeId = toStringOrUndefined(value.node_id);
  if (!nodeId) {
    return { node: null, invalid: false };
  }

  const rawModels = value.models;
  const models: ModelAssignment[] = [];
  let invalid = false;

  if (Array.isArray(rawModels)) {
    for (const rawModel of rawModels) {
      const normalized = normalizeModel(rawModel);
      if (normalized.invalid) {
        invalid = true;
        continue;
      }
      if (normalized.model) {
        models.push(normalized.model);
      }
    }
  }

  const node: NodeConfig = {
    node_id: nodeId,
    placement_mode: 'pooled',
    models,
  };

  const hostname = toStringOrUndefined(value.hostname);
  if (hostname !== undefined) {
    node.hostname = hostname;
  }

  const placementMode = normalizePlacementMode(value.placement_mode);
  if (!placementMode) {
    return { node: null, invalid: true };
  }
  node.placement_mode = placementMode;

  return { node, invalid };
}

function normalizeConfig(value: unknown): MeshConfig | null {
  if (!isRecord(value)) {
    return null;
  }

  const version = toNonNegativeIntegerOrUndefined(value.version);
  if (version !== AUTHORED_CONFIG_VERSION) {
    return null;
  }

  const rawNodes = value.nodes;
  const nodes: NodeConfig[] = [];

  if (Array.isArray(rawNodes)) {
    for (const rawNode of rawNodes) {
      const normalized = normalizeNode(rawNode);
      if (normalized.invalid) {
        return null;
      }
      if (normalized.node) {
        nodes.push(normalized.node);
      }
    }
  }

  return { version: AUTHORED_CONFIG_VERSION, nodes };
}

export function parseConfig(inputToml: string): MeshConfig | null {
  try {
    const parsed = parse(inputToml);
    return normalizeConfig(parsed);
  } catch {
    return null;
  }
}

export function serializeConfig(config: MeshConfig): string {
  const normalized = normalizeConfig(config) ?? createEmptyConfig();
  return stringify(normalized);
}

export function clampConfigCtxSizes(
  config: MeshConfig,
  maxCtxByModel: Map<string, number>,
): { config: MeshConfig; clamped: boolean } {
  let needsClamp = false;
  for (const node of config.nodes) {
    for (const model of node.models) {
      const maxCtx = maxCtxByModel.get(model.name);
      if (maxCtx != null && model.ctx_size != null && model.ctx_size > maxCtx) {
        needsClamp = true;
        break;
      }
    }
    if (needsClamp) break;
  }

  if (!needsClamp) {
    return { config, clamped: false };
  }

  return {
    config: {
      ...config,
      nodes: config.nodes.map((node) => ({
        ...node,
        models: node.models.map((model) => {
          const maxCtx = maxCtxByModel.get(model.name);
          if (maxCtx != null && model.ctx_size != null && model.ctx_size > maxCtx) {
            return { ...model, ctx_size: maxCtx };
          }
          return model;
        }),
      })),
    },
    clamped: true,
  };
}

export function createEmptyConfig(): MeshConfig {
  return { version: AUTHORED_CONFIG_VERSION, nodes: [] };
}

export function validateSplits(nodes: NodeConfig[]): ConfigValidationError[] {
  const errors: ConfigValidationError[] = [];

  type SplitEntry = {
    nodeIdx: number;
    modelIdx: number;
    split: ModelSplit;
  };

  const groups = new Map<string, SplitEntry[]>();

  for (let nodeIdx = 0; nodeIdx < nodes.length; nodeIdx++) {
    const node = nodes[nodeIdx];
    for (let modelIdx = 0; modelIdx < node.models.length; modelIdx++) {
      const model = node.models[modelIdx];
      if (!model.split) continue;

      const groupKey = `${model.name}::${model.model_key ?? ''}::${model.split.total}`;
      const group = groups.get(groupKey);
      if (group) {
        group.push({ nodeIdx, modelIdx, split: model.split });
      } else {
        groups.set(groupKey, [{ nodeIdx, modelIdx, split: model.split }]);
      }
    }
  }

  for (const entries of groups.values()) {
    const sorted = [...entries].sort((a, b) => a.split.start - b.split.start);
    const first = sorted[0];
    const last = sorted[sorted.length - 1];
    const total = first.split.total;

    for (let i = 0; i < sorted.length - 1; i++) {
      const curr = sorted[i];
      const next = sorted[i + 1];

      if (next.split.start < curr.split.end) {
        errors.push({
          code: 'overlapping_split_ranges',
          path: `nodes[${curr.nodeIdx}].models[${curr.modelIdx}].split`,
          message: `Split ranges overlap: [${curr.split.start}, ${curr.split.end}) and [${next.split.start}, ${next.split.end})`,
        });
      } else if (curr.split.end !== next.split.start) {
        errors.push({
          code: 'split_gap',
          path: `nodes[${curr.nodeIdx}].models[${curr.modelIdx}].split`,
          message: `Gap in split coverage between ${curr.split.end} and ${next.split.start}`,
        });
      }
    }

    if (first.split.start !== 0 || last.split.end !== total) {
      errors.push({
        code: 'incomplete_split_coverage',
        path: `nodes[${first.nodeIdx}].split`,
        message: `Split coverage is incomplete: expected [0, ${total}], got [${first.split.start}, ${last.split.end}]`,
      });
    }
  }

  return errors;
}
