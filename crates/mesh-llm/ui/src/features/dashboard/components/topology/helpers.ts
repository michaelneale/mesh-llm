import type { TopologyNode } from "../../../app-shell/lib/topology-types";

import type { ColorTuple } from "./types";

export const TAU = Math.PI * 2;
export const ENTRY_ANIMATION_DURATION_MS = 1100;
export const LINE_REVEAL_DURATION_MS = 260;

export function hashString(value: string) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

export function color(hex: string, alpha = 1): ColorTuple {
  const normalized = hex.replace("#", "");
  const expanded =
    normalized.length === 3
      ? normalized
          .split("")
          .map((part) => `${part}${part}`)
          .join("")
      : normalized;
  const r = parseInt(expanded.slice(0, 2), 16) / 255;
  const g = parseInt(expanded.slice(2, 4), 16) / 255;
  const b = parseInt(expanded.slice(4, 6), 16) / 255;
  return [r, g, b, alpha];
}

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function mixColor(
  left: ColorTuple,
  right: ColorTuple,
  amount: number,
): ColorTuple {
  const mixAmount = clamp(amount, 0, 1);
  return [
    left[0] + (right[0] - left[0]) * mixAmount,
    left[1] + (right[1] - left[1]) * mixAmount,
    left[2] + (right[2] - left[2]) * mixAmount,
    left[3] + (right[3] - left[3]) * mixAmount,
  ];
}

export function withAlpha(value: ColorTuple, alpha: number): ColorTuple {
  return [value[0], value[1], value[2], alpha];
}

export function nodeUpdateSignature(node: TopologyNode) {
  return JSON.stringify({
    self: node.self,
    vram: node.vram,
    host: node.host,
    client: node.client,
    serving: node.serving,
    servingModels: node.servingModels,
    state: node.state,
    latencyMs: node.latencyMs,
    hostname: node.hostname,
    isSoc: node.isSoc,
    gpus:
      node.gpus?.map((gpu) => ({
        name: gpu.name,
        vram_bytes: gpu.vram_bytes,
        bandwidth_gbps: gpu.bandwidth_gbps,
      })) ?? [],
  });
}
