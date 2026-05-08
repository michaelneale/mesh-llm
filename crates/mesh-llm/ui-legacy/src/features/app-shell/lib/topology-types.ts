import type { LiveNodeState } from "./status-types";

export type TopologyNode = {
  id: string;
  vram: number;
  state: LiveNodeState;
  self: boolean;
  host: boolean;
  client: boolean;
  serving: string;
  servingModels: string[];
  statusLabel?: string;
  ageSeconds?: number | null;

  latencyMs?: number | null;
  hostname?: string;
  isSoc?: boolean;
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
};
