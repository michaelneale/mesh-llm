export type TopologyNode = {
  id: string;
  vram: number;
  self: boolean;
  host: boolean;
  client: boolean;
  serving: string;
  servingModels: string[];
  statusLabel: string;
  latencyMs?: number | null;
  hostname?: string;
  isSoc?: boolean;
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
};

export type TopologyLayoutBucket = "center" | "serving" | "worker" | "client";

export type PositionedTopologyNode = TopologyNode & {
  x: number;
  y: number;
  bucket: TopologyLayoutBucket;
};

export type BucketedTopologyNode = TopologyNode & {
  bucket: TopologyLayoutBucket;
  width: number;
  height: number;
};

export type TopologyNodeInfo = {
  role: string;
  statusLabel: string;
  latencyMs?: number | null;
  loadedModel: string;
  loadedModels: string[];
  vramGb: number;
  vramSharePct: number;
  hostname?: string;
  isSoc?: boolean;
  gpus?: { name: string; vram_bytes: number }[];
};

export type TopologyLayoutMode = "elk" | "classic";

export type TopologyLayoutContext = {
  center: TopologyNode;
  serving: TopologyNode[];
  workers: TopologyNode[];
  clients: TopologyNode[];
  nodeRadius: number;
  nodes: BucketedTopologyNode[];
};

export type TopologyLayoutDefinition = {
  label: string;
  edgeType: "straight" | "smoothstep";
  direction: "horizontal" | "vertical";
  getImmediateLayout: (
    context: TopologyLayoutContext,
  ) => PositionedTopologyNode[];
  getLayout: (
    context: TopologyLayoutContext,
  ) => Promise<PositionedTopologyNode[]>;
};
