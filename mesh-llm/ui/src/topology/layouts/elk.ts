import ELK from "elkjs/lib/elk.bundled.js";

import type {
  BucketedTopologyNode,
  PositionedTopologyNode,
  TopologyLayoutDefinition,
  TopologyNode,
  TopologyNodeInfo,
} from "./types";
const TOPOLOGY_NODE_BASE_HEIGHT = 118;
const TOPOLOGY_NODE_MODEL_ROW_HEIGHT = 16;
const TOPOLOGY_NODE_CHIP_ROW_HEIGHT = 24;
const TOPOLOGY_NODE_CHIPS_PER_ROW = 3;
const TOPOLOGY_LAYER_GAP = 56;
const TOPOLOGY_ROW_GAP = 28;
const TOPOLOGY_PADDING = 16;

let _topologyElk: InstanceType<typeof ELK> | null = null;
function getTopologyElk(): InstanceType<typeof ELK> {
  if (!_topologyElk) {
    _topologyElk = new ELK();
  }
  return _topologyElk;
}

export function estimateTopologyNodeHeight(
  node: TopologyNode,
  info?: TopologyNodeInfo,
) {
  const modelRows = info && info.loadedModels.length > 1 ? info.loadedModels.length : 1;
  const chipCount =
    (info?.hostname ? 1 : 0) +
    1 +
    (node.self ? 0 : 1) +
    1 +
    1 +
    (info?.gpus?.length ?? 0);
  const chipRows = Math.max(1, Math.ceil(chipCount / TOPOLOGY_NODE_CHIPS_PER_ROW));

  return (
    TOPOLOGY_NODE_BASE_HEIGHT +
    modelRows * TOPOLOGY_NODE_MODEL_ROW_HEIGHT +
    chipRows * TOPOLOGY_NODE_CHIP_ROW_HEIGHT
  );
}

export function buildFallbackTopologyLayout(
  nodes: BucketedTopologyNode[],
): PositionedTopologyNode[] {
  const center = nodes.find((node) => node.bucket === "center");
  if (!center) return [];

  const { width: centerWidth, height: _centerHeight, ...centerNode } = center;
  const positioned: PositionedTopologyNode[] = [
    { ...centerNode, x: 0, y: 0 },
  ];

  const peerNodes = (["serving", "worker", "client"] as const).flatMap(
    (bucket) => nodes.filter((node) => node.bucket === bucket),
  );
  const maxPerCol = Math.max(1, Math.ceil(Math.sqrt(peerNodes.length)));
  let x = centerWidth + TOPOLOGY_LAYER_GAP;

  for (let colStart = 0; colStart < peerNodes.length; colStart += maxPerCol) {
    const col = peerNodes.slice(colStart, colStart + maxPerCol);
    const maxWidth = Math.max(...col.map((node) => node.width));
    const totalHeight = col.reduce((sum, node) => sum + node.height, 0);
    const totalGap = TOPOLOGY_ROW_GAP * Math.max(0, col.length - 1);
    let y = -((totalHeight + totalGap) / 2);

    for (const node of col) {
      const { width: _width, height, ...topologyNode } = node;
      y += height / 2;
      positioned.push({
        ...topologyNode,
        x,
        y,
      });
      y += height / 2 + TOPOLOGY_ROW_GAP;
    }

    x += maxWidth + TOPOLOGY_LAYER_GAP;
  }

  return positioned;
}

async function layoutTopologyNodesWithElk(
  nodes: BucketedTopologyNode[],
): Promise<PositionedTopologyNode[]> {
  const center = nodes.find((node) => node.bucket === "center");
  if (!center) return [];
  if (nodes.length === 1) {
    const { width: _width, height: _height, ...centerNode } = center;
    return [{ ...centerNode, x: 0, y: 0 }];
  }

  const realNodeIds = new Set(nodes.map((node) => node.id));

  // Distribute non-center nodes across partitions, capping each partition at
  // ceil(sqrt(count)) nodes so no single layer becomes excessively tall.
  const peerNodes = (["serving", "worker", "client"] as const).flatMap(
    (bucket) => nodes.filter((node) => node.bucket === bucket),
  );
  const maxPerPartition = Math.max(1, Math.ceil(Math.sqrt(peerNodes.length)));
  const nodePartition = new Map<string, number>([[center.id, 0]]);
  peerNodes.forEach((node, i) => {
    nodePartition.set(node.id, 1 + Math.floor(i / maxPerPartition));
  });

  const graphEdges = nodes
    .filter((node) => node.id !== center.id)
    .map((node) => ({
      id: `layout-${center.id}-${node.id}`,
      sources: [center.id],
      targets: [node.id],
    }));

  const graph = {
    id: "mesh-topology",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "RIGHT",
      "elk.partitioning.activate": "true",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
      "elk.layered.crossingMinimization.forceNodeModelOrder": "true",
      "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
      "elk.layered.spacing.nodeNodeBetweenLayers": String(TOPOLOGY_LAYER_GAP),
      "elk.spacing.nodeNode": String(TOPOLOGY_ROW_GAP),
      "elk.padding": `[top=${TOPOLOGY_PADDING},left=${TOPOLOGY_PADDING},bottom=${TOPOLOGY_PADDING},right=${TOPOLOGY_PADDING}]`,
    },
    children: nodes.map((node) => ({
      id: node.id,
      width: node.width,
      height: node.height,
      layoutOptions: {
        "elk.partitioning.partition": String(nodePartition.get(node.id) ?? 0),
      },
    })),
    edges: graphEdges,
  };

  const layout = await getTopologyElk().layout(graph);
  const positions = new Map<string, { x: number; y: number }>();

  for (const child of layout.children ?? []) {
    if (!realNodeIds.has(child.id)) continue;
    positions.set(child.id, {
      x: child.x ?? 0,
      y: child.y ?? 0,
    });
  }

  if (!nodes.every((node) => positions.has(node.id))) {
    return [];
  }

  const centerPosition = positions.get(center.id) ?? { x: 0, y: 0 };
  return nodes.map((node) => {
    const nodePosition = positions.get(node.id)!;
    const { width: _width, height: _height, ...topologyNode } = node;

    return {
      ...topologyNode,
      x: nodePosition.x - centerPosition.x,
      y: nodePosition.y - centerPosition.y,
    };
  });
}

function hasCompleteLayout(
  candidate: PositionedTopologyNode[],
  nodes: BucketedTopologyNode[],
) {
  const candidateIds = new Set(candidate.map((node) => node.id));
  return (
    candidate.length === nodes.length &&
    nodes.every((node) => candidateIds.has(node.id))
  );
}

export const elkTopologyLayout: TopologyLayoutDefinition = {
  label: "Tiered",
  edgeType: "smoothstep",
  direction: "horizontal",
  getImmediateLayout: ({ nodes }) => buildFallbackTopologyLayout(nodes),
  getLayout: async ({ nodes }) => {
    const fallbackLayout = buildFallbackTopologyLayout(nodes);

    try {
      const next = await layoutTopologyNodesWithElk(nodes);
      return hasCompleteLayout(next, nodes) ? next : fallbackLayout;
    } catch {
      return fallbackLayout;
    }
  },
};
