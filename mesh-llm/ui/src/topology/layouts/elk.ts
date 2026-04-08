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

/**
 * Simple tiered layout — center node on the left, peers arranged in columns
 * to the right.  Each column holds at most ceil(sqrt(N)) nodes so the graph
 * stays roughly square.  Nodes are vertically centered within their column.
 *
 * This replaces the previous ELK-based layout (elkjs added ~1.5 MB / 340 KB
 * brotli to the bundle) while producing visually equivalent results for the
 * star-shaped mesh topology.
 */
function layoutTiered(
  nodes: BucketedTopologyNode[],
): PositionedTopologyNode[] {
  const center = nodes.find((node) => node.bucket === "center");
  if (!center) return [];

  if (nodes.length === 1) {
    const { width: _w, height: _h, ...rest } = center;
    return [{ ...rest, x: 0, y: 0 }];
  }

  const { width: centerWidth, height: _ch, ...centerRest } = center;
  const positioned: PositionedTopologyNode[] = [
    { ...centerRest, x: 0, y: 0 },
  ];

  // Preserve bucket ordering: serving → worker → client
  const peerNodes = (["serving", "worker", "client"] as const).flatMap(
    (bucket) => nodes.filter((node) => node.bucket === bucket),
  );

  if (peerNodes.length === 0) return positioned;

  const maxPerCol = Math.max(1, Math.ceil(Math.sqrt(peerNodes.length)));
  let x = centerWidth + TOPOLOGY_LAYER_GAP;

  for (let colStart = 0; colStart < peerNodes.length; colStart += maxPerCol) {
    const col = peerNodes.slice(colStart, colStart + maxPerCol);
    const maxWidth = Math.max(...col.map((node) => node.width));
    const totalHeight = col.reduce((sum, node) => sum + node.height, 0);
    const totalGap = TOPOLOGY_ROW_GAP * Math.max(0, col.length - 1);
    let y = -((totalHeight + totalGap) / 2);

    for (const node of col) {
      const { width: _w, height, ...rest } = node;
      y += height / 2;
      positioned.push({ ...rest, x, y });
      y += height / 2 + TOPOLOGY_ROW_GAP;
    }

    x += maxWidth + TOPOLOGY_LAYER_GAP;
  }

  return positioned;
}

export const elkTopologyLayout: TopologyLayoutDefinition = {
  label: "Tiered",
  edgeType: "smoothstep",
  direction: "horizontal",
  getImmediateLayout: ({ nodes }) => layoutTiered(nodes),
  getLayout: async ({ nodes }) => layoutTiered(nodes),
};
