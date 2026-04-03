import { TOPOLOGY_NODE_WIDTH } from "./constants";
import type {
  BucketedTopologyNode,
  PositionedTopologyNode,
  TopologyLayoutDefinition,
  TopologyNode,
} from "./types";

const DEFAULT_TOPOLOGY_NODE_HEIGHT = Math.round(TOPOLOGY_NODE_WIDTH * 0.7);

function getClassicNodeFootprint(nodes: BucketedTopologyNode[]) {
  const centerNode = nodes.find((node) => node.bucket === "center");
  const peerNodes = nodes.filter((node) => node.bucket !== "center");

  return {
    centerWidth: Math.max(TOPOLOGY_NODE_WIDTH, centerNode?.width ?? 0),
    centerHeight: Math.max(
      DEFAULT_TOPOLOGY_NODE_HEIGHT,
      centerNode?.height ?? 0,
    ),
    peerWidth: Math.max(
      TOPOLOGY_NODE_WIDTH,
      ...peerNodes.map((node) => node.width),
    ),
    peerHeight: Math.max(
      DEFAULT_TOPOLOGY_NODE_HEIGHT,
      ...peerNodes.map((node) => node.height),
    ),
  };
}

export function layoutTopologyNodesClassic(
  center: TopologyNode,
  serving: TopologyNode[],
  workers: TopologyNode[],
  clients: TopologyNode[],
  nodeRadius: number,
  nodes: BucketedTopologyNode[] = [],
): PositionedTopologyNode[] {
  const placeRow = (
    row: TopologyNode[],
    bucket: PositionedTopologyNode["bucket"],
    y: number,
    horizontalSpacing: number,
    positioned: PositionedTopologyNode[],
  ) => {
    const startX = -((row.length - 1) * horizontalSpacing) / 2;
    row.forEach((node, index) => {
      positioned.push({
        ...node,
        bucket,
        x: startX + index * horizontalSpacing,
        y,
      });
    });
  };

  const all: Array<PositionedTopologyNode> = [
    ...serving.map((n) => ({ ...n, bucket: "serving" as const, x: 0, y: 0 })),
    ...workers.map((n) => ({ ...n, bucket: "worker" as const, x: 0, y: 0 })),
    ...clients.map((n) => ({ ...n, bucket: "client" as const, x: 0, y: 0 })),
  ];

  const positioned: PositionedTopologyNode[] = [
    { ...center, x: 0, y: 0, bucket: "center" },
  ];
  const peerCount = all.length;
  if (peerCount === 0) return positioned;

  const { centerWidth, centerHeight, peerWidth, peerHeight } =
    getClassicNodeFootprint(nodes);
  const horizontalGap = Math.max(40, Math.round(peerWidth * 0.18));
  const verticalGap = Math.max(28, Math.round(peerHeight * 0.16));
  const minCenterRadius = Math.max(
    (centerWidth + peerWidth) / 2 + horizontalGap,
    Math.max(centerHeight, peerHeight) + verticalGap,
  );
  const ringSpacing = Math.max(peerHeight + verticalGap, peerWidth * 0.72);
  const minArcLength = peerWidth + horizontalGap;

  if (peerCount <= 6) {
    const smallMeshHorizontalGap = Math.max(
      horizontalGap,
      Math.round((centerWidth + peerWidth) * 0.16),
    );
    const smallMeshVerticalGap = Math.max(
      verticalGap,
      Math.round((centerHeight + peerHeight) * 0.18),
    );
    const smallMeshHorizontalSpacing = peerWidth + smallMeshHorizontalGap;
    const smallMeshBandOffset =
      Math.max(centerHeight, peerHeight) + smallMeshVerticalGap;
    const smallMeshRowStep = peerHeight + smallMeshVerticalGap;

    const topRows: Array<{
      nodes: TopologyNode[];
      bucket: PositionedTopologyNode["bucket"];
    }> = [];
    const bottomRows: Array<{
      nodes: TopologyNode[];
      bucket: PositionedTopologyNode["bucket"];
    }> = [];

    if (serving.length) {
      topRows.push({ nodes: serving, bucket: "serving" });
    }

    if (workers.length) {
      if (serving.length === 0) {
        topRows.push({ nodes: workers, bucket: "worker" });
      } else {
        bottomRows.push({ nodes: workers, bucket: "worker" });
      }
    }

    if (clients.length) {
      bottomRows.push({ nodes: clients, bucket: "client" });
    }

    topRows.forEach((row, index) => {
      const distanceFromCenter =
        smallMeshBandOffset + (topRows.length - index - 1) * smallMeshRowStep;
      placeRow(
        row.nodes,
        row.bucket,
        -distanceFromCenter,
        smallMeshHorizontalSpacing,
        positioned,
      );
    });

    bottomRows.forEach((row, index) => {
      const distanceFromCenter = smallMeshBandOffset + index * smallMeshRowStep;
      placeRow(
        row.nodes,
        row.bucket,
        distanceFromCenter,
        smallMeshHorizontalSpacing,
        positioned,
      );
    });

    return positioned;
  }

  const baseRadius =
    peerCount <= 10
      ? Math.max(
          minCenterRadius,
          (peerCount * minArcLength) / (2 * Math.PI),
        )
      : Math.max(
          minCenterRadius,
          Math.ceil((8 * minArcLength) / (2 * Math.PI)),
          nodeRadius * 9 + 96,
        );

  if (peerCount <= 10) {
    for (let i = 0; i < peerCount; i += 1) {
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / peerCount;
      const node = all[i];
      positioned.push({
        ...node,
        x: Math.cos(angle) * baseRadius,
        y: Math.sin(angle) * baseRadius,
      });
    }
    return positioned;
  }

  let offset = 0;
  let ring = 0;
  while (offset < peerCount) {
    const radius = baseRadius + ring * ringSpacing;
    const capacity = Math.max(
      8,
      Math.floor((2 * Math.PI * radius) / minArcLength),
    );
    const take = Math.min(capacity, peerCount - offset);
    const phase = ring % 2 === 0 ? 0 : Math.PI / Math.max(6, take);
    for (let i = 0; i < take; i += 1) {
      const angle = -Math.PI / 2 + phase + (2 * Math.PI * i) / take;
      const node = all[offset + i];
      positioned.push({
        ...node,
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      });
    }
    offset += take;
    ring += 1;
  }

  return positioned;
}

export const classicTopologyLayout: TopologyLayoutDefinition = {
  label: "Ring",
  edgeType: "straight",
  direction: "vertical",
  getImmediateLayout: ({ center, serving, workers, clients, nodeRadius, nodes }) =>
    layoutTopologyNodesClassic(
      center,
      serving,
      workers,
      clients,
      nodeRadius,
      nodes,
    ),
  getLayout: async ({ center, serving, workers, clients, nodeRadius, nodes }) =>
    layoutTopologyNodesClassic(
      center,
      serving,
      workers,
      clients,
      nodeRadius,
      nodes,
    ),
};
