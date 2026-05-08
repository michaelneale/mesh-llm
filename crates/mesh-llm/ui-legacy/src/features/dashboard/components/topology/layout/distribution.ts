import type { TopologyNode } from "../../../../app-shell/lib/topology-types";

import { clamp, hashString, TAU } from "../helpers";
import type { RenderNode } from "../types";

const CROSSING_EPSILON = 1e-6;
const CROSSING_SWAP_PASSES = 4;

/** Binary min-heap keyed by distance for O((V+E) log V) Dijkstra. */
class MinHeap {
  private heap: [number, string][] = [];

  push(distance: number, id: string) {
    this.heap.push([distance, id]);
    this.bubbleUp(this.heap.length - 1);
  }

  pop(): [number, string] | undefined {
    if (this.heap.length === 0) return undefined;
    const min = this.heap[0];
    const last = this.heap.pop()!;
    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.sinkDown(0);
    }
    return min;
  }

  get size() {
    return this.heap.length;
  }

  private bubbleUp(i: number) {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.heap[parent][0] <= this.heap[i][0]) break;
      [this.heap[parent], this.heap[i]] = [this.heap[i], this.heap[parent]];
      i = parent;
    }
  }

  private sinkDown(i: number) {
    const n = this.heap.length;
    while (true) {
      let smallest = i;
      const left = 2 * i + 1;
      const right = 2 * i + 2;
      if (left < n && this.heap[left][0] < this.heap[smallest][0]) smallest = left;
      if (right < n && this.heap[right][0] < this.heap[smallest][0]) smallest = right;
      if (smallest === i) break;
      [this.heap[smallest], this.heap[i]] = [this.heap[i], this.heap[smallest]];
      i = smallest;
    }
  }
}

type Point = {
  x: number;
  y: number;
};

export type TopologyGraphNode = {
  id: string;
  x: number;
  y: number;
  role: string;
  selectedModelMatch: boolean;
};

export type TopologyGraphEdge = {
  pairKey: string;
  leftId: string;
  rightId: string;
  distance: number;
  threshold: number;
};

export type TopologyAngularPlacement<TNode extends { id: string }> = {
  node: TNode;
  x: number;
  y: number;
  angle: number;
  role: string;
  selectedModelMatch: boolean;
  locked?: boolean;
  latencyNorm?: number;
  positionAtAngle: (angle: number) => Point;
};

type WorkingPlacement<TNode extends { id: string }> = TopologyAngularPlacement<TNode> & {
  originalX: number;
  originalY: number;
};

type WorkingBand<TNode extends { id: string }> = {
  locked: WorkingPlacement<TNode>[];
  movableOrder: WorkingPlacement<TNode>[];
  slotAngles: number[];
};

function normalizeAngle(angle: number) {
  const wrapped = angle % TAU;
  return wrapped < 0 ? wrapped + TAU : wrapped;
}

function angleFromPoint(point: Point, center: Point) {
  return normalizeAngle(Math.atan2(point.y - center.y, point.x - center.x));
}

function placementComparator<TNode extends { id: string }>(
  left: TopologyAngularPlacement<TNode>,
  right: TopologyAngularPlacement<TNode>,
) {
  return left.angle - right.angle || left.node.id.localeCompare(right.node.id);
}

function isClientRole(role: string) {
  return role === "Client";
}

function pairKeyFor(leftId: string, rightId: string) {
  return [leftId, rightId].sort((first, second) => first.localeCompare(second)).join("::");
}

function proximityThreshold(
  left: Pick<TopologyGraphNode, "id" | "role">,
  right: Pick<TopologyGraphNode, "id" | "role">,
  centerNodeId?: string,
) {
  let threshold = 0.21;
  if (left.id === centerNodeId || right.id === centerNodeId) threshold += 0.03;
  if (isClientRole(left.role) || isClientRole(right.role)) threshold += 0.08;
  if (left.role === "Host" || right.role === "Host") threshold += 0.03;
  return threshold;
}

function clientAnchorThreshold(
  client: Pick<TopologyGraphNode, "id" | "role">,
  anchor: Pick<TopologyGraphNode, "id" | "role">,
  centerNodeId?: string,
) {
  const baseThreshold = proximityThreshold(client, anchor, centerNodeId);
  let threshold = Math.max(baseThreshold + 0.06, 0.35);
  if (anchor.role === "Host") threshold = Math.max(threshold, 0.39);
  else if (anchor.role === "Serving") threshold = Math.max(threshold, 0.37);
  else if (anchor.role === "Worker") threshold = Math.max(threshold, 0.36);
  if (anchor.id === centerNodeId) threshold += 0.01;
  return threshold;
}

// Ownership/invariants: this module owns topology pair planning, placement/crossing reduction,
// perimeter client placement, and node sizing. Render-space edge geometry belongs in
// `render/line-builders.ts`.
export function buildTopologyPairPlan(
  nodes: TopologyGraphNode[],
  centerNodeId?: string,
): TopologyGraphEdge[] {
  if (nodes.length < 2) {
    return [];
  }

  const backboneNodes = nodes.filter((node) => !isClientRole(node.role));
  if (!backboneNodes.length) {
    return [];
  }

  const rootNode =
    (centerNodeId ? backboneNodes.find((node) => node.id === centerNodeId) : undefined) ??
    backboneNodes[0];

  const rootRadius = (node: TopologyGraphNode) =>
    Math.hypot(node.x - rootNode.x, node.y - rootNode.y);
  const rootEdgePenalty = (left: TopologyGraphNode, right: TopologyGraphNode) => {
    if (left.id !== rootNode.id && right.id !== rootNode.id) return 0;
    const fartherRadius = Math.max(rootRadius(left), rootRadius(right));
    return Math.max(0, fartherRadius - 0.16) * 0.8;
  };
  const backboneThreshold = (left: TopologyGraphNode, right: TopologyGraphNode) => {
    let threshold = Math.max(proximityThreshold(left, right, centerNodeId) + 0.04, 0.3);
    if (left.id === rootNode.id || right.id === rootNode.id) threshold += 0.02;
    return threshold;
  };

  type CandidateEdge = TopologyGraphEdge & {
    cost: number;
  };

  const baseEdgesByKey = new Map<string, CandidateEdge>();
  const backboneById = new Map(backboneNodes.map((node) => [node.id, node]));
  const upsertBackboneEdge = (edge: CandidateEdge) => {
    const existing = baseEdgesByKey.get(edge.pairKey);
    if (
      !existing ||
      edge.cost < existing.cost ||
      (edge.cost === existing.cost && edge.distance < existing.distance)
    ) {
      baseEdgesByKey.set(edge.pairKey, edge);
    }
  };

  for (const node of backboneNodes) {
    const neighborCandidates = backboneNodes
      .filter((other) => other.id !== node.id)
      .map((other) => {
        const distance = Math.hypot(node.x - other.x, node.y - other.y);
        const threshold = backboneThreshold(node, other);
        return {
          other,
          distance,
          threshold,
          radialDelta: rootRadius(other) - rootRadius(node),
        };
      })
      .sort(
        (left, right) =>
          left.distance - right.distance || left.other.id.localeCompare(right.other.id),
      );

    if (!neighborCandidates.length) continue;

    const selectedCandidates = new Map<string, (typeof neighborCandidates)[number]>();
    const nearestOverall = neighborCandidates[0];
    const nearestInward = neighborCandidates.find((candidate) => candidate.radialDelta < -0.01);
    const inThreshold = neighborCandidates.filter(
      (candidate) => candidate.distance <= candidate.threshold,
    );
    const keepCount = node.id === rootNode.id ? 4 : 3;
    const isOuterBackboneNode = rootRadius(node) > 0.22;
    const bestInwardRelay = inThreshold.find(
      (candidate) => candidate.other.id !== rootNode.id && candidate.radialDelta < -0.015,
    );
    const suppressRootShortcut =
      node.id !== rootNode.id && isOuterBackboneNode && bestInwardRelay != null;

    for (const candidate of inThreshold.slice(0, keepCount)) {
      if (suppressRootShortcut && candidate.other.id === rootNode.id) continue;
      selectedCandidates.set(candidate.other.id, candidate);
    }
    if (nearestOverall && !(suppressRootShortcut && nearestOverall.other.id === rootNode.id)) {
      selectedCandidates.set(nearestOverall.other.id, nearestOverall);
    }
    if (nearestInward) {
      selectedCandidates.set(nearestInward.other.id, nearestInward);
    }

    for (const candidate of selectedCandidates.values()) {
      upsertBackboneEdge({
        pairKey: pairKeyFor(node.id, candidate.other.id),
        leftId: node.id,
        rightId: candidate.other.id,
        distance: candidate.distance,
        threshold: candidate.threshold,
        cost: candidate.distance + rootEdgePenalty(node, candidate.other),
      });
    }
  }

  const runShortestPathTree = (bridgeEdges: CandidateEdge[]) => {
    const adjacency = new Map<string, CandidateEdge[]>();
    for (const node of backboneNodes) {
      adjacency.set(node.id, []);
    }

    for (const edge of [...baseEdgesByKey.values(), ...bridgeEdges]) {
      adjacency.get(edge.leftId)?.push(edge);
      adjacency.get(edge.rightId)?.push(edge);
    }

    const distances = new Map(backboneNodes.map((node) => [node.id, Number.POSITIVE_INFINITY]));
    const previous = new Map(backboneNodes.map((node) => [node.id, null as string | null]));
    const visited = new Set<string>();
    distances.set(rootNode.id, 0);

    const heap = new MinHeap();
    heap.push(0, rootNode.id);

    while (heap.size > 0) {
      const [currentDistance, currentId] = heap.pop()!;
      if (visited.has(currentId)) continue;
      if (!Number.isFinite(currentDistance)) break;
      visited.add(currentId);

      for (const edge of adjacency.get(currentId) ?? []) {
        const neighborId = edge.leftId === currentId ? edge.rightId : edge.leftId;
        if (visited.has(neighborId)) continue;

        const nextDistance = currentDistance + edge.cost;
        const knownDistance = distances.get(neighborId) ?? Number.POSITIVE_INFINITY;
        if (nextDistance < knownDistance) {
          distances.set(neighborId, nextDistance);
          previous.set(neighborId, currentId);
          heap.push(nextDistance, neighborId);
        }
      }
    }

    return { distances, previous };
  };

  const bridgeEdges: CandidateEdge[] = [];
  let backboneTree = runShortestPathTree(bridgeEdges);
  while (
    backboneNodes.some(
      (node) =>
        !Number.isFinite(backboneTree.distances.get(node.id) ?? Number.POSITIVE_INFINITY),
    )
  ) {
    const reachable = backboneNodes.filter((node) =>
      Number.isFinite(backboneTree.distances.get(node.id) ?? Number.POSITIVE_INFINITY),
    );
    const unreachable = backboneNodes.filter(
      (node) =>
        !Number.isFinite(backboneTree.distances.get(node.id) ?? Number.POSITIVE_INFINITY),
    );

    let bestBridge: CandidateEdge | null = null;
    for (const left of reachable) {
      for (const right of unreachable) {
        const pairKey = pairKeyFor(left.id, right.id);
        if (
          baseEdgesByKey.has(pairKey) ||
          bridgeEdges.some((edge) => edge.pairKey === pairKey)
        ) {
          continue;
        }

        const distance = Math.hypot(left.x - right.x, left.y - right.y);
        const bridge = {
          pairKey,
          leftId: left.id,
          rightId: right.id,
          distance,
          threshold: backboneThreshold(left, right) * 1.2,
          cost: distance + rootEdgePenalty(left, right),
        };

        if (
          !bestBridge ||
          bridge.cost < bestBridge.cost ||
          (bridge.cost === bestBridge.cost && bridge.pairKey.localeCompare(bestBridge.pairKey) < 0)
        ) {
          bestBridge = bridge;
        }
      }
    }

    if (!bestBridge) break;
    bridgeEdges.push(bestBridge);
    backboneTree = runShortestPathTree(bridgeEdges);
  }

  const allBackboneEdges = new Map(baseEdgesByKey);
  for (const edge of bridgeEdges) {
    allBackboneEdges.set(edge.pairKey, edge);
  }

  const plannedPairs: TopologyGraphEdge[] = [];
  const renderedPairs = new Set<string>();
  const pushPlannedPair = (edge: TopologyGraphEdge) => {
    if (renderedPairs.has(edge.pairKey)) return;
    renderedPairs.add(edge.pairKey);
    plannedPairs.push(edge);
  };

  for (const node of backboneNodes) {
    if (node.id === rootNode.id) continue;
    const previousId = backboneTree.previous.get(node.id);
    if (!previousId) continue;

    const pairKey = pairKeyFor(node.id, previousId);
    const edge = allBackboneEdges.get(pairKey);
    const previousNode = backboneById.get(previousId);
    if (!edge || !previousNode) continue;

    pushPlannedPair({
      pairKey,
      leftId: previousNode.id,
      rightId: node.id,
      distance: edge.distance,
      threshold: edge.threshold,
    });
  }

  type ClientAttachment = {
    pairKey: string;
    anchor: TopologyGraphNode;
    distance: number;
    threshold: number;
    score: number;
  };

  const isBetterClientAttachment = (
    nextCandidate: ClientAttachment,
    currentCandidate: ClientAttachment | null,
  ) =>
    !currentCandidate ||
    nextCandidate.score < currentCandidate.score ||
    (nextCandidate.score === currentCandidate.score &&
      nextCandidate.pairKey.localeCompare(currentCandidate.pairKey) < 0);

  for (const client of nodes) {
    if (!isClientRole(client.role)) continue;

    let bestCandidate: ClientAttachment | null = null;
    let bestEmergencyCandidate: ClientAttachment | null = null;

    for (const anchor of backboneNodes) {
      if (anchor.id === client.id) continue;

      const pairKey = pairKeyFor(client.id, anchor.id);
      if (renderedPairs.has(pairKey)) continue;

      const distance = Math.hypot(client.x - anchor.x, client.y - anchor.y);
      const threshold = clientAnchorThreshold(client, anchor, centerNodeId);
      const score =
        distance +
        (anchor.id === rootNode.id ? 0.02 : 0) -
        (anchor.role === "Host"
          ? 0.045
          : anchor.role === "Serving"
            ? 0.03
            : anchor.role === "Worker"
              ? 0.026
              : 0) -
        (anchor.selectedModelMatch ? 0.008 : 0);

      const candidate = {
        pairKey,
        anchor,
        distance,
        threshold,
        score,
      };

      if (isBetterClientAttachment(candidate, bestEmergencyCandidate)) {
        bestEmergencyCandidate = candidate;
      }
      if (distance <= threshold && isBetterClientAttachment(candidate, bestCandidate)) {
        bestCandidate = candidate;
      }
    }

    const attachment = bestCandidate ?? bestEmergencyCandidate;
    if (!attachment) continue;

    pushPlannedPair({
      pairKey: attachment.pairKey,
      leftId: attachment.anchor.id,
      rightId: client.id,
      distance: attachment.distance,
      threshold: attachment.threshold * (bestCandidate ? 1 : 1.12),
    });
  }

  return plannedPairs;
}

function orientation(start: Point, middle: Point, end: Point) {
  const value =
    (middle.y - start.y) * (end.x - middle.x) -
    (middle.x - start.x) * (end.y - middle.y);
  if (Math.abs(value) <= CROSSING_EPSILON) return 0;
  return value > 0 ? 1 : -1;
}

function isInteriorPointOnSegment(point: Point, start: Point, end: Point) {
  if (
    (Math.abs(point.x - start.x) <= CROSSING_EPSILON &&
      Math.abs(point.y - start.y) <= CROSSING_EPSILON) ||
    (Math.abs(point.x - end.x) <= CROSSING_EPSILON &&
      Math.abs(point.y - end.y) <= CROSSING_EPSILON)
  ) {
    return false;
  }

  return (
    point.x <= Math.max(start.x, end.x) + CROSSING_EPSILON &&
    point.x >= Math.min(start.x, end.x) - CROSSING_EPSILON &&
    point.y <= Math.max(start.y, end.y) + CROSSING_EPSILON &&
    point.y >= Math.min(start.y, end.y) - CROSSING_EPSILON
  );
}

function segmentsCrossExcludingEndpoints(
  leftStart: Point,
  leftEnd: Point,
  rightStart: Point,
  rightEnd: Point,
) {
  const leftRightStart = orientation(leftStart, leftEnd, rightStart);
  const leftRightEnd = orientation(leftStart, leftEnd, rightEnd);
  const rightLeftStart = orientation(rightStart, rightEnd, leftStart);
  const rightLeftEnd = orientation(rightStart, rightEnd, leftEnd);

  if (leftRightStart !== leftRightEnd && rightLeftStart !== rightLeftEnd) {
    return true;
  }

  if (leftRightStart === 0 && isInteriorPointOnSegment(rightStart, leftStart, leftEnd)) {
    return true;
  }
  if (leftRightEnd === 0 && isInteriorPointOnSegment(rightEnd, leftStart, leftEnd)) {
    return true;
  }
  if (rightLeftStart === 0 && isInteriorPointOnSegment(leftStart, rightStart, rightEnd)) {
    return true;
  }
  if (rightLeftEnd === 0 && isInteriorPointOnSegment(leftEnd, rightStart, rightEnd)) {
    return true;
  }

  return false;
}

export function countBadTopologyEdgeCrossings(
  edges: TopologyGraphEdge[],
  nodes: TopologyGraphNode[],
) {
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  let crossings = 0;

  for (let leftIndex = 0; leftIndex < edges.length; leftIndex += 1) {
    const leftEdge = edges[leftIndex];
    const leftStart = nodeById.get(leftEdge.leftId);
    const leftEnd = nodeById.get(leftEdge.rightId);
    if (!leftStart || !leftEnd) continue;

    for (let rightIndex = leftIndex + 1; rightIndex < edges.length; rightIndex += 1) {
      const rightEdge = edges[rightIndex];
      if (
        leftEdge.leftId === rightEdge.leftId ||
        leftEdge.leftId === rightEdge.rightId ||
        leftEdge.rightId === rightEdge.leftId ||
        leftEdge.rightId === rightEdge.rightId
      ) {
        continue;
      }

      const rightStart = nodeById.get(rightEdge.leftId);
      const rightEnd = nodeById.get(rightEdge.rightId);
      if (!rightStart || !rightEnd) continue;

      if (
        segmentsCrossExcludingEndpoints(
          leftStart,
          leftEnd,
          rightStart,
          rightEnd,
        )
      ) {
        crossings += 1;
      }
    }
  }

  return crossings;
}

function buildAdjacency(edges: TopologyGraphEdge[]) {
  const adjacency = new Map<string, Set<string>>();

  for (const edge of edges) {
    if (!adjacency.has(edge.leftId)) adjacency.set(edge.leftId, new Set());
    if (!adjacency.has(edge.rightId)) adjacency.set(edge.rightId, new Set());
    adjacency.get(edge.leftId)?.add(edge.rightId);
    adjacency.get(edge.rightId)?.add(edge.leftId);
  }

  return adjacency;
}

function materializeBand<TNode extends { id: string }>(
  band: WorkingBand<TNode>,
): TopologyAngularPlacement<TNode>[] {
  const placedMovable = band.movableOrder.map((entry, index) => {
    const angle = band.slotAngles[index] ?? entry.angle;
    const point = entry.positionAtAngle(angle);
    return {
      ...entry,
      angle,
      x: point.x,
      y: point.y,
    };
  });

  return [...band.locked, ...placedMovable].sort(placementComparator);
}

function materializeBands<TNode extends { id: string }>(
  bands: WorkingBand<TNode>[],
): TopologyAngularPlacement<TNode>[][] {
  return bands.map((band) => materializeBand(band));
}

function flattenBands<TNode extends { id: string }>(
  bands: TopologyAngularPlacement<TNode>[][],
): TopologyGraphNode[] {
  return bands.flat().map((entry) => ({
    id: entry.node.id,
    x: entry.x,
    y: entry.y,
    role: entry.role,
    selectedModelMatch: entry.selectedModelMatch,
  }));
}

function scoreLayout<TNode extends { id: string }>(
  pairPlan: TopologyGraphEdge[],
  bands: WorkingBand<TNode>[],
  selfNode: TopologyGraphNode,
  originalPositions: Map<string, Point>,
) {
  const materializedBands = materializeBands(bands);
  const graphNodes = [...flattenBands(materializedBands), selfNode];
  const nodeById = new Map(graphNodes.map((node) => [node.id, node]));
  const edgeLength = pairPlan.reduce((total, edge) => {
    const left = nodeById.get(edge.leftId);
    const right = nodeById.get(edge.rightId);
    if (!left || !right) return total;
    return total + Math.hypot(left.x - right.x, left.y - right.y);
  }, 0);
  const movement = materializedBands.reduce((total, band) => {
    return (
      total +
      band.reduce((bandTotal, entry) => {
        if (entry.locked) return bandTotal;
        const original = originalPositions.get(entry.node.id);
        if (!original) return bandTotal;
        return bandTotal + Math.hypot(entry.x - original.x, entry.y - original.y);
      }, 0)
    );
  }, 0);

  return {
    badCrossings: countBadTopologyEdgeCrossings(pairPlan, graphNodes),
    edgeLength,
    movement,
  };
}

function compareLayoutScore(
  left: ReturnType<typeof scoreLayout>,
  right: ReturnType<typeof scoreLayout>,
) {
  if (left.badCrossings !== right.badCrossings) {
    return left.badCrossings - right.badCrossings;
  }
  if (Math.abs(left.edgeLength - right.edgeLength) > CROSSING_EPSILON) {
    return left.edgeLength - right.edgeLength;
  }
  if (Math.abs(left.movement - right.movement) > CROSSING_EPSILON) {
    return left.movement - right.movement;
  }
  return 0;
}

export function optimizeTopologyPlacementForPlan<TNode extends { id: string }>(
  bands: Array<Array<TopologyAngularPlacement<TNode>>>,
  selfNode: TopologyGraphNode,
  pairPlan: TopologyGraphEdge[],
) {
  if (!pairPlan.length) {
    return bands.map((band) => [...band].sort(placementComparator));
  }

  const originalPositions = new Map<string, Point>();
  const workingBands = bands.map<WorkingBand<TNode>>((band) => {
    const sortedBand = [...band]
      .map((entry) => {
        originalPositions.set(entry.node.id, { x: entry.x, y: entry.y });
        return {
          ...entry,
          originalX: entry.x,
          originalY: entry.y,
        } satisfies WorkingPlacement<TNode>;
      })
      .sort(placementComparator);

    return {
      locked: sortedBand.filter((entry) => entry.locked),
      movableOrder: sortedBand.filter((entry) => !entry.locked),
      slotAngles: sortedBand.filter((entry) => !entry.locked).map((entry) => entry.angle),
    };
  });

  const currentNodes = [
    ...flattenBands(materializeBands(workingBands)),
    selfNode,
  ];
  const currentNodeById = new Map(currentNodes.map((node) => [node.id, node]));
  const adjacency = buildAdjacency(pairPlan);
  const centerPoint = { x: selfNode.x, y: selfNode.y };

  for (const band of workingBands) {
    if (band.movableOrder.length < 2) continue;

    const targetAngles = new Map<string, number>();
    for (const entry of band.movableOrder) {
      const neighbors = [...(adjacency.get(entry.node.id) ?? [])]
        .map((neighborId) => currentNodeById.get(neighborId))
        .filter((neighbor): neighbor is TopologyGraphNode => neighbor != null);
      if (!neighbors.length) {
        targetAngles.set(entry.node.id, entry.angle);
        continue;
      }

      let sumX = 0;
      let sumY = 0;
      for (const neighbor of neighbors) {
        const neighborAngle = angleFromPoint(neighbor, centerPoint);
        const weight = 1 / Math.max(Math.hypot(neighbor.x - centerPoint.x, neighbor.y - centerPoint.y), 0.05);
        sumX += Math.cos(neighborAngle) * weight;
        sumY += Math.sin(neighborAngle) * weight;
      }
      targetAngles.set(
        entry.node.id,
        Math.abs(sumX) <= CROSSING_EPSILON && Math.abs(sumY) <= CROSSING_EPSILON
          ? entry.angle
          : normalizeAngle(Math.atan2(sumY, sumX)),
      );
    }

    band.movableOrder.sort(
      (left, right) =>
        (targetAngles.get(left.node.id) ?? left.angle) -
          (targetAngles.get(right.node.id) ?? right.angle) ||
        left.node.id.localeCompare(right.node.id),
    );
  }

  let bestScore = scoreLayout(pairPlan, workingBands, selfNode, originalPositions);

  for (let pass = 0; pass < CROSSING_SWAP_PASSES; pass += 1) {
    let improved = false;

    for (let bandIndex = 0; bandIndex < workingBands.length; bandIndex += 1) {
      const band = workingBands[bandIndex];
      if (band.movableOrder.length < 2) continue;

      for (let swapIndex = 0; swapIndex < band.movableOrder.length - 1; swapIndex += 1) {
        const nextOrder = [...band.movableOrder];
        [nextOrder[swapIndex], nextOrder[swapIndex + 1]] = [
          nextOrder[swapIndex + 1],
          nextOrder[swapIndex],
        ];

        const candidateBands = workingBands.map((entry, index) =>
          index === bandIndex ? { ...entry, movableOrder: nextOrder } : entry,
        );
        const candidateScore = scoreLayout(pairPlan, candidateBands, selfNode, originalPositions);
        if (compareLayoutScore(candidateScore, bestScore) < 0) {
          workingBands[bandIndex] = { ...band, movableOrder: nextOrder };
          bestScore = candidateScore;
          improved = true;
        }
      }
    }

    if (!improved) {
      break;
    }
  }

  return materializeBands(workingBands);
}

export function reduceTopologyCrossings<TNode extends { id: string }>(
  bands: Array<Array<TopologyAngularPlacement<TNode>>>,
  selfNode: TopologyGraphNode,
  centerNodeId?: string,
) {
  const pairPlan = buildTopologyPairPlan(
    [...flattenBands(bands), selfNode],
    centerNodeId,
  );
  return optimizeTopologyPlacementForPlan(bands, selfNode, pairPlan);
}

export function distributePerimeterClients(nodes: TopologyNode[]) {
  const placed: Array<{ x: number; y: number }> = [];

  return nodes.map((node) => {
    const angle = hashString(`${node.id}:angle`) * TAU;
    const edgeBias = 0.74 + hashString(`${node.id}:radius`) * 0.16;
    const tangentJitter = (hashString(`${node.id}:tangent`) - 0.5) * 0.06;
    const radialJitter = (hashString(`${node.id}:radial`) - 0.5) * 0.025;
    const positionAtAngle = (nextAngle: number) => {
      const tangentAngle = nextAngle + Math.PI / 2;
      const ellipseX = 0.5 + Math.cos(nextAngle) * 0.4 * edgeBias;
      const ellipseY = 0.5 + Math.sin(nextAngle) * 0.35 * edgeBias;
      return {
        x: clamp(
          ellipseX +
            Math.cos(tangentAngle) * tangentJitter +
            Math.cos(nextAngle) * radialJitter,
          0.08,
          0.92,
        ),
        y: clamp(
          ellipseY +
            Math.sin(tangentAngle) * tangentJitter +
            Math.sin(nextAngle) * radialJitter,
          0.1,
          0.9,
        ),
      };
    };
    let { x, y } = positionAtAngle(angle);

    for (const prior of placed) {
      const dx = x - prior.x;
      const dy = y - prior.y;
      const distance = Math.hypot(dx, dy);
      if (distance > 0 && distance < 0.032) {
        const push = (0.032 - distance) * 0.5;
        x = clamp(x + (dx / distance) * push, 0.08, 0.92);
        y = clamp(y + (dy / distance) * push, 0.1, 0.9);
      }
    }
    placed.push({ x, y });

    return {
      node,
      x,
      y,
      angle,
      positionAtAngle,
    };
  });
}

function normalizedLatency(
  node: TopologyNode,
  minLatency: number,
  maxLatency: number,
) {
  if (node.latencyMs == null || !Number.isFinite(node.latencyMs)) {
    return 0.45;
  }
  if (maxLatency <= minLatency) {
    return 0.2;
  }
  return clamp((node.latencyMs - minLatency) / (maxLatency - minLatency), 0, 1);
}

function radiusMixFromLatencyMs(
  node: TopologyNode,
  minLatency: number,
  maxLatency: number,
  radiusSeed: number,
  radialBias: number,
) {
  const latencyNorm = Math.pow(normalizedLatency(node, minLatency, maxLatency), 0.9);
  return clamp(latencyNorm * 0.78 + radiusSeed * 0.22 + radialBias, 0, 1);
}

export function distributeLatencyBand(
  nodes: TopologyNode[],
  minLatency: number,
  maxLatency: number,
  angleStart: number,
  angleEnd: number,
  innerRadiusX: number,
  outerRadiusX: number,
  innerRadiusY: number,
  outerRadiusY: number,
  armCount = 3,
  radialBias = 0,
  curveLimit = 0.7,
) {
  const angleSpan = angleEnd - angleStart;

  return nodes.map((node) => {
    const latencyNorm = normalizedLatency(node, minLatency, maxLatency);
    const identity = `${node.id}:${node.hostname ?? ""}:${node.serving}`;
    const armSeed = hashString(`${identity}:arm`);
    const armIndex = Math.floor(armSeed * armCount) % armCount;
    const armBandStart = angleStart + (armIndex / armCount) * angleSpan;
    const armBandEnd = angleStart + ((armIndex + 1) / armCount) * angleSpan;
    const angleSeed = hashString(`${identity}:angle`);
    const angle = armBandStart + angleSeed * (armBandEnd - armBandStart);
    const radiusSeed = hashString(`${identity}:radius`);
    const radiusMix = radiusMixFromLatencyMs(
      node,
      minLatency,
      maxLatency,
      radiusSeed,
      radialBias,
    );
    const radiusX = innerRadiusX + (outerRadiusX - innerRadiusX) * radiusMix;
    const radiusY = innerRadiusY + (outerRadiusY - innerRadiusY) * radiusMix;
    const tangentDrift =
      (hashString(`${identity}:tangent-drift`) - 0.5) * Math.min(0.05, curveLimit * 0.08);
    const radialDrift = (hashString(`${identity}:radial-drift`) - 0.5) * 0.024;
    const positionAtAngle = (nextAngle: number) => {
      const tangentAngle = nextAngle + Math.PI / 2;
      const driftX =
        Math.cos(tangentAngle) * tangentDrift + Math.cos(nextAngle) * radialDrift;
      const driftY =
        Math.sin(tangentAngle) * tangentDrift + Math.sin(nextAngle) * radialDrift;
      return {
        x: clamp(0.5 + Math.cos(nextAngle) * radiusX + driftX, 0.12, 0.88),
        y: clamp(0.52 + Math.sin(nextAngle) * radiusY + driftY, 0.16, 0.84),
      };
    };
    const { x, y } = positionAtAngle(angle);

    return {
      node,
      latencyNorm,
      x,
      y,
      angle,
      positionAtAngle,
    };
  });
}

export function nodeSize(node: TopologyNode, emphasis: number) {
  const base = node.client ? 8 : 10;
  const vramBoost = node.client ? 0 : Math.sqrt(Math.max(0, node.vram)) * 1.55;
  const maxBandwidthGbps = Math.max(
    0,
    ...(node.gpus?.map((gpu) => gpu.bandwidth_gbps ?? 0) ?? []),
  );
  const bandwidthBoost =
    node.client || maxBandwidthGbps <= 0 ? 0 : clamp(Math.sqrt(maxBandwidthGbps) / 20, 0, 1.25);
  return clamp(base + vramBoost + bandwidthBoost + emphasis, node.client ? 6 : 10, 38);
}

/**
 * Iterative repulsion-based overlap resolver.
 *
 * Instead of a single-pass sampling approach, this runs a bounded force
 * relaxation inspired by ECharts/D3 force layouts:
 *   1. Pairwise repulsion pushes overlapping nodes apart (magnitude ∝ overlap).
 *   2. A gentle restoration force pulls each node back toward its original
 *      band-placed position so the overall layout shape is preserved.
 *   3. Gravity toward the center node, weighted by VRAM (node size).
 *      High-VRAM backbone nodes get strong pull; clients get near-zero.
 *   4. Exponential cooling (friction *= COOLING_RATE) converges smoothly.
 *
 * Fixed nodes (e.g. the self/center node) exert repulsion but do not move.
 */
export function resolveNodeOverlap(
  nodes: RenderNode[],
  fixedNodes: RenderNode[] = [],
) {
  if (nodes.length === 0) return [];

  const MAX_ITERATIONS = 50;
  const MAX_NODE_SIZE = 38;
  const FRICTION_INITIAL = 0.5;
  const COOLING_RATE = 0.97;
  const CONVERGENCE_THRESHOLD = 0.0005;
  const MIN_SEP_BASE = 0.025;
  const MIN_SEP_SIZE_FACTOR = 0.003;
  const RESTORE_STRENGTH = 0.08;
  const GRAVITY_BASE = 0.06;
  const GRAVITY_CLIENT = 0.005;

  type WorkingNode = {
    id: string;
    x: number;
    y: number;
    size: number;
    origX: number;
    origY: number;
    fixed: boolean;
    gravity: number;
  };

  const gravityForNode = (node: RenderNode) =>
    node.role === "Client"
      ? GRAVITY_CLIENT
      : GRAVITY_BASE * (node.size / MAX_NODE_SIZE);

  const workingNodes: WorkingNode[] = [
    ...fixedNodes.map((node) => ({
      id: node.id,
      x: node.x,
      y: node.y,
      size: node.size,
      origX: node.x,
      origY: node.y,
      fixed: true,
      gravity: 0,
    })),
    ...nodes.map((node) => ({
      id: node.id,
      x: node.x,
      y: node.y,
      size: node.size,
      origX: node.x,
      origY: node.y,
      fixed: false,
      gravity: gravityForNode(node),
    })),
  ];

  const gravityCenterX = fixedNodes.length
    ? fixedNodes.reduce((sum, n) => sum + n.x, 0) / fixedNodes.length
    : 0.5;
  const gravityCenterY = fixedNodes.length
    ? fixedNodes.reduce((sum, n) => sum + n.y, 0) / fixedNodes.length
    : 0.52;

  let friction = FRICTION_INITIAL;

  for (let iter = 0; iter < MAX_ITERATIONS; iter += 1) {
    let maxDisplacement = 0;

    // --- pairwise repulsion for overlapping nodes ---
    for (let i = 0; i < workingNodes.length; i += 1) {
      for (let j = i + 1; j < workingNodes.length; j += 1) {
        const a = workingNodes[i];
        const b = workingNodes[j];
        if (a.fixed && b.fixed) continue;

        const dx = b.x - a.x;
        const dy = b.y - a.y;
        let d = Math.hypot(dx, dy);
        const minSep = MIN_SEP_BASE + (a.size + b.size) * MIN_SEP_SIZE_FACTOR;
        if (d >= minSep) continue;

        // push direction: from a toward b
        let nx: number;
        let ny: number;
        if (d < 1e-6) {
          // deterministic jitter for exactly-overlapping nodes
          const jitterAngle = hashString(`${a.id}:${b.id}:jitter`) * TAU;
          nx = Math.cos(jitterAngle);
          ny = Math.sin(jitterAngle);
          d = 1e-6;
        } else {
          nx = dx / d;
          ny = dy / d;
        }

        const overlap = minSep - d;
        const push = overlap * 0.5 * friction;

        if (a.fixed) {
          b.x += nx * push * 2;
          b.y += ny * push * 2;
          maxDisplacement = Math.max(maxDisplacement, push * 2);
        } else if (b.fixed) {
          a.x -= nx * push * 2;
          a.y -= ny * push * 2;
          maxDisplacement = Math.max(maxDisplacement, push * 2);
        } else {
          a.x -= nx * push;
          a.y -= ny * push;
          b.x += nx * push;
          b.y += ny * push;
          maxDisplacement = Math.max(maxDisplacement, push);
        }
      }
    }

    // --- restoration force toward original band position ---
    for (const node of workingNodes) {
      if (node.fixed) continue;
      node.x += (node.origX - node.x) * RESTORE_STRENGTH * friction;
      node.y += (node.origY - node.y) * RESTORE_STRENGTH * friction;
    }

    // --- gravity toward center, weighted by VRAM (size) ---
    for (const node of workingNodes) {
      if (node.fixed) continue;
      node.x += (gravityCenterX - node.x) * node.gravity * friction;
      node.y += (gravityCenterY - node.y) * node.gravity * friction;
    }

    // --- clamp to layout bounds ---
    for (const node of workingNodes) {
      if (node.fixed) continue;
      node.x = clamp(node.x, 0.08, 0.92);
      node.y = clamp(node.y, 0.1, 0.9);
    }

    friction *= COOLING_RATE;

    if (maxDisplacement < CONVERGENCE_THRESHOLD) break;
  }

  // map resolved positions back onto input nodes
  const resolvedPositions = new Map<string, { x: number; y: number }>();
  for (const wn of workingNodes) {
    if (!wn.fixed) {
      resolvedPositions.set(wn.id, { x: wn.x, y: wn.y });
    }
  }

  return nodes.map((node) => {
    const pos = resolvedPositions.get(node.id);
    return pos ? { ...node, x: pos.x, y: pos.y } : node;
  });
}
