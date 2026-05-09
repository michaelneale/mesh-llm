import { buildTopologyPairPlan, type TopologyGraphEdge } from "../layout/distribution";
import { clamp, mixColor, withAlpha } from "../helpers";
import type {
  ColorTuple,
  LineBuilderInput,
  LineBuilderOutput,
  ScreenNode,
} from "../types";
import { LIGHT_GRAPHITE, LIGHT_INK, LIGHT_SLATE } from "../theme/color-tokens";

const ROUTE_BLOCKER_CLEARANCE_PX = 6;
const ROUTE_POINT_EPSILON_PX = 0.5;

// --- Pair plan last-one cache ---
// buildTopologyPairPlan runs a shortest-path tree + client attachment algorithm.
// During transitions, buildProximityLines is called 2-3× per frame with different
// alpha/visibility params but identical node positions — the pair plan is the same.
// This cache also survives across frames when node positions don't change.
let _pairPlanFp = NaN;
let _pairPlanCache: TopologyGraphEdge[] = [];

function cachedBuildTopologyPairPlan(
  screenNodes: ScreenNode[],
  centerNodeId?: string,
): TopologyGraphEdge[] {
  let hash = 0x811c9dc5;
  hash = ((hash * 0x01000193) ^ screenNodes.length) | 0;
  if (centerNodeId) {
    for (let i = 0; i < centerNodeId.length; i++) {
      hash = ((hash * 0x01000193) ^ centerNodeId.charCodeAt(i)) | 0;
    }
  }
  for (const node of screenNodes) {
    hash = ((hash * 0x01000193) ^ (node.x * 1e4)) | 0;
    hash = ((hash * 0x01000193) ^ (node.y * 1e4)) | 0;
    hash = ((hash * 0x01000193) ^ node.role.charCodeAt(0)) | 0;
    hash = ((hash * 0x01000193) ^ node.role.length) | 0;
    hash = ((hash * 0x01000193) ^ (node.selectedModelMatch ? 1 : 0)) | 0;
    for (let i = 0; i < node.id.length; i++) {
      hash = ((hash * 0x01000193) ^ node.id.charCodeAt(i)) | 0;
    }
  }

  if (hash === _pairPlanFp) return _pairPlanCache;

  _pairPlanCache = buildTopologyPairPlan(screenNodes, centerNodeId);
  _pairPlanFp = hash;
  return _pairPlanCache;
}

type Point = {
  x: number;
  y: number;
};

function pushLineSegment(
  linePositions: number[],
  lineColors: number[],
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  fromColor: ColorTuple,
  toColor: ColorTuple,
  devicePixelRatio: number,
) {
  linePositions.push(
    fromX * devicePixelRatio,
    fromY * devicePixelRatio,
    toX * devicePixelRatio,
    toY * devicePixelRatio,
  );
  lineColors.push(
    fromColor[0],
    fromColor[1],
    fromColor[2],
    fromColor[3],
    toColor[0],
    toColor[1],
    toColor[2],
    toColor[3],
  );
}

function nodeBoundaryClearance(node: ScreenNode) {
  return Math.min(node.x, node.y, 1 - node.x, 1 - node.y);
}

function isClientNode(node: ScreenNode) {
  return node.role === "Client";
}

type LineRoute = {
  points: Point[];
  signature: string;
};

type RouteObstacle = {
  id: string;
  x: number;
  y: number;
  radius: number;
};

function nodeBodyRadius(node: ScreenNode) {
  return Math.max(node.size * 0.5, isClientNode(node) ? 7 : 8);
}

function interpolateColor(left: ColorTuple, right: ColorTuple, amount: number): ColorTuple {
  return [
    left[0] + (right[0] - left[0]) * amount,
    left[1] + (right[1] - left[1]) * amount,
    left[2] + (right[2] - left[2]) * amount,
    left[3] + (right[3] - left[3]) * amount,
  ];
}

function pointDistanceToSegment(point: Point, start: Point, end: Point) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  if (Math.abs(dx) <= ROUTE_POINT_EPSILON_PX && Math.abs(dy) <= ROUTE_POINT_EPSILON_PX) {
    return Math.hypot(point.x - start.x, point.y - start.y);
  }

  const projection = clamp(
    ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy),
    0,
    1,
  );
  const closestX = start.x + dx * projection;
  const closestY = start.y + dy * projection;
  return Math.hypot(point.x - closestX, point.y - closestY);
}

function segmentIntersectsObstacle(start: Point, end: Point, obstacle: RouteObstacle) {
  return pointDistanceToSegment({ x: obstacle.x, y: obstacle.y }, start, end) < obstacle.radius;
}

function routeLength(points: Point[]) {
  let total = 0;
  for (let index = 1; index < points.length; index += 1) {
    total += Math.hypot(points[index].x - points[index - 1].x, points[index].y - points[index - 1].y);
  }
  return total;
}

function compactRoutePoints(points: Point[]) {
  const compacted: Point[] = [];

  for (const point of points) {
    const previous = compacted[compacted.length - 1];
    if (
      previous &&
      Math.hypot(point.x - previous.x, point.y - previous.y) <= ROUTE_POINT_EPSILON_PX
    ) {
      continue;
    }
    compacted.push(point);
  }

  const simplified: Point[] = [];
  for (const point of compacted) {
    simplified.push(point);
    while (simplified.length >= 3) {
      const start = simplified[simplified.length - 3];
      const middle = simplified[simplified.length - 2];
      const end = simplified[simplified.length - 1];
      const cross = (middle.x - start.x) * (end.y - middle.y) - (middle.y - start.y) * (end.x - middle.x);
      if (Math.abs(cross) > ROUTE_POINT_EPSILON_PX) break;
      simplified.splice(simplified.length - 2, 1);
    }
  }

  return simplified;
}

function buildRouteSignature(
  mode: "straight" | "relay",
  options: { blockerIds?: string[] } = {},
) {
  return JSON.stringify({
    mode,
    blockerIds: [...(options.blockerIds ?? [])].sort(),
  });
}

function extractSignatureBlockerIds(signature: string): string[] {
  try {
    const parsed = JSON.parse(signature) as { blockerIds?: string[] };
    return parsed.blockerIds ?? [];
  } catch {
    return [];
  }
}

function buildRouteObstacles(screenNodes: ScreenNode[], endpointIds: Set<string>) {
  return screenNodes
    .filter((node) => !endpointIds.has(node.id))
    .map<RouteObstacle>((node) => ({
      id: node.id,
      x: node.px,
      y: node.py,
      radius: nodeBodyRadius(node) + ROUTE_BLOCKER_CLEARANCE_PX,
    }));
}

function pushRouteSegments(
  linePositions: number[],
  lineColors: number[],
  route: LineRoute,
  fromColor: ColorTuple,
  toColor: ColorTuple,
  devicePixelRatio: number,
) {
  if (route.points.length < 2) return;

  const totalLength = routeLength(route.points);
  if (totalLength <= ROUTE_POINT_EPSILON_PX) return;

  let traversed = 0;
  for (let index = 1; index < route.points.length; index += 1) {
    const start = route.points[index - 1];
    const end = route.points[index];
    const segmentLength = Math.hypot(end.x - start.x, end.y - start.y);
    if (segmentLength <= ROUTE_POINT_EPSILON_PX) continue;
    const startRatio = traversed / totalLength;
    traversed += segmentLength;
    const endRatio = traversed / totalLength;
    pushLineSegment(
      linePositions,
      lineColors,
      start.x,
      start.y,
      end.x,
      end.y,
      interpolateColor(fromColor, toColor, startRatio),
      interpolateColor(fromColor, toColor, endRatio),
      devicePixelRatio,
    );
  }
}

function buildConnectionPoint(
  node: ScreenNode,
  centerDistance: number,
  unitX: number,
  unitY: number,
  direction: 1 | -1,
) {
  const boundaryBoost = clamp((0.16 - nodeBoundaryClearance(node)) / 0.16, 0, 1);
  const inset = Math.min(
    centerDistance * 0.3,
    6 + node.size * (isClientNode(node) ? 0.5 : 0.38) + boundaryBoost * 8,
  );

  return {
    x: node.px + unitX * inset * direction,
    y: node.py + unitY * inset * direction,
  };
}

function buildStraightRoute(from: ScreenNode, to: ScreenNode): LineRoute {
  const dx = to.px - from.px;
  const dy = to.py - from.py;
  const distance = Math.hypot(dx, dy);
  if (distance <= 1) {
    return {
      points: [
        { x: from.px, y: from.py },
        { x: to.px, y: to.py },
      ],
      signature: buildRouteSignature("straight"),
    };
  }

  const unitX = dx / distance;
  const unitY = dy / distance;
  const start = buildConnectionPoint(from, distance, unitX, unitY, 1);
  const end = buildConnectionPoint(to, distance, unitX, unitY, -1);
  if (Math.hypot(end.x - start.x, end.y - start.y) < 8) {
    return {
      points: [
        { x: from.px, y: from.py },
        { x: to.px, y: to.py },
      ],
      signature: buildRouteSignature("straight"),
    };
  }

  return {
    points: [start, end],
    signature: buildRouteSignature("straight"),
  };
}

function buildRoutedLine(
  from: ScreenNode,
  to: ScreenNode,
  screenNodes: ScreenNode[],
  depth = 0,
): LineRoute | null {
  const straightRoute = buildStraightRoute(from, to);
  const endpointIds = new Set([from.id, to.id]);
  const obstacles = buildRouteObstacles(screenNodes, endpointIds);
  const blockingObstacles = obstacles.filter((obstacle) =>
    segmentIntersectsObstacle(straightRoute.points[0], straightRoute.points[1], obstacle),
  );
  if (blockingObstacles.length === 0) {
    return straightRoute;
  }

  if (depth >= 4) {
    return null;
  }

  const fromPoint = straightRoute.points[0];
  const sortedBlockers = blockingObstacles
    .map((obstacle) => ({
      ...obstacle,
      node: screenNodes.find((n) => n.id === obstacle.id),
      distFromStart: Math.hypot(obstacle.x - fromPoint.x, obstacle.y - fromPoint.y),
    }))
    .filter((b): b is typeof b & { node: ScreenNode } => b.node != null)
    .sort((a, b) => a.distFromStart - b.distFromStart);

  for (const blocker of sortedBlockers) {
    const intermediateNode = blocker.node;
    const routeAI = buildRoutedLine(from, intermediateNode, screenNodes, depth + 1);
    const routeIB = buildRoutedLine(intermediateNode, to, screenNodes, depth + 1);
    if (routeAI && routeIB) {
      const allBlockerIds = [
        intermediateNode.id,
        ...extractSignatureBlockerIds(routeAI.signature),
        ...extractSignatureBlockerIds(routeIB.signature),
      ];
      return {
        points: compactRoutePoints([...routeAI.points, ...routeIB.points]),
        signature: buildRouteSignature("relay", { blockerIds: allBlockerIds }),
      };
    }
  }

  return null;
}

function pushStraightInsetLineSegment(
  linePositions: number[],
  lineColors: number[],
  from: ScreenNode,
  to: ScreenNode,
  screenNodes: ScreenNode[],
  fromColor: ColorTuple,
  toColor: ColorTuple,
  devicePixelRatio: number,
): LineRoute | null {
  const route = buildRoutedLine(from, to, screenNodes);
  if (!route) return null;
  pushRouteSegments(linePositions, lineColors, route, fromColor, toColor, devicePixelRatio);
  return route;
}

function emptyLineBuilderOutput(): LineBuilderOutput {
  return {
    positions: new Float32Array(0),
    colors: new Float32Array(0),
    pairKeys: [],
    pairRouteSignatures: new Map(),
  };
}

export function buildProximityLines({
  screenNodes,
  centerNodeId,
  highlightedNodeIds,
  devicePixelRatio,
  lineTailAlpha,
  visiblePairKeys,
  pairAlphaOverrides,
}: LineBuilderInput): LineBuilderOutput {
  if (screenNodes.length < 2) {
    return emptyLineBuilderOutput();
  }

  const linePositions: number[] = [];
  const lineColors: number[] = [];
  const renderedPairs = new Set<string>();
  const pairRouteSignatures = new Map<string, string>();
  const nodesById = new Map(screenNodes.map((node) => [node.id, node]));
  const pairPlan = cachedBuildTopologyPairPlan(screenNodes, centerNodeId);

  const pushLine = (
    left: ScreenNode,
    right: ScreenNode,
    distance: number,
    threshold: number,
    pairKey: string,
  ) => {
    if (renderedPairs.has(pairKey)) return;
    if (visiblePairKeys && !visiblePairKeys.has(pairKey)) return;

    const pairAlphaScale = pairAlphaOverrides?.get(pairKey) ?? 1;
    if (pairAlphaScale <= 0.001) return;

    const touchesHighlight =
      highlightedNodeIds.has(left.id) || highlightedNodeIds.has(right.id);
    const touchesCenter = left.id === centerNodeId || right.id === centerNodeId;
    const touchesHost = left.role === "Host" || right.role === "Host";
    const touchesClient = isClientNode(left) || isClientNode(right);
    const revealFactor = Math.min(left.lineRevealProgress, right.lineRevealProgress);
    if (revealFactor <= 0.001) return;

    const boundaryBoost = Math.max(
      clamp((0.16 - nodeBoundaryClearance(left)) / 0.16, 0, 1),
      clamp((0.16 - nodeBoundaryClearance(right)) / 0.16, 0, 1),
    );
    const distanceRatio = clamp(distance / threshold, 0, 1);
    const proximityStrength = 1 - distanceRatio;
    const peripheralBoost = touchesCenter ? 0 : 0.035;
    const baseAlpha = clamp(
      0.22 +
        proximityStrength * 0.18 +
        (touchesCenter ? 0.105 : peripheralBoost + 0.012) +
        (touchesHighlight ? 0.145 : 0) +
        (touchesHost ? 0.055 : 0) +
        boundaryBoost * 0.09 -
        (touchesClient ? 0.055 : 0),
      0.22,
      touchesHighlight ? 0.62 : touchesClient ? 0.42 : 0.54,
    );
    const leftEmphasis =
      0.98 +
      (left.id === centerNodeId ? 0.12 : 0) +
      (left.selectedModelMatch ? 0.1 : 0) +
      (left.role === "Host" ? 0.07 : 0);
    const rightEmphasis =
      0.98 +
      (right.id === centerNodeId ? 0.12 : 0) +
      (right.selectedModelMatch ? 0.1 : 0) +
      (right.role === "Host" ? 0.07 : 0);
    const neutralMix =
      (touchesHighlight ? 0.42 : touchesCenter ? 0.36 : 0.33) + boundaryBoost * 0.08;
    const focusMix =
      (touchesHighlight ? 0.25 : touchesCenter ? 0.22 : 0.19) +
      boundaryBoost * 0.05 -
      (touchesClient ? 0.05 : 0);
    const targetInk = touchesClient ? LIGHT_SLATE : LIGHT_INK;
    const leftInk = mixColor(
      mixColor(left.lineColor, LIGHT_GRAPHITE, neutralMix),
      targetInk,
      focusMix,
    );
    const rightInk = mixColor(
      mixColor(right.lineColor, LIGHT_GRAPHITE, neutralMix),
      targetInk,
      focusMix,
    );
    const lineAlphaScale = 0.92;

    const route = pushStraightInsetLineSegment(
      linePositions,
      lineColors,
      left,
      right,
      screenNodes,
      withAlpha(
        leftInk,
        clamp(baseAlpha * leftEmphasis * 1.04, lineTailAlpha + 0.19, 0.64) *
          revealFactor *
          lineAlphaScale *
          pairAlphaScale,
      ),
      withAlpha(
        rightInk,
        clamp(baseAlpha * rightEmphasis * 0.46, lineTailAlpha + 0.03, 0.24) *
          revealFactor *
          lineAlphaScale *
          pairAlphaScale,
      ),
      devicePixelRatio,
    );
    if (!route) return;

    pairRouteSignatures.set(pairKey, route.signature);
    renderedPairs.add(pairKey);
  };

  for (const edge of pairPlan) {
    const left = nodesById.get(edge.leftId);
    const right = nodesById.get(edge.rightId);
    if (!left || !right) continue;
    pushLine(left, right, edge.distance, edge.threshold, edge.pairKey);
  }

  return {
    positions: new Float32Array(linePositions),
    colors: new Float32Array(lineColors),
    pairKeys: [...renderedPairs],
    pairRouteSignatures,
  };
}
