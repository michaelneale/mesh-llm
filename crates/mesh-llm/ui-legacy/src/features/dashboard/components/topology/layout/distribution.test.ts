import { describe, expect, it } from "vitest";

import {
  buildTopologyPairPlan,
  countBadTopologyEdgeCrossings,
  nodeSize,
  optimizeTopologyPlacementForPlan,
  type TopologyAngularPlacement,
  type TopologyGraphEdge,
  type TopologyGraphNode,
} from "./distribution";

type TestNode = {
  id: string;
};

function pointOnCircle(angle: number, radius: number, center = 0.5) {
  return {
    x: center + Math.cos(angle) * radius,
    y: center + Math.sin(angle) * radius,
  };
}

function createPlacement(
  id: string,
  angle: number,
  radius: number,
  overrides: Partial<Omit<TopologyAngularPlacement<TestNode>, "node" | "x" | "y" | "angle" | "positionAtAngle">> = {},
): TopologyAngularPlacement<TestNode> {
  const point = pointOnCircle(angle, radius);

  return {
    node: { id },
    x: point.x,
    y: point.y,
    angle,
    role: overrides.role ?? "Worker",
    selectedModelMatch: overrides.selectedModelMatch ?? false,
    locked: overrides.locked,
    latencyNorm: overrides.latencyNorm,
    positionAtAngle: (nextAngle) => pointOnCircle(nextAngle, radius),
  };
}

function flattenPlacements(
  bands: Array<Array<TopologyAngularPlacement<TestNode>>>,
  selfNode: TopologyGraphNode,
) {
  return [
    ...bands.flat().map((entry) => ({
      id: entry.node.id,
      x: entry.x,
      y: entry.y,
      role: entry.role,
      selectedModelMatch: entry.selectedModelMatch,
    })),
    selfNode,
  ];
}

function createGraphNode(
  id: string,
  x: number,
  y: number,
  role: TopologyGraphNode["role"] = "Worker",
): TopologyGraphNode {
  return {
    id,
    x,
    y,
    role,
    selectedModelMatch: false,
  };
}

function pairKeys(edges: TopologyGraphEdge[]) {
  return edges.map((edge) => edge.pairKey).sort();
}

describe("distribution crossing helpers", () => {
  it("does not count crossings when two edges only share an endpoint", () => {
    const nodes: TopologyGraphNode[] = [
      { id: "center", x: 0.5, y: 0.5, role: "Host", selectedModelMatch: false },
      { id: "top", x: 0.5, y: 0.2, role: "Worker", selectedModelMatch: false },
      { id: "bottom", x: 0.5, y: 0.8, role: "Worker", selectedModelMatch: false },
    ];
    const edges: TopologyGraphEdge[] = [
      {
        pairKey: "center::top",
        leftId: "center",
        rightId: "top",
        distance: 0.3,
        threshold: 1,
      },
      {
        pairKey: "bottom::center",
        leftId: "center",
        rightId: "bottom",
        distance: 0.3,
        threshold: 1,
      },
    ];

    expect(countBadTopologyEdgeCrossings(edges, nodes)).toBe(0);
  });

  it("deterministically swaps angular slots when that removes a bad crossing", () => {
    const selfNode: TopologyGraphNode = {
      id: "self",
      x: 0.5,
      y: 0.5,
      role: "Host",
      selectedModelMatch: false,
    };
    const lockedBand = [
      createPlacement("northwest", Math.PI * 1.25, 0.18, { locked: true }),
      createPlacement("northeast", Math.PI * 1.75, 0.18, { locked: true }),
    ];
    const movableBand = [
      createPlacement("southeast", Math.PI * 0.25, 0.32),
      createPlacement("southwest", Math.PI * 0.75, 0.32),
    ];
    const crossingPlan: TopologyGraphEdge[] = [
      {
        pairKey: "northwest::southeast",
        leftId: "northwest",
        rightId: "southeast",
        distance: 1,
        threshold: 1,
      },
      {
        pairKey: "northeast::southwest",
        leftId: "northeast",
        rightId: "southwest",
        distance: 1,
        threshold: 1,
      },
    ];

    const before = flattenPlacements([lockedBand, movableBand], selfNode);
    const optimizedBands = optimizeTopologyPlacementForPlan(
      [lockedBand, movableBand],
      selfNode,
      crossingPlan,
    );
    const after = flattenPlacements(optimizedBands, selfNode);
    const optimizedById = new Map(after.map((node) => [node.id, node]));
    const expectedLeftSlot = pointOnCircle(Math.PI * 0.75, 0.32);
    const expectedRightSlot = pointOnCircle(Math.PI * 0.25, 0.32);

    expect(countBadTopologyEdgeCrossings(crossingPlan, before)).toBe(1);
    expect(countBadTopologyEdgeCrossings(crossingPlan, after)).toBe(0);
    expect(optimizedById.get("southeast")?.x).toBeCloseTo(expectedLeftSlot.x, 6);
    expect(optimizedById.get("southwest")?.x).toBeCloseTo(expectedRightSlot.x, 6);
  });
});

describe("buildTopologyPairPlan", () => {
  it("keeps the cheapest relay chain when a node is first discovered by a more expensive root edge", () => {
    const nodes: TopologyGraphNode[] = [
      createGraphNode("root", 0.1, 0.5, "Host"),
      createGraphNode("n1", 0.251, 0.5),
      createGraphNode("n2", 0.402, 0.5),
      createGraphNode("n3", 0.553, 0.5),
      createGraphNode("n4", 0.704, 0.5),
      createGraphNode("n5", 0.855, 0.5),
    ];

    const plan = buildTopologyPairPlan(nodes, "root");

    expect(pairKeys(plan)).toEqual(
      ["n1::root", "n1::n2", "n2::n3", "n3::n4", "n4::n5"].sort(),
    );
  });

  it("connects a larger backbone without dropping nodes", () => {
    const step = 0.151;
    const root = createGraphNode("root", 0.05, 0.5, "Host");
    const workers = Array.from({ length: 6 }, (_, index) =>
      createGraphNode(`node-${index + 1}`, root.x + step * (index + 1), 0.5),
    );

    const plan = buildTopologyPairPlan([root, ...workers], "root");
    const connectedNodeIds = new Set(plan.flatMap((edge) => [edge.leftId, edge.rightId]));

    expect(plan).toHaveLength(workers.length);
    expect(connectedNodeIds).toEqual(new Set([root.id, ...workers.map((node) => node.id)]));
  });
});

describe("nodeSize", () => {
  it("uses GPU bandwidth as a small secondary signal when VRAM is tied", () => {
    const lowBandwidthSize = nodeSize(
      {
        id: "slow",
        vram: 16,
        self: false,
        host: false,
        client: false,
        serving: "",
        servingModels: [],
        state: "standby",
        gpus: [{ name: "slow", vram_bytes: 16 * 1024 ** 3, bandwidth_gbps: 120 }],
      },
      0,
    );
    const highBandwidthSize = nodeSize(
      {
        id: "fast",
        vram: 16,
        self: false,
        host: false,
        client: false,
        serving: "",
        servingModels: [],
        state: "standby",
        gpus: [{ name: "fast", vram_bytes: 16 * 1024 ** 3, bandwidth_gbps: 960 }],
      },
      0,
    );

    expect(highBandwidthSize).toBeGreaterThan(lowBandwidthSize);
    expect(highBandwidthSize - lowBandwidthSize).toBeLessThan(1.25);
  });

  it("keeps VRAM dominant over the capped bandwidth boost", () => {
    const lowerVramHighBandwidth = nodeSize(
      {
        id: "balanced",
        vram: 8,
        self: false,
        host: false,
        client: false,
        serving: "",
        servingModels: [],
        state: "standby",
        gpus: [{ name: "balanced", vram_bytes: 8 * 1024 ** 3, bandwidth_gbps: 4000 }],
      },
      0,
    );
    const higherVramNoBandwidth = nodeSize(
      {
        id: "vram-heavy",
        vram: 16,
        self: false,
        host: false,
        client: false,
        serving: "",
        servingModels: [],
        state: "standby",
        gpus: [{ name: "vram-heavy", vram_bytes: 16 * 1024 ** 3 }],
      },
      0,
    );

    expect(higherVramNoBandwidth).toBeGreaterThan(lowerVramHighBandwidth);
  });
});
