import { describe, expect, it } from "vitest";

import type { ScreenNode } from "../types";
import { buildProximityLines } from "./line-builders";
function createScreenNode(
  id: string,
  overrides: Partial<ScreenNode> = {},
): ScreenNode {
  const x = overrides.x ?? 0.5;
  const y = overrides.y ?? 0.5;
  const size = overrides.size ?? 16;

  return {
    id,
    label: id,
    subtitle: id,
    role: "Worker",
    latencyLabel: "0ms",
    vramLabel: "0 GB",
    modelLabel: "",
    gpuLabel: "",
    x,
    y,
    size,
    color: [0.2, 0.3, 0.4, 1],
    lineColor: [0.2, 0.3, 0.4, 1],
    pulse: 0,
    selectedModelMatch: false,
    z: 0,
    px: x * 1000,
    py: y * 1000,
    hitSize: size,
    lineRevealProgress: 1,
    ...overrides,
  };
}

describe("buildProximityLines", () => {
  it("keeps unobstructed forced pairs straight and reports a straight route signature", () => {
    const lines = buildProximityLines({
      screenNodes: [
        createScreenNode("a", { role: "Host", x: 0.24, y: 0.5, size: 18 }),
        createScreenNode("c", { x: 0.76, y: 0.5, size: 16 }),
        createScreenNode("observer", { role: "Client", x: 0.5, y: 0.72, size: 18 }),
      ],
      centerNodeId: "a",
      highlightedNodeIds: new Set(),
      devicePixelRatio: 1,
      lineTailAlpha: 0.04,
      visiblePairKeys: new Set(["a::c"]),
    });

    expect(lines.pairKeys).toEqual(["a::c"]);
    expect(lines.positions).toHaveLength(4);
    expect(lines.pairRouteSignatures?.get("a::c")).toBe(
      JSON.stringify({ mode: "straight", blockerIds: [] }),
    );
    const yValues = [...lines.positions].filter((_, index) => index % 2 === 1);
    expect(yValues.every((value) => Math.abs(value - 500) < 0.001)).toBe(true);
  });

  it("routes blocked forced pairs through intervening node bodies as relay waypoints and exposes the route signature", () => {
    const lines = buildProximityLines({
      screenNodes: [
        createScreenNode("a", { role: "Host", x: 0.24, y: 0.5, size: 18 }),
        createScreenNode("c", { x: 0.76, y: 0.5, size: 16 }),
        createScreenNode("blocker", { role: "Client", x: 0.5, y: 0.51, size: 24 }),
      ],
      centerNodeId: "a",
      highlightedNodeIds: new Set(),
      devicePixelRatio: 1,
      lineTailAlpha: 0.04,
      visiblePairKeys: new Set(["a::c"]),
    });

    expect(lines.pairKeys).toEqual(["a::c"]);
    expect(lines.positions.length).toBeGreaterThan(4);
    expect(lines.positions.length % 4).toBe(0);
    expect(lines.pairRouteSignatures?.get("a::c")).toContain('"mode":"relay"');
    expect(lines.pairRouteSignatures?.get("a::c")).toContain('"blockerIds":["blocker"]');
  });

  it("relays blocked forced pairs through a huge intervening node body when no clear path exists", () => {
    const lines = buildProximityLines({
      screenNodes: [
        createScreenNode("a", { role: "Host", x: 0.24, y: 0.5, size: 18 }),
        createScreenNode("c", { x: 0.76, y: 0.5, size: 16 }),
        createScreenNode("blocker", { role: "Client", x: 0.5, y: 0.5, size: 1200 }),
      ],
      centerNodeId: "a",
      highlightedNodeIds: new Set(),
      devicePixelRatio: 1,
      lineTailAlpha: 0.04,
      visiblePairKeys: new Set(["a::c"]),
    });

    expect(lines.pairKeys).toEqual(["a::c"]);
    expect(lines.positions.length).toBeGreaterThan(0);
    expect(lines.pairRouteSignatures?.get("a::c")).toContain('"mode":"relay"');
  });

  it("falls back to a single direct segment when nodes are too close for an inset", () => {
    const lines = buildProximityLines({
      screenNodes: [
        createScreenNode("a", { role: "Host", x: 0.49, y: 0.5, size: 18 }),
        createScreenNode("b", { x: 0.51, y: 0.5, size: 16 }),
      ],
      centerNodeId: "a",
      highlightedNodeIds: new Set(),
      devicePixelRatio: 1,
      lineTailAlpha: 0.04,
      visiblePairKeys: new Set(["a::b"]),
    });

    expect(lines.pairKeys).toEqual(["a::b"]);
    expect(lines.positions).toHaveLength(4);
    expect([...lines.positions].filter((_, index) => index % 2 === 1)).toEqual([500, 500]);
  });

  it("draws exactly one straight segment per rendered proximity pair", () => {
    const lines = buildProximityLines({
      screenNodes: [
        createScreenNode("root", { role: "Host", x: 0.34, y: 0.5, size: 18 }),
        createScreenNode("mid", { x: 0.5, y: 0.5 }),
        createScreenNode("leaf", { x: 0.66, y: 0.5 }),
      ],
      centerNodeId: "root",
      highlightedNodeIds: new Set(),
      devicePixelRatio: 1,
      lineTailAlpha: 0.04,
    });
    const pairKeys = lines.pairKeys ?? [];

    expect(new Set(pairKeys).size).toBe(pairKeys.length);
    const yValues = [...lines.positions].filter((_, index) => index % 2 === 1);
    expect(yValues.every((value) => Math.abs(value - 500) < 0.001)).toBe(true);
  });
});
