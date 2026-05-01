import { describe, expect, it } from "vitest";

import { buildLineMesh } from "./line-mesh";

describe("buildLineMesh", () => {
  it("expands a segment into a two-triangle quad with DPR-scaled width", () => {
    const mesh = buildLineMesh({
      positions: new Float32Array([0, 0, 10, 0]),
      colors: new Float32Array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
      lineWidthPx: 2,
      devicePixelRatio: 2,
    });

    expect([...mesh.positions]).toEqual([0, -2, 0, 2, 10, -2, 10, -2, 0, 2, 10, 2]);
    expect(mesh.colors).toHaveLength(24);
    expect([...mesh.lineCoords]).toEqual([0, -1, 0, 1, 1, -1, 1, -1, 0, 1, 1, 1]);
  });

  it("skips degenerate segments safely", () => {
    const mesh = buildLineMesh({
      positions: new Float32Array([12, 12, 12, 12]),
      colors: new Float32Array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
      lineWidthPx: 1.5,
      devicePixelRatio: 1,
    });

    expect(mesh.positions).toHaveLength(0);
    expect(mesh.colors).toHaveLength(0);
    expect(mesh.lineCoords).toHaveLength(0);
  });

  it("keeps along coordinates continuous across chained consecutive segments", () => {
    const mesh = buildLineMesh({
      positions: new Float32Array([0, 0, 4, 0, 4, 0, 10, 0]),
      colors: new Float32Array([
        0.2, 0.3, 0.4, 0.8,
        0.2, 0.3, 0.4, 0.8,
        0.2, 0.3, 0.4, 0.8,
        0.4, 0.5, 0.6, 0.2,
      ]),
      lineWidthPx: 1,
      devicePixelRatio: 1,
    });

    const alongValues = [...mesh.lineCoords].filter((_, index) => index % 2 === 0);
    const roundedAlongValues = alongValues.map((value) => Number(value.toFixed(4)));
    expect(roundedAlongValues).toEqual([0, 0, 0.4, 0.4, 0, 0.4, 0.4, 0.4, 1, 1, 0.4, 1]);
    expect(alongValues[6]).toBeCloseTo(0.4, 5);
  });

  it("does not chain separate logical edges that only happen to touch", () => {
    const mesh = buildLineMesh({
      positions: new Float32Array([0, 0, 4, 0, 4, 0, 10, 0]),
      colors: new Float32Array([
        0.2, 0.3, 0.4, 0.8,
        0.25, 0.35, 0.45, 0.4,
        0.2, 0.3, 0.4, 0.8,
        0.4, 0.5, 0.6, 0.2,
      ]),
      lineWidthPx: 1,
      devicePixelRatio: 1,
    });

    const alongValues = [...mesh.lineCoords].filter((_, index) => index % 2 === 0);
    expect(alongValues).toEqual([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]);
  });
});
