import { describe, expect, it, vi } from "vitest";

import {
  DARK_LINE_FRAGMENT_SHADER,
  LIGHT_LINE_FRAGMENT_SHADER,
} from "../render/shaders";
import type { ScreenNode } from "../types";
import { RENDER_VARIANTS } from "./render-variants";

function createScreenNode(id: string, x: number, y: number): ScreenNode {
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
    px: x * 800,
    py: y * 600,
    hitSize: 16,
    size: 16,
    color: [0.4, 0.7, 0.9, 1],
    lineColor: [0.4, 0.7, 0.9, 1],
    pulse: 0,
    selectedModelMatch: false,
    z: 1,
    lineRevealProgress: 1,
  };
}

describe("RENDER_VARIANTS.dark", () => {
  it("renders connective lines even without highlighted nodes", () => {
    const screenNodes = [createScreenNode("self", 0.42, 0.42), createScreenNode("peer", 0.58, 0.44)];
    const input = {
      screenNodes,
      centerNodeId: "self",
      highlightedNodeIds: new Set<string>(),
      devicePixelRatio: 1,
      lineTailAlpha: RENDER_VARIANTS.dark.lineTailAlpha,
    };

    const variantOutput = RENDER_VARIANTS.dark.buildLines(input);

    expect(variantOutput.positions.length).toBeGreaterThan(0);
    expect(variantOutput.pairKeys).toEqual(["peer::self"]);
  });

  it("uses a dedicated dark dodge-style line shader and blend mode", () => {
    const gl = {
      ONE: 1,
      ONE_MINUS_SRC_ALPHA: 0x0303,
      SRC_ALPHA: 0x0302,
      blendFunc: vi.fn(),
      blendFuncSeparate: vi.fn(),
    } as unknown as WebGLRenderingContext;

    expect(RENDER_VARIANTS.dark.lineFragmentShader).toBe(DARK_LINE_FRAGMENT_SHADER);
    expect(RENDER_VARIANTS.dark.lineWidthPx).toBeGreaterThan(RENDER_VARIANTS.light.lineWidthPx);
    expect(DARK_LINE_FRAGMENT_SHADER).toContain("lineSmoothstep(0.0, 0.18, abs(v_lineCoord.y))");
    expect(DARK_LINE_FRAGMENT_SHADER).toContain("vec3(1.0, 0.985, 0.955)");
    expect(DARK_LINE_FRAGMENT_SHADER).toContain("core * 0.74");

    RENDER_VARIANTS.dark.applyLineBlendMode(gl);

    expect(gl.blendFunc).toHaveBeenCalledWith(gl.SRC_ALPHA, gl.ONE);
    expect(gl.blendFuncSeparate).not.toHaveBeenCalled();
  });
});

describe("RENDER_VARIANTS.light", () => {
  it("keeps the light line shader and blend mode on the normal path", () => {
    const gl = {
      ONE: 1,
      ONE_MINUS_SRC_ALPHA: 0x0303,
      SRC_ALPHA: 0x0302,
      blendFunc: vi.fn(),
      blendFuncSeparate: vi.fn(),
    } as unknown as WebGLRenderingContext;

    expect(RENDER_VARIANTS.light.lineFragmentShader).toBe(LIGHT_LINE_FRAGMENT_SHADER);
    expect(RENDER_VARIANTS.light.lineWidthPx).toBeGreaterThan(0);

    RENDER_VARIANTS.light.applyLineBlendMode(gl);

    expect(gl.blendFunc).not.toHaveBeenCalled();
    expect(gl.blendFuncSeparate).toHaveBeenCalledWith(
      gl.SRC_ALPHA,
      gl.ONE_MINUS_SRC_ALPHA,
      gl.ONE,
      gl.ONE_MINUS_SRC_ALPHA,
    );
  });
});
