// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";
import { cleanup, render } from "@testing-library/react";
import { type MutableRefObject } from "react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import { ENTRY_ANIMATION_DURATION_MS, LINE_REVEAL_DURATION_MS } from "../helpers";
import { RENDER_VARIANTS } from "../theme/render-variants";
import type {
  EntryAnimation,
  ExitAnimation,
  LineTransition,
  LineRevealAnimation,
  PendingLineTransition,
  RenderNode,
  RenderVariant,
  ScreenNode,
  UpdateTwinkle,
} from "../types";
import { buildLineMesh } from "./line-mesh";
import { useTopologyCanvas } from "./useTopologyCanvas";

type ScenarioName =
  | "join"
  | "join-route-change"
  | "join-spillover"
  | "join-spillover-incoming-only"
  | "leave"
  | "leave-spillover";

const mocks = vi.hoisted(() => {
  const state = { scenario: "join" as ScenarioName };

  const buildProximityLinesMock = vi.fn(
    (input: {
      screenNodes: Array<{
        id: string;
        size: number;
        hitSize: number;
        lineRevealProgress: number;
      }>;
      highlightedNodeIds: Set<string>;
      visiblePairKeys?: Set<string>;
      pairAlphaOverrides?: Map<string, number>;
    }) => {
      const ids = new Set(input.screenNodes.map((node) => node.id));

      let pairKeys: string[] = [];
      let pairRouteSignatures = new Map<string, string>();
      if (input.visiblePairKeys) {
        pairKeys = [...input.visiblePairKeys];
      } else if (state.scenario === "join") {
        pairKeys = ids.has("new-node")
          ? ["anchor::new-node", "new-node::peer"]
          : ["anchor::peer", "stable-a::stable-b"];
      } else if (state.scenario === "join-route-change") {
        pairKeys = ["anchor::peer", "stable-a::stable-b"];
      } else if (state.scenario === "join-spillover") {
        pairKeys = ids.has("new-node")
          ? [
              "anchor::new-node",
              "new-node::peer",
              "peer::worker-2",
              "worker::remote-2",
              "stable-a::stable-b",
            ]
          : [
              "anchor::peer",
              "peer::worker",
              "worker::remote",
              "stable-a::stable-b",
            ];
      } else if (state.scenario === "join-spillover-incoming-only") {
        pairKeys = ids.has("new-node")
          ? [
              "anchor::new-node",
              "new-node::peer",
              "worker::remote-2",
              "peer::worker-2",
              "stable-a::stable-b",
            ]
          : ["stable-a::stable-b", "worker::remote", "peer::worker"];
      } else if (state.scenario === "leave") {
        pairKeys = ids.has("replacement")
          ? ["peer::replacement"]
          : ["peer::removed-node", "stable-a::stable-b"];
      } else {
        pairKeys = ids.has("replacement")
          ? [
              "peer::replacement",
              "replacement::worker-2",
              "worker-2::remote-2",
              "stable-a::stable-b",
            ]
          : ["peer::removed-node", "stable-a::stable-b"];
      }

      pairRouteSignatures = new Map(
        pairKeys.map((pairKey) => {
          if (state.scenario === "join-route-change" && pairKey === "anchor::peer") {
            const signature = ids.has("new-node")
              ? JSON.stringify({
                  mode: "detour",
                  axis: "y",
                  side: "up",
                  blockerIds: ["new-node"],
                })
              : JSON.stringify({ mode: "straight", blockerIds: [] });
            return [pairKey, signature];
          }

          return [pairKey, JSON.stringify({ mode: "straight", blockerIds: [] })];
        }),
      );

      return {
        positions: new Float32Array(),
        colors: new Float32Array(),
        pairKeys,
        pairRouteSignatures,
      };
    },
  );

  const createProgramMock = vi.fn(() => ({}));
  const buildLineFragmentShaderSourceMock = vi.fn(
    (fragmentSource: string, options: { useStandardDerivatives: boolean }) =>
      options.useStandardDerivatives ? `derivatives:${fragmentSource}` : fragmentSource,
  );

  return {
    state,
    buildProximityLinesMock,
    createProgramMock,
    buildLineFragmentShaderSourceMock,
  };
});

vi.mock("./line-builders", () => ({
  buildProximityLines: mocks.buildProximityLinesMock,
}));

vi.mock("./shaders", () => ({
  buildLineFragmentShaderSource: mocks.buildLineFragmentShaderSourceMock,
  createProgram: mocks.createProgramMock,
  DARK_LINE_FRAGMENT_SHADER: "dark-line-fragment-shader",
  DARK_POINT_FRAGMENT_SHADER: "dark-point-fragment-shader",
  LIGHT_LINE_FRAGMENT_SHADER: "light-line-fragment-shader",
  LIGHT_POINT_FRAGMENT_SHADER: "light-point-fragment-shader",
  LINE_VERTEX_SHADER: "line-vertex-shader",
  POINT_VERTEX_SHADER: "point-vertex-shader",
}));

class MockResizeObserver {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

const clock = { now: 0 };
const rafState = {
  nextId: 1,
  callbacks: new Map<number, FrameRequestCallback>(),
};

let supportsStandardDerivatives = false;

let activeGl: ReturnType<typeof createFakeWebGLContext>;

function flushAnimationFrame(timestamp = clock.now) {
  const callbacks = [...rafState.callbacks.values()];
  rafState.callbacks.clear();
  for (const callback of callbacks) {
    callback(timestamp);
  }
}

function createFakeWebGLContext() {
  const attributeLocations = new Map<string, number>([
    ["a_position", 0],
    ["a_size", 1],
    ["a_color", 2],
    ["a_pulse", 3],
    ["a_twinkle", 4],
    ["a_lineCoord", 5],
  ]);

  const uniformLocation = {} as WebGLUniformLocation;
  const buffer = {} as WebGLBuffer;
  const program = {} as WebGLProgram;

  return {
    ARRAY_BUFFER: 0x8892,
    BLEND: 0x0be2,
    COLOR_BUFFER_BIT: 0x4000,
    DYNAMIC_DRAW: 0x88e8,
    FLOAT: 0x1406,
    LINES: 0x0001,
    ONE: 1,
    ONE_MINUS_SRC_ALPHA: 0x0303,
    POINTS: 0x0000,
    SRC_ALPHA: 0x0302,
    TRIANGLES: 0x0004,
    blendFunc: vi.fn(),
    blendFuncSeparate: vi.fn(),
    bindBuffer: vi.fn(),
    bufferData: vi.fn(),
    bufferSubData: vi.fn(),
    clear: vi.fn(),
    clearColor: vi.fn(),
    createBuffer: vi.fn(() => buffer),
    deleteBuffer: vi.fn(),
    deleteProgram: vi.fn(),
    disableVertexAttribArray: vi.fn(),
    drawArrays: vi.fn(),
    enable: vi.fn(),
    enableVertexAttribArray: vi.fn(),
    getAttribLocation: vi.fn((_program: WebGLProgram, name: string) =>
      attributeLocations.get(name) ?? 0,
    ),
    getExtension: vi.fn((name: string) =>
      name === "OES_standard_derivatives" && supportsStandardDerivatives ? {} : null,
    ),
    getUniformLocation: vi.fn(() => uniformLocation),
    lineWidth: vi.fn(),
    uniform1f: vi.fn(),
    uniform2f: vi.fn(),
    useProgram: vi.fn(),
    vertexAttribPointer: vi.fn(),
    viewport: vi.fn(),
    createProgram: vi.fn(() => program),
  };
}

function createRenderNode(id: string, overrides: Partial<RenderNode> = {}): RenderNode {
  return {
    id,
    label: id,
    subtitle: id,
    role: "Worker",
    latencyLabel: "0ms",
    vramLabel: "0 GB",
    modelLabel: "",
    gpuLabel: "",
    x: 0.1,
    y: 0.1,
    size: 10,
    color: [0.2, 0.3, 0.4, 1],
    lineColor: [0.2, 0.3, 0.4, 1],
    pulse: 0,
    selectedModelMatch: false,
    z: 0,
    ...overrides,
  };
}

function createScreenNode(node: RenderNode): ScreenNode {
  return {
    ...node,
    px: node.x * 800,
    py: node.y * 600,
    hitSize: node.size,
    lineRevealProgress: 1,
  };
}

function createMutableRef<T>(current: T): MutableRefObject<T> {
  return { current };
}

function Harness({
  animationRef,
  canvasRef,
  exitAnimationRef,
  hostRef,
  hoveredNodeIdRef,
  lastScreenPositionsRef,
  lineTransitionRef,
  lineRevealRef,
  panRef,
  pendingLineTransitionRef,
  renderNodes,
  renderVariant = RENDER_VARIANTS.light,
  screenNodesRef,
  seenNodeIdsRef,
  selfNodeId,
  twinkleAnimationRef,
  zoomRef,
}: {
  canvasRef: MutableRefObject<HTMLCanvasElement | null>;
  hostRef: MutableRefObject<HTMLDivElement | null>;
  screenNodesRef: MutableRefObject<ScreenNode[]>;
  animationRef: MutableRefObject<Map<string, EntryAnimation>>;
  lineRevealRef: MutableRefObject<Map<string, LineRevealAnimation>>;
  exitAnimationRef: MutableRefObject<Map<string, ExitAnimation>>;
  twinkleAnimationRef: MutableRefObject<Map<string, UpdateTwinkle>>;
  pendingLineTransitionRef: MutableRefObject<PendingLineTransition | null>;
  lineTransitionRef: MutableRefObject<LineTransition | null>;
  lastScreenPositionsRef: MutableRefObject<Map<string, { x: number; y: number }>>;
  seenNodeIdsRef: MutableRefObject<Set<string>>;
  hoveredNodeIdRef: MutableRefObject<string | null>;
  zoomRef: MutableRefObject<number>;
  panRef: MutableRefObject<{ x: number; y: number }>;
  renderNodes: RenderNode[];
  renderVariant?: RenderVariant;
  selfNodeId?: string;
}) {
  useTopologyCanvas({
    animationRef,
    canvasRef,
    exitAnimationRef,
    hostRef,
    hoveredNodeIdRef,
    lastScreenPositionsRef,
    lineTransitionRef,
    lineRevealRef,
    panRef,
    pendingLineTransitionRef,
    renderNodes,
    renderVariant,
    screenNodesRef,
    seenNodeIdsRef,
    selfNodeId,
    twinkleAnimationRef,
    zoomRef,
  });

  return (
    <div ref={hostRef}>
      <canvas ref={canvasRef} />
    </div>
  );
}

describe("useTopologyCanvas light transition regressions", () => {
  beforeAll(() => {
    Object.defineProperty(window, "ResizeObserver", {
      configurable: true,
      writable: true,
      value: MockResizeObserver,
    });

    Object.defineProperty(globalThis, "WebGLRenderingContext", {
      configurable: true,
      writable: true,
      value: function WebGLRenderingContext() {},
    });

    Object.defineProperty(window, "devicePixelRatio", {
      configurable: true,
      value: 1,
    });

    Object.defineProperty(HTMLElement.prototype, "getBoundingClientRect", {
      configurable: true,
      value: () => ({
        width: 800,
        height: 600,
        x: 0,
        y: 0,
        top: 0,
        left: 0,
        right: 800,
        bottom: 600,
        toJSON: () => ({}),
      }),
    });

    Object.defineProperty(window, "requestAnimationFrame", {
      configurable: true,
      writable: true,
      value: (callback: FrameRequestCallback) => {
        const id = rafState.nextId;
        rafState.nextId += 1;
        rafState.callbacks.set(id, callback);
        return id;
      },
    });

    Object.defineProperty(window, "cancelAnimationFrame", {
      configurable: true,
      writable: true,
      value: (id: number) => {
        rafState.callbacks.delete(id);
      },
    });

    Object.defineProperty(performance, "now", {
      configurable: true,
      value: () => clock.now,
    });
  });

  beforeEach(() => {
    mocks.state.scenario = "join";
    mocks.buildProximityLinesMock.mockClear();
    mocks.buildLineFragmentShaderSourceMock.mockClear();
    mocks.createProgramMock.mockClear();
    rafState.callbacks.clear();
    rafState.nextId = 1;
    clock.now = 0;
    supportsStandardDerivatives = false;

    activeGl = createFakeWebGLContext();
    Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
      configurable: true,
      value: (contextType: string) => (contextType === "webgl" ? activeGl : null),
    });
  });

  afterEach(() => {
    cleanup();
  });

  it("uses the render variant line shader and applies line blend state separately", () => {
    const applyLineBlendMode = vi.fn();
    const applyPointBlendMode = vi.fn();
    const renderVariant: RenderVariant = {
      ...RENDER_VARIANTS.dark,
      lineFragmentShader: "custom-line-fragment-shader",
      pointFragmentShader: "custom-point-fragment-shader",
      applyLineBlendMode,
      applyPointBlendMode,
      buildLines: () => ({
        positions: new Float32Array([0, 0, 80, 80]),
        colors: new Float32Array([0.4, 0.6, 0.9, 0.28, 0.4, 0.6, 0.9, 0.12]),
      }),
    };

    render(
      <Harness
        animationRef={createMutableRef<Map<string, EntryAnimation>>(new Map())}
        canvasRef={createMutableRef<HTMLCanvasElement | null>(null)}
        exitAnimationRef={createMutableRef<Map<string, ExitAnimation>>(new Map())}
        hostRef={createMutableRef<HTMLDivElement | null>(null)}
        hoveredNodeIdRef={createMutableRef<string | null>(null)}
        lastScreenPositionsRef={createMutableRef<Map<string, { x: number; y: number }>>(new Map())}
        lineTransitionRef={createMutableRef<LineTransition | null>(null)}
        lineRevealRef={createMutableRef<Map<string, LineRevealAnimation>>(new Map())}
        panRef={createMutableRef({ x: 0, y: 0 })}
        pendingLineTransitionRef={createMutableRef<PendingLineTransition | null>(null)}
        renderNodes={[createRenderNode("peer", { x: 0.2, y: 0.3 })]}
        renderVariant={renderVariant}
       
        screenNodesRef={createMutableRef<ScreenNode[]>([])}
        seenNodeIdsRef={createMutableRef<Set<string>>(new Set(["peer"]))}
        selfNodeId="peer"
        twinkleAnimationRef={createMutableRef<Map<string, UpdateTwinkle>>(new Map())}
        zoomRef={createMutableRef(1)}
      />,
    );

    expect(mocks.createProgramMock).toHaveBeenNthCalledWith(
      1,
      expect.anything(),
      "point-vertex-shader",
      "custom-point-fragment-shader",
    );
    expect(mocks.buildLineFragmentShaderSourceMock).toHaveBeenCalledWith(
      "custom-line-fragment-shader",
      { useStandardDerivatives: false },
    );
    expect(mocks.createProgramMock).toHaveBeenNthCalledWith(
      2,
      expect.anything(),
      "line-vertex-shader",
      "custom-line-fragment-shader",
    );
    const expectedMesh = buildLineMesh({
      positions: new Float32Array([0, 0, 80, 80]),
      colors: new Float32Array([0.4, 0.6, 0.9, 0.28, 0.4, 0.6, 0.9, 0.12]),
      lineWidthPx: renderVariant.lineWidthPx,
      devicePixelRatio: 1,
    });
    expect(applyLineBlendMode).toHaveBeenCalledWith(activeGl);
    expect(applyPointBlendMode).toHaveBeenCalledWith(activeGl);
    expect(activeGl.drawArrays).toHaveBeenNthCalledWith(
      1,
      activeGl.TRIANGLES,
      0,
      expectedMesh.positions.length / 2,
    );
    expect(activeGl.drawArrays).toHaveBeenNthCalledWith(2, activeGl.POINTS, 0, 1);
    expect(activeGl.lineWidth).not.toHaveBeenCalled();
    expect(applyLineBlendMode.mock.invocationCallOrder[0]).toBeLessThan(
      activeGl.drawArrays.mock.invocationCallOrder[0],
    );
    expect(applyPointBlendMode.mock.invocationCallOrder[0]).toBeGreaterThan(
      applyLineBlendMode.mock.invocationCallOrder[0],
    );
    expect(applyPointBlendMode.mock.invocationCallOrder[0]).toBeLessThan(
      activeGl.drawArrays.mock.invocationCallOrder[1],
    );
  });

  it("enables derivative-aware line shader source when the WebGL extension is available", () => {
    supportsStandardDerivatives = true;
    activeGl = createFakeWebGLContext();
    Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
      configurable: true,
      value: (contextType: string) => (contextType === "webgl" ? activeGl : null),
    });

    render(
      <Harness
        animationRef={createMutableRef<Map<string, EntryAnimation>>(new Map())}
        canvasRef={createMutableRef<HTMLCanvasElement | null>(null)}
        exitAnimationRef={createMutableRef<Map<string, ExitAnimation>>(new Map())}
        hostRef={createMutableRef<HTMLDivElement | null>(null)}
        hoveredNodeIdRef={createMutableRef<string | null>(null)}
        lastScreenPositionsRef={createMutableRef<Map<string, { x: number; y: number }>>(new Map())}
        lineTransitionRef={createMutableRef<LineTransition | null>(null)}
        lineRevealRef={createMutableRef<Map<string, LineRevealAnimation>>(new Map())}
        panRef={createMutableRef({ x: 0, y: 0 })}
        pendingLineTransitionRef={createMutableRef<PendingLineTransition | null>(null)}
        renderNodes={[createRenderNode("peer", { x: 0.2, y: 0.3 })]}
       
        screenNodesRef={createMutableRef<ScreenNode[]>([])}
        seenNodeIdsRef={createMutableRef<Set<string>>(new Set(["peer"]))}
        selfNodeId="peer"
        twinkleAnimationRef={createMutableRef<Map<string, UpdateTwinkle>>(new Map())}
        zoomRef={createMutableRef(1)}
      />,
    );

    expect(mocks.buildLineFragmentShaderSourceMock).toHaveBeenCalledWith(
      RENDER_VARIANTS.light.lineFragmentShader,
      { useStandardDerivatives: true },
    );
    expect(mocks.createProgramMock).toHaveBeenNthCalledWith(
      2,
      expect.anything(),
      "line-vertex-shader",
      `derivatives:${RENDER_VARIANTS.light.lineFragmentShader}`,
    );
  });

  it("keeps light line rendering unchanged when hover is active", () => {
    render(
      <Harness
        animationRef={createMutableRef<Map<string, EntryAnimation>>(new Map())}
        canvasRef={createMutableRef<HTMLCanvasElement | null>(null)}
        exitAnimationRef={createMutableRef<Map<string, ExitAnimation>>(new Map())}
        hostRef={createMutableRef<HTMLDivElement | null>(null)}
        hoveredNodeIdRef={createMutableRef<string | null>("peer")}
        lastScreenPositionsRef={createMutableRef<Map<string, { x: number; y: number }>>(new Map())}
        lineTransitionRef={createMutableRef<LineTransition | null>(null)}
        lineRevealRef={createMutableRef<Map<string, LineRevealAnimation>>(new Map())}
        panRef={createMutableRef({ x: 0, y: 0 })}
        pendingLineTransitionRef={createMutableRef<PendingLineTransition | null>(null)}
        renderNodes={[
          createRenderNode("anchor", { x: 0.2, y: 0.2 }),
          createRenderNode("peer", { x: 0.8, y: 0.2 }),
          createRenderNode("stable-a", { x: 0.2, y: 0.8 }),
          createRenderNode("stable-b", { x: 0.8, y: 0.8 }),
        ]}
       
        screenNodesRef={createMutableRef<ScreenNode[]>([])}
        seenNodeIdsRef={createMutableRef<Set<string>>(new Set(["anchor", "peer", "stable-a", "stable-b"]))}
        selfNodeId="peer"
        twinkleAnimationRef={createMutableRef<Map<string, UpdateTwinkle>>(new Map())}
        zoomRef={createMutableRef(1)}
      />,
    );

    expect(mocks.buildProximityLinesMock).toHaveBeenCalledTimes(1);
    const baseCall = mocks.buildProximityLinesMock.mock.calls[0]?.[0];

    expect([...baseCall.highlightedNodeIds]).toEqual([]);
    expect(baseCall.visiblePairKeys).toBeUndefined();
    expect(new Set(baseCall.screenNodes.map((node: { id: string }) => node.id))).toEqual(
      new Set(["anchor", "peer", "stable-a", "stable-b"]),
    );
    expect(activeGl.drawArrays).toHaveBeenCalledTimes(1);
    expect(activeGl.drawArrays).toHaveBeenCalledWith(activeGl.POINTS, 0, 4);
  });

  it("does not change line-builder screen-node sizes when hover changes in the normal light path", () => {
    const hoveredNodeIdRef = createMutableRef<string | null>(null);
    const renderNodes = [
      createRenderNode("anchor", { x: 0.2, y: 0.2, size: 16 }),
      createRenderNode("peer", { x: 0.8, y: 0.2, size: 14 }),
      createRenderNode("stable-a", { x: 0.2, y: 0.8, size: 10 }),
      createRenderNode("stable-b", { x: 0.8, y: 0.8, size: 11 }),
    ];
    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef,
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef: createMutableRef<LineTransition | null>(null),
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef: createMutableRef<PendingLineTransition | null>(null),
      screenNodesRef: createMutableRef<ScreenNode[]>([]),
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(renderNodes.map((node) => node.id))),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1),
    };

    const { rerender } = render(
      <Harness {...refs} renderNodes={renderNodes} selfNodeId="peer" />,
    );

    const baseCall = mocks.buildProximityLinesMock.mock.calls[0]?.[0];
    const baseIds = baseCall.screenNodes.map((node: { id: string }) => node.id);
    const baseSizes = new Map(
      baseCall.screenNodes.map((node: { id: string; size: number }) => [node.id, node.size]),
    );
    expect(baseIds).toEqual(["anchor", "peer", "stable-b", "stable-a"]);

    mocks.buildProximityLinesMock.mockClear();
    hoveredNodeIdRef.current = "peer";
    rerender(
      <Harness
        {...refs}
        renderNodes={renderNodes.map((node) => ({ ...node }))}
        
        selfNodeId="peer"
      />,
    );
    flushAnimationFrame();

    expect(mocks.buildProximityLinesMock).not.toHaveBeenCalled();
    expect(refs.screenNodesRef.current.map((node) => node.id)).toEqual([
      "peer",
      "anchor",
      "stable-b",
      "stable-a",
    ]);
    const hoveredScreenNode = refs.screenNodesRef.current.find((node) => node.id === "peer");
    expect(hoveredScreenNode?.hitSize).toBeGreaterThan(hoveredScreenNode?.size ?? 0);
    const stableLineNodes = [...refs.screenNodesRef.current]
      .sort((left, right) => left.id.localeCompare(right.id))
      .map((node) => node.id);
    expect(stableLineNodes.slice().sort()).toEqual(baseIds.slice().sort());
    expect(
      new Map(refs.screenNodesRef.current.map((node) => [node.id, node.size])),
    ).toEqual(baseSizes);
  });

  it("keeps snapshot lines stable during the join pre-reveal branch", () => {
    mocks.state.scenario = "join";

    const previousScreenNodes = [
      createScreenNode(createRenderNode("anchor", { x: 0.2, y: 0.2 })),
      createScreenNode(createRenderNode("peer", { x: 0.8, y: 0.2 })),
      createScreenNode(createRenderNode("stable-a", { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode("stable-b", { x: 0.8, y: 0.8 })),
    ];
    const currentRenderNodes = [
      createRenderNode("anchor", { x: 0.2, y: 0.2 }),
      createRenderNode("peer", { x: 0.8, y: 0.2 }),
      createRenderNode("stable-a", { x: 0.2, y: 0.8 }),
      createRenderNode("stable-b", { x: 0.8, y: 0.8 }),
      createRenderNode("new-node", { x: 0.5, y: 0.5 }),
    ];

    const screenNodesRef = createMutableRef(previousScreenNodes);
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(["new-node"]),
      removedNodeIds: new Set(),
    });
    const lineTransitionRef = createMutableRef<LineTransition | null>(null);

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef: createMutableRef<string | null>(null),
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef,
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef,
      screenNodesRef,
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(["anchor", "peer", "stable-a", "stable-b"])),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1),
    };

    const { rerender } = render(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes}
       
        selfNodeId="peer"
      />,
    );

    clock.now = LINE_REVEAL_DURATION_MS + 1;
    rerender(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes.map((node) => ({ ...node }))}
       
        selfNodeId="peer"
      />,
    );

    const finalCall =
      mocks.buildProximityLinesMock.mock.calls[
        mocks.buildProximityLinesMock.mock.calls.length - 1
      ]?.[0];
    expect(finalCall.screenNodes.map((node: { id: string }) => node.id)).toEqual([
      "anchor",
      "peer",
      "stable-a",
      "stable-b",
    ]);
    expect(finalCall.visiblePairKeys).toBeUndefined();
    const pairAlphaOverrides = finalCall.pairAlphaOverrides;
    expect(pairAlphaOverrides).toBeInstanceOf(Map);
    if (!pairAlphaOverrides) {
      throw new Error("Expected pairAlphaOverrides during the join pre-reveal branch");
    }
    expect([...pairAlphaOverrides.keys()]).toEqual(["anchor::peer"]);
  });

  it("does not build a hover-specific light line pass during the join pre-reveal branch", () => {
    mocks.state.scenario = "join";

    const previousScreenNodes = [
      createScreenNode(createRenderNode("anchor", { x: 0.2, y: 0.2 })),
      createScreenNode(createRenderNode("peer", { x: 0.8, y: 0.2 })),
      createScreenNode(createRenderNode("stable-a", { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode("stable-b", { x: 0.8, y: 0.8 })),
    ];
    const currentRenderNodes = [
      createRenderNode("anchor", { x: 0.2, y: 0.2 }),
      createRenderNode("peer", { x: 0.8, y: 0.2 }),
      createRenderNode("stable-a", { x: 0.2, y: 0.8 }),
      createRenderNode("stable-b", { x: 0.8, y: 0.8 }),
      createRenderNode("new-node", { x: 0.5, y: 0.5 }),
    ];

    const hoveredNodeIdRef = createMutableRef<string | null>(null);
    const screenNodesRef = createMutableRef(previousScreenNodes);
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(["new-node"]),
      removedNodeIds: new Set(),
    });
    const lineTransitionRef = createMutableRef<LineTransition | null>(null);

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef,
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef,
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef,
      screenNodesRef,
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(["anchor", "peer", "stable-a", "stable-b"])),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1),
    };

    const { rerender } = render(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes}
       
        selfNodeId="peer"
      />,
    );

    mocks.buildProximityLinesMock.mockClear();
    hoveredNodeIdRef.current = "peer";
    clock.now = LINE_REVEAL_DURATION_MS + 1;
    rerender(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes.map((node) => ({ ...node }))}
        
        selfNodeId="peer"
      />,
    );
    flushAnimationFrame();

    const calls = mocks.buildProximityLinesMock.mock.calls.map((call) => call[0]);
    const finalCall = calls[calls.length - 1];

    expect(calls).toHaveLength(2);
    expect(calls.map((call) => [...call.highlightedNodeIds])).toEqual([[], []]);
    expect(calls.every((call) => call.visiblePairKeys === undefined)).toBe(true);
    expect(finalCall.pairAlphaOverrides).toBeInstanceOf(Map);
    if (!finalCall.pairAlphaOverrides) {
      throw new Error("Expected pairAlphaOverrides during the hover join pre-reveal branch");
    }
    expect([...finalCall.pairAlphaOverrides.keys()]).toEqual(["anchor::peer"]);
    expect(new Set(finalCall.screenNodes.map((node: { id: string }) => node.id))).toEqual(
      new Set(["anchor", "peer", "stable-a", "stable-b"]),
    );
  });

  it("only fades the changed outgoing edge during the join pre-entry snapshot branch", () => {
    mocks.state.scenario = "join";

    const previousScreenNodes = [
      createScreenNode(createRenderNode("anchor", { x: 0.2, y: 0.2 })),
      createScreenNode(createRenderNode("peer", { x: 0.8, y: 0.2 })),
      createScreenNode(createRenderNode("stable-a", { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode("stable-b", { x: 0.8, y: 0.8 })),
    ];
    const currentRenderNodes = [
      createRenderNode("anchor", { x: 0.2, y: 0.2 }),
      createRenderNode("peer", { x: 0.8, y: 0.2 }),
      createRenderNode("stable-a", { x: 0.2, y: 0.8 }),
      createRenderNode("stable-b", { x: 0.8, y: 0.8 }),
      createRenderNode("new-node", { x: 0.5, y: 0.5 }),
    ];

    const screenNodesRef = createMutableRef(previousScreenNodes);
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(["new-node"]),
      removedNodeIds: new Set(),
    });
    const lineTransitionRef = createMutableRef<LineTransition | null>(null);

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef: createMutableRef<string | null>(null),
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef,
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef,
      screenNodesRef,
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(["anchor", "peer", "stable-a", "stable-b"])),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1),
    };

    const { rerender } = render(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes}
       
        selfNodeId="peer"
      />,
    );

    mocks.buildProximityLinesMock.mockClear();
    clock.now = LINE_REVEAL_DURATION_MS - 1;
    rerender(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes.map((node) => ({ ...node }))}
        
        selfNodeId="peer"
      />,
    );
    flushAnimationFrame();

    const finalCall =
      mocks.buildProximityLinesMock.mock.calls[
        mocks.buildProximityLinesMock.mock.calls.length - 1
      ]?.[0];
    expect(finalCall.screenNodes.map((node: { id: string }) => node.id)).toEqual([
      "anchor",
      "peer",
      "stable-a",
      "stable-b",
    ]);
    expect(finalCall.visiblePairKeys).toBeUndefined();

    const pairAlphaOverrides = finalCall.pairAlphaOverrides;
    expect(pairAlphaOverrides).toBeInstanceOf(Map);
    if (!pairAlphaOverrides) {
      throw new Error("Expected pairAlphaOverrides during the join pre-entry snapshot branch");
    }
    expect([...pairAlphaOverrides.keys()]).toEqual(["anchor::peer"]);
    expect(pairAlphaOverrides.get("anchor::peer")).toBeGreaterThan(0);
    expect(pairAlphaOverrides.get("anchor::peer")).toBeLessThan(1);
    expect(pairAlphaOverrides.has("stable-a::stable-b")).toBe(false);
  });

  it("keeps transition creation local when join churn spills through existing hubs", () => {
    mocks.state.scenario = "join-spillover";

    const previousScreenNodes = [
      createScreenNode(createRenderNode("anchor", { x: 0.15, y: 0.2 })),
      createScreenNode(createRenderNode("peer", { x: 0.4, y: 0.35 })),
      createScreenNode(createRenderNode("worker", { x: 0.65, y: 0.45 })),
      createScreenNode(createRenderNode("remote", { x: 0.8, y: 0.65 })),
      createScreenNode(createRenderNode("stable-a", { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode("stable-b", { x: 0.85, y: 0.82 })),
    ];
    const currentRenderNodes = [
      createRenderNode("anchor", { x: 0.15, y: 0.2 }),
      createRenderNode("peer", { x: 0.4, y: 0.35 }),
      createRenderNode("worker", { x: 0.65, y: 0.45 }),
      createRenderNode("remote", { x: 0.8, y: 0.65 }),
      createRenderNode("stable-a", { x: 0.2, y: 0.8 }),
      createRenderNode("stable-b", { x: 0.85, y: 0.82 }),
      createRenderNode("new-node", { x: 0.28, y: 0.3 }),
    ];

    const screenNodesRef = createMutableRef(previousScreenNodes);
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(["new-node"]),
      removedNodeIds: new Set(),
    });
    const lineTransitionRef = createMutableRef<LineTransition | null>(null);

    render(
      <Harness
        animationRef={createMutableRef<Map<string, EntryAnimation>>(new Map())}
        canvasRef={createMutableRef<HTMLCanvasElement | null>(null)}
        exitAnimationRef={createMutableRef<Map<string, ExitAnimation>>(new Map())}
        hostRef={createMutableRef<HTMLDivElement | null>(null)}
        hoveredNodeIdRef={createMutableRef<string | null>(null)}
        lastScreenPositionsRef={createMutableRef<Map<string, { x: number; y: number }>>(new Map())}
        lineTransitionRef={lineTransitionRef}
        lineRevealRef={createMutableRef<Map<string, LineRevealAnimation>>(new Map())}
        panRef={createMutableRef({ x: 0, y: 0 })}
        pendingLineTransitionRef={pendingLineTransitionRef}
        renderNodes={currentRenderNodes}
       
        screenNodesRef={screenNodesRef}
        seenNodeIdsRef={createMutableRef<Set<string>>(
          new Set(["anchor", "peer", "worker", "remote", "stable-a", "stable-b"]),
        )}
        selfNodeId="peer"
        twinkleAnimationRef={createMutableRef<Map<string, UpdateTwinkle>>(new Map())}
        zoomRef={createMutableRef(1)}
      />,
    );

    expect(lineTransitionRef.current?.outgoingPairKeys).toEqual(new Set(["anchor::peer"]));
    expect(lineTransitionRef.current?.incomingPairKeys).toEqual(
      new Set(["anchor::new-node", "new-node::peer"]),
    );
    expect(lineTransitionRef.current?.stableVisiblePairKeys).toEqual(
      new Set([
        "peer::worker",
        "worker::remote",
        "peer::worker-2",
        "worker::remote-2",
        "stable-a::stable-b",
      ]),
    );
  });

  it("treats a same-pair reroute around a new blocker as an outgoing and incoming light-edge transition", () => {
    mocks.state.scenario = "join-route-change";

    const previousScreenNodes = [
      createScreenNode(createRenderNode("anchor", { x: 0.15, y: 0.2 })),
      createScreenNode(createRenderNode("peer", { x: 0.45, y: 0.35 })),
      createScreenNode(createRenderNode("stable-a", { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode("stable-b", { x: 0.85, y: 0.82 })),
    ];
    const currentRenderNodes = [
      createRenderNode("anchor", { x: 0.15, y: 0.2 }),
      createRenderNode("peer", { x: 0.45, y: 0.35 }),
      createRenderNode("stable-a", { x: 0.2, y: 0.8 }),
      createRenderNode("stable-b", { x: 0.85, y: 0.82 }),
      createRenderNode("new-node", { x: 0.28, y: 0.3 }),
    ];

    const screenNodesRef = createMutableRef(previousScreenNodes);
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(["new-node"]),
      removedNodeIds: new Set(),
    });
    const lineTransitionRef = createMutableRef<LineTransition | null>(null);

    render(
      <Harness
        animationRef={createMutableRef<Map<string, EntryAnimation>>(new Map())}
        canvasRef={createMutableRef<HTMLCanvasElement | null>(null)}
        exitAnimationRef={createMutableRef<Map<string, ExitAnimation>>(new Map())}
        hostRef={createMutableRef<HTMLDivElement | null>(null)}
        hoveredNodeIdRef={createMutableRef<string | null>(null)}
        lastScreenPositionsRef={createMutableRef<Map<string, { x: number; y: number }>>(new Map())}
        lineTransitionRef={lineTransitionRef}
        lineRevealRef={createMutableRef<Map<string, LineRevealAnimation>>(new Map())}
        panRef={createMutableRef({ x: 0, y: 0 })}
        pendingLineTransitionRef={pendingLineTransitionRef}
        renderNodes={currentRenderNodes}
       
        screenNodesRef={screenNodesRef}
        seenNodeIdsRef={createMutableRef<Set<string>>(
          new Set(["anchor", "peer", "stable-a", "stable-b"]),
        )}
        selfNodeId="peer"
        twinkleAnimationRef={createMutableRef<Map<string, UpdateTwinkle>>(new Map())}
        zoomRef={createMutableRef(1)}
      />,
    );

    expect(lineTransitionRef.current?.outgoingPairKeys).toEqual(new Set(["anchor::peer"]));
    expect(lineTransitionRef.current?.incomingPairKeys).toEqual(new Set(["anchor::peer"]));
    expect(lineTransitionRef.current?.stableVisiblePairKeys).toEqual(
      new Set(["stable-a::stable-b"]),
    );
  });

  it("keeps remote current-only spillover out of the no-outgoing join pre-reveal branch", () => {
    mocks.state.scenario = "join-spillover-incoming-only";

    const previousScreenNodes = [
      createScreenNode(createRenderNode("anchor", { x: 0.15, y: 0.2 })),
      createScreenNode(createRenderNode("peer", { x: 0.4, y: 0.35 })),
      createScreenNode(createRenderNode("worker", { x: 0.65, y: 0.45 })),
      createScreenNode(createRenderNode("remote", { x: 0.8, y: 0.65 })),
      createScreenNode(createRenderNode("stable-a", { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode("stable-b", { x: 0.85, y: 0.82 })),
    ];
    const currentRenderNodes = [
      createRenderNode("anchor", { x: 0.15, y: 0.2 }),
      createRenderNode("peer", { x: 0.4, y: 0.35 }),
      createRenderNode("worker", { x: 0.65, y: 0.45 }),
      createRenderNode("remote", { x: 0.8, y: 0.65 }),
      createRenderNode("stable-a", { x: 0.2, y: 0.8 }),
      createRenderNode("stable-b", { x: 0.85, y: 0.82 }),
      createRenderNode("new-node", { x: 0.28, y: 0.3 }),
    ];

    const screenNodesRef = createMutableRef(previousScreenNodes);
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(["new-node"]),
      removedNodeIds: new Set(),
    });
    const lineTransitionRef = createMutableRef<LineTransition | null>(null);

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef: createMutableRef<string | null>(null),
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef,
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef,
      screenNodesRef,
      seenNodeIdsRef: createMutableRef<Set<string>>(
        new Set(["anchor", "peer", "worker", "remote", "stable-a", "stable-b"]),
      ),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1),
    };

    const { rerender } = render(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes}
       
        selfNodeId="peer"
      />,
    );

    mocks.buildProximityLinesMock.mockClear();
    clock.now = 30;
    rerender(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes.map((node) => ({ ...node }))}
        
        selfNodeId="peer"
      />,
    );
    flushAnimationFrame();

    const finalCall =
      mocks.buildProximityLinesMock.mock.calls[
        mocks.buildProximityLinesMock.mock.calls.length - 1
      ]?.[0];
    expect(lineTransitionRef.current?.outgoingPairKeys).toEqual(new Set());
    expect(lineTransitionRef.current?.incomingPairKeys).toEqual(
      new Set(["anchor::new-node", "new-node::peer"]),
    );
    expect(finalCall.screenNodes.map((node: { id: string }) => node.id)).toEqual([
      "anchor",
      "peer",
      "worker",
      "remote",
      "stable-a",
      "stable-b",
    ]);
    expect(finalCall.visiblePairKeys).toBeUndefined();
    const pairAlphaOverrides = finalCall.pairAlphaOverrides;
    expect(pairAlphaOverrides).toBeInstanceOf(Map);
    if (!pairAlphaOverrides) {
      throw new Error("Expected pairAlphaOverrides during the no-outgoing join pre-reveal branch");
    }
    expect(pairAlphaOverrides.size).toBe(0);
  });

  it("keeps the previous snapshot during the leave move-settle window before revealing replacement edges", () => {
    mocks.state.scenario = "leave";

    const previousScreenNodes = [
      createScreenNode(createRenderNode("peer", { x: 0.25, y: 0.25 })),
      createScreenNode(createRenderNode("removed-node", { x: 0.8, y: 0.2 })),
      createScreenNode(createRenderNode("stable-a", { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode("stable-b", { x: 0.8, y: 0.8 })),
    ];
    const currentRenderNodes = [
      createRenderNode("peer", { x: 0.25, y: 0.25 }),
      createRenderNode("replacement", { x: 0.82, y: 0.26 }),
      createRenderNode("stable-a", { x: 0.2, y: 0.8 }),
      createRenderNode("stable-b", { x: 0.8, y: 0.8 }),
    ];

    const screenNodesRef = createMutableRef(previousScreenNodes);
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(),
      removedNodeIds: new Set(["removed-node"]),
    });
    const lineTransitionRef = createMutableRef<LineTransition | null>(null);

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef: createMutableRef<string | null>(null),
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef,
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef,
      screenNodesRef,
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(["peer", "removed-node", "stable-a", "stable-b"])),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1),
    };

    const { rerender } = render(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes}
       
        selfNodeId="peer"
      />,
    );

    mocks.buildProximityLinesMock.mockClear();
    clock.now = LINE_REVEAL_DURATION_MS + 1;
    rerender(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes.map((node) => ({ ...node }))}
        
        selfNodeId="peer"
      />,
    );
    flushAnimationFrame();

    const finalCall =
      mocks.buildProximityLinesMock.mock.calls[
        mocks.buildProximityLinesMock.mock.calls.length - 1
      ]?.[0];
    expect(finalCall.screenNodes.map((node: { id: string }) => node.id)).toEqual([
      "peer",
      "removed-node",
      "stable-a",
      "stable-b",
    ]);
    expect(finalCall.visiblePairKeys).toBeUndefined();
    const pairAlphaOverrides = finalCall.pairAlphaOverrides;
    expect(pairAlphaOverrides).toBeInstanceOf(Map);
    if (!pairAlphaOverrides) {
      throw new Error("Expected pairAlphaOverrides during the leave move-settle window");
    }
    expect([...pairAlphaOverrides.keys()]).toEqual(["peer::removed-node"]);
    expect(pairAlphaOverrides.get("peer::removed-node")).toBe(0);
  });

  it("reveals the replacement edge only after the leave move-settle window completes", () => {
    mocks.state.scenario = "leave";

    const previousScreenNodes = [
      createScreenNode(createRenderNode("peer", { x: 0.25, y: 0.25 })),
      createScreenNode(createRenderNode("removed-node", { x: 0.8, y: 0.2 })),
      createScreenNode(createRenderNode("stable-a", { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode("stable-b", { x: 0.8, y: 0.8 })),
    ];
    const currentRenderNodes = [
      createRenderNode("peer", { x: 0.25, y: 0.25 }),
      createRenderNode("replacement", { x: 0.82, y: 0.26 }),
      createRenderNode("stable-a", { x: 0.2, y: 0.8 }),
      createRenderNode("stable-b", { x: 0.8, y: 0.8 }),
    ];

    const screenNodesRef = createMutableRef(previousScreenNodes);
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(),
      removedNodeIds: new Set(["removed-node"]),
    });
    const lineTransitionRef = createMutableRef<LineTransition | null>(null);

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef: createMutableRef<string | null>(null),
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(new Map()),
      lineTransitionRef,
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef,
      screenNodesRef,
      seenNodeIdsRef: createMutableRef<Set<string>>(new Set(["peer", "removed-node", "stable-a", "stable-b"])),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1),
    };

    const { rerender } = render(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes}
       
        selfNodeId="peer"
      />,
    );

    mocks.buildProximityLinesMock.mockClear();
    clock.now = LINE_REVEAL_DURATION_MS + ENTRY_ANIMATION_DURATION_MS + 1;
    rerender(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes.map((node) => ({ ...node }))}
        
        selfNodeId="peer"
      />,
    );
    flushAnimationFrame();

    const finalCall =
      mocks.buildProximityLinesMock.mock.calls[
        mocks.buildProximityLinesMock.mock.calls.length - 1
      ]?.[0];
    expect(finalCall.visiblePairKeys).toEqual(
      new Set(["peer::replacement", "stable-a::stable-b"]),
    );
    const pairAlphaOverrides = finalCall.pairAlphaOverrides;
    expect(pairAlphaOverrides).toBeInstanceOf(Map);
    if (!pairAlphaOverrides) {
      throw new Error("Expected pairAlphaOverrides during the leave reveal branch");
    }
    expect([...pairAlphaOverrides.keys()]).toEqual(["peer::replacement"]);
    expect(pairAlphaOverrides.get("peer::replacement")).toBeGreaterThan(0);
    expect(pairAlphaOverrides.get("peer::replacement")).toBeLessThanOrEqual(1);
  });

  it("keeps moved existing nodes line-hidden until they settle during the light leave reveal path", () => {
    mocks.state.scenario = "leave";

    const previousScreenNodes = [
      createScreenNode(createRenderNode("peer", { x: 0.25, y: 0.25 })),
      createScreenNode(createRenderNode("removed-node", { x: 0.8, y: 0.2 })),
      createScreenNode(createRenderNode("stable-a", { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode("stable-b", { x: 0.8, y: 0.8 })),
    ];
    const currentRenderNodes = [
      createRenderNode("peer", { x: 0.25, y: 0.25 }),
      createRenderNode("replacement", { x: 0.82, y: 0.26 }),
      createRenderNode("stable-a", { x: 0.52, y: 0.62 }),
      createRenderNode("stable-b", { x: 0.8, y: 0.8 }),
    ];

    const refs = {
      animationRef: createMutableRef<Map<string, EntryAnimation>>(new Map()),
      canvasRef: createMutableRef<HTMLCanvasElement | null>(null),
      exitAnimationRef: createMutableRef<Map<string, ExitAnimation>>(new Map()),
      hostRef: createMutableRef<HTMLDivElement | null>(null),
      hoveredNodeIdRef: createMutableRef<string | null>(null),
      lastScreenPositionsRef: createMutableRef<Map<string, { x: number; y: number }>>(
        new Map([
          ["peer", { x: 0.25 * 800, y: 0.25 * 600 }],
          ["stable-a", { x: 0.2 * 800, y: 0.8 * 600 }],
          ["stable-b", { x: 0.8 * 800, y: 0.8 * 600 }],
        ]),
      ),
      lineTransitionRef: createMutableRef<LineTransition | null>(null),
      lineRevealRef: createMutableRef<Map<string, LineRevealAnimation>>(new Map()),
      panRef: createMutableRef({ x: 0, y: 0 }),
      pendingLineTransitionRef: createMutableRef<PendingLineTransition | null>({
        addedNodeIds: new Set(),
        removedNodeIds: new Set(["removed-node"]),
      }),
      screenNodesRef: createMutableRef(previousScreenNodes),
      seenNodeIdsRef: createMutableRef<Set<string>>(
        new Set(["peer", "removed-node", "stable-a", "stable-b"]),
      ),
      twinkleAnimationRef: createMutableRef<Map<string, UpdateTwinkle>>(new Map()),
      zoomRef: createMutableRef(1),
    };

    const { rerender } = render(
      <Harness {...refs} renderNodes={currentRenderNodes} selfNodeId="peer" />,
    );

    mocks.buildProximityLinesMock.mockClear();
    clock.now = LINE_REVEAL_DURATION_MS + ENTRY_ANIMATION_DURATION_MS + 1;
    rerender(
      <Harness
        {...refs}
        renderNodes={currentRenderNodes.map((node) => ({ ...node }))}
        
        selfNodeId="peer"
      />,
    );
    flushAnimationFrame();

    const finalCall =
      mocks.buildProximityLinesMock.mock.calls[
        mocks.buildProximityLinesMock.mock.calls.length - 1
      ]?.[0];
    const movedNode = finalCall.screenNodes.find((node: { id: string }) => node.id === "stable-a");

    expect(finalCall.visiblePairKeys).toEqual(new Set(["peer::replacement"]));
    expect(movedNode?.lineRevealProgress).toBe(0);
  });

  it("keeps removal transition creation local when current churn spills through replacement edges", () => {
    mocks.state.scenario = "leave-spillover";

    const previousScreenNodes = [
      createScreenNode(createRenderNode("peer", { x: 0.25, y: 0.25 })),
      createScreenNode(createRenderNode("removed-node", { x: 0.52, y: 0.22 })),
      createScreenNode(createRenderNode("stable-a", { x: 0.2, y: 0.8 })),
      createScreenNode(createRenderNode("stable-b", { x: 0.8, y: 0.8 })),
    ];
    const currentRenderNodes = [
      createRenderNode("peer", { x: 0.25, y: 0.25 }),
      createRenderNode("replacement", { x: 0.54, y: 0.24 }),
      createRenderNode("worker-2", { x: 0.68, y: 0.44 }),
      createRenderNode("remote-2", { x: 0.82, y: 0.62 }),
      createRenderNode("stable-a", { x: 0.2, y: 0.8 }),
      createRenderNode("stable-b", { x: 0.8, y: 0.8 }),
    ];

    const screenNodesRef = createMutableRef(previousScreenNodes);
    const pendingLineTransitionRef = createMutableRef<PendingLineTransition | null>({
      addedNodeIds: new Set(),
      removedNodeIds: new Set(["removed-node"]),
    });
    const lineTransitionRef = createMutableRef<LineTransition | null>(null);

    render(
      <Harness
        animationRef={createMutableRef<Map<string, EntryAnimation>>(new Map())}
        canvasRef={createMutableRef<HTMLCanvasElement | null>(null)}
        exitAnimationRef={createMutableRef<Map<string, ExitAnimation>>(new Map())}
        hostRef={createMutableRef<HTMLDivElement | null>(null)}
        hoveredNodeIdRef={createMutableRef<string | null>(null)}
        lastScreenPositionsRef={createMutableRef<Map<string, { x: number; y: number }>>(new Map())}
        lineTransitionRef={lineTransitionRef}
        lineRevealRef={createMutableRef<Map<string, LineRevealAnimation>>(new Map())}
        panRef={createMutableRef({ x: 0, y: 0 })}
        pendingLineTransitionRef={pendingLineTransitionRef}
        renderNodes={currentRenderNodes}
       
        screenNodesRef={screenNodesRef}
        seenNodeIdsRef={createMutableRef<Set<string>>(
          new Set(["peer", "removed-node", "stable-a", "stable-b"]),
        )}
        selfNodeId="peer"
        twinkleAnimationRef={createMutableRef<Map<string, UpdateTwinkle>>(new Map())}
        zoomRef={createMutableRef(1)}
      />,
    );

    expect(lineTransitionRef.current?.outgoingPairKeys).toEqual(new Set(["peer::removed-node"]));
    expect(lineTransitionRef.current?.incomingPairKeys).toEqual(new Set(["peer::replacement"]));
    expect(lineTransitionRef.current?.stableVisiblePairKeys).toEqual(
      new Set(["replacement::worker-2", "worker-2::remote-2", "stable-a::stable-b"]),
    );
  });
});
