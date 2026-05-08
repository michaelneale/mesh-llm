import { useEffect, useRef, type MutableRefObject } from "react";

import {
  clamp,
  ENTRY_ANIMATION_DURATION_MS,
  hashString,
  LINE_REVEAL_DURATION_MS,
  TAU,
} from "../helpers";
import { buildProximityLines } from "./line-builders";
import { buildLineMesh, type LineMesh } from "./line-mesh";
import { narrowPairDiffToChangedComponent } from "./line-transition";
import { createTravelAnimation, cubicBezier, easeOutCubic } from "./animations";
import {
  buildLineFragmentShaderSource,
  createProgram,
  LINE_VERTEX_SHADER,
  POINT_VERTEX_SHADER,
} from "./shaders";
import type {
  EntryAnimation,
  ExitAnimation,
  LineBuilderOutput,
  LineTransition,
  LineRevealAnimation,
  PendingLineTransition,
  RenderNode,
  RenderVariant,
  ScreenNode,
  UpdateTwinkle,
} from "../types";

type UseTopologyCanvasArgs = {
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
  renderVariant: RenderVariant;
  selfNodeId?: string;
  hopDelayMs?: number;
};

const STAGGER_THRESHOLD = 4;

function computePairHopDelays(
  pairKeys: string[],
  centerNodeId: string,
  delayPerHopMs: number,
): { pairHopDelays: Map<string, number>; maxHopDelay: number } {
  const adjacency = new Map<string, Set<string>>();
  const addEdge = (a: string, b: string) => {
    let set = adjacency.get(a);
    if (!set) { set = new Set(); adjacency.set(a, set); }
    set.add(b);
  };
  for (const pk of pairKeys) {
    const sep = pk.indexOf("::");
    if (sep < 0) continue;
    const left = pk.slice(0, sep);
    const right = pk.slice(sep + 2);
    addEdge(left, right);
    addEdge(right, left);
  }

  const depth = new Map<string, number>();
  depth.set(centerNodeId, 0);
  const queue: string[] = [centerNodeId];
  let head = 0;
  while (head < queue.length) {
    const node = queue[head++];
    const d = depth.get(node)!;
    for (const neighbor of adjacency.get(node) ?? []) {
      if (!depth.has(neighbor)) {
        depth.set(neighbor, d + 1);
        queue.push(neighbor);
      }
    }
  }

  const pairHopDelays = new Map<string, number>();
  let maxHopDelay = 0;
  for (const pk of pairKeys) {
    const sep = pk.indexOf("::");
    if (sep < 0) continue;
    const dLeft = depth.get(pk.slice(0, sep)) ?? 0;
    const dRight = depth.get(pk.slice(sep + 2)) ?? 0;
    const wave = Math.min(dLeft, dRight);
    const delay = wave * delayPerHopMs;
    pairHopDelays.set(pk, delay);
    if (delay > maxHopDelay) maxHopDelay = delay;
  }
  return { pairHopDelays, maxHopDelay };
}

export function useTopologyCanvas({
  canvasRef,
  hostRef,
  screenNodesRef,
  animationRef,
  lineRevealRef,
  exitAnimationRef,
  twinkleAnimationRef,
  pendingLineTransitionRef,
  lineTransitionRef,
  lastScreenPositionsRef,
  seenNodeIdsRef,
  hoveredNodeIdRef,
  zoomRef,
  panRef,
  renderNodes,
    renderVariant,
    selfNodeId,
    hopDelayMs,
}: UseTopologyCanvasArgs) {
  const lineScreenNodesRef = useRef<ScreenNode[]>([]);
  const renderNodesRef = useRef(renderNodes);
  renderNodesRef.current = renderNodes;

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = hostRef.current;
    if (!canvas || !host) return;
    if (typeof WebGLRenderingContext === "undefined") return;

    const gl = canvas.getContext("webgl", {
      alpha: true,
      antialias: false,
      premultipliedAlpha: true,
    });
    if (!gl) return;

    const supportsLineDerivatives = Boolean(gl.getExtension("OES_standard_derivatives"));
    const lineFragmentShader = buildLineFragmentShaderSource(renderVariant.lineFragmentShader, {
      useStandardDerivatives: supportsLineDerivatives,
    });

    const pointProgram = createProgram(gl, POINT_VERTEX_SHADER, renderVariant.pointFragmentShader);
    const lineProgram = createProgram(gl, LINE_VERTEX_SHADER, lineFragmentShader);
    if (!pointProgram || !lineProgram) return;

    const pointPositionLocation = gl.getAttribLocation(pointProgram, "a_position");
    const pointSizeLocation = gl.getAttribLocation(pointProgram, "a_size");
    const pointColorLocation = gl.getAttribLocation(pointProgram, "a_color");
    const pointPulseLocation = gl.getAttribLocation(pointProgram, "a_pulse");
    const pointTwinkleLocation = gl.getAttribLocation(pointProgram, "a_twinkle");
    const pointResolutionLocation = gl.getUniformLocation(pointProgram, "u_resolution");
    const pointTimeLocation = gl.getUniformLocation(pointProgram, "u_time");

    const linePositionLocation = gl.getAttribLocation(lineProgram, "a_position");
    const lineColorLocation = gl.getAttribLocation(lineProgram, "a_color");
    const lineCoordLocation = gl.getAttribLocation(lineProgram, "a_lineCoord");
    const lineResolutionLocation = gl.getUniformLocation(lineProgram, "u_resolution");

    const pointPositionBuffer = gl.createBuffer();
    const pointSizeBuffer = gl.createBuffer();
    const pointColorBuffer = gl.createBuffer();
    const pointPulseBuffer = gl.createBuffer();
    const pointTwinkleBuffer = gl.createBuffer();
    const linePositionBuffer = gl.createBuffer();
    const lineColorBuffer = gl.createBuffer();
    const lineCoordBuffer = gl.createBuffer();
    if (
      !pointPositionBuffer ||
      !pointSizeBuffer ||
      !pointColorBuffer ||
      !pointPulseBuffer ||
      !pointTwinkleBuffer ||
      !linePositionBuffer ||
      !lineColorBuffer ||
      !lineCoordBuffer
    ) {
      return;
    }

    const lineAttributeLocations = [linePositionLocation, lineColorLocation, lineCoordLocation];
    const pointAttributeLocations = [
      pointPositionLocation,
      pointSizeLocation,
      pointColorLocation,
      pointPulseLocation,
      pointTwinkleLocation,
    ];
    const disableVertexAttributes = (locations: number[]) => {
      for (const location of locations) {
        if (location >= 0) {
          gl.disableVertexAttribArray(location);
        }
      }
    };

    let frame = 0;
    let animationFrame = 0;
    let width = 0;
    let height = 0;
    let cssWidth = 0;
    let cssHeight = 0;
    let devicePixelRatio = 1;
    let _resizeDirty = false;
    const sortStableLineNodes = (nodes: ScreenNode[]) =>
      [...nodes].sort((left, right) => right.z - left.z || right.size - left.size);

    const MAX_CANVAS_PIXELS = 8_000_000;
    const resize = () => {
      const rect = host.getBoundingClientRect();
      cssWidth = Math.max(1, rect.width);
      cssHeight = Math.max(1, rect.height);
      // Cap DPR so physical pixel count stays within budget — prevents
      // fill-rate bottleneck on large/fullscreen canvases.
      const rawDpr = window.devicePixelRatio || 1;
      const maxDpr = Math.sqrt(MAX_CANVAS_PIXELS / (cssWidth * cssHeight));
      devicePixelRatio = Math.min(rawDpr, Math.max(1, maxDpr));
      width = Math.max(1, Math.round(cssWidth * devicePixelRatio));
      height = Math.max(1, Math.round(cssHeight * devicePixelRatio));
      canvas.width = width;
      canvas.height = height;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      gl.viewport(0, 0, width, height);
      _resizeDirty = true;
    };

    // --- Preallocated scratch arrays and typed-array buffers (persist across frames) ---
    // V8 retains the backing store when .length is set to 0
    const _scratchPointPos: number[] = [];
    const _scratchPointSize: number[] = [];
    const _scratchPointColor: number[] = [];
    const _scratchPointPulse: number[] = [];
    const _scratchPointTwinkle: number[] = [];

    // Grow-only Float32Array output buffers (doubled on overflow, never shrink)
    let _pointPosBuf = new Float32Array(128);
    let _pointSizeBuf = new Float32Array(64);
    let _pointColorBuf = new Float32Array(256);
    let _pointPulseBuf = new Float32Array(64);
    let _pointTwinkleBuf = new Float32Array(64);

    // Reusable Set objects (.clear() is O(1), keeps hash-table backing)
    const _pointHighlightedSet = new Set<string>();
    const _baseLineHighlightedSet = new Set<string>();
    const _activeIds = new Set<string>();

    let _lineFp = NaN;
    let _lineCache: LineBuilderOutput = {
      positions: new Float32Array(0),
      colors: new Float32Array(0),
    };

    const ensureFloat32 = (
      buf: Float32Array<ArrayBuffer>,
      needed: number,
    ): Float32Array<ArrayBuffer> =>
      buf.length >= needed ? buf : new Float32Array(Math.max(needed, buf.length * 2));

    let _glPointPosCap = 0;
    let _glPointSizeCap = 0;
    let _glPointColorCap = 0;
    let _glPointPulseCap = 0;
    let _glPointTwinkleCap = 0;
    let _glLinePosCap = 0;
    let _glLineColorCap = 0;
    let _glLineCoordCap = 0;

    const uploadToGLBuffer = (
      buffer: WebGLBuffer,
      data: Float32Array,
      prevCap: number,
    ): number => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      if (data.byteLength > prevCap) {
        const cap = Math.max(data.byteLength * 2, 512);
        gl.bufferData(gl.ARRAY_BUFFER, cap, gl.DYNAMIC_DRAW);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
        return cap;
      }
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
      return prevCap;
    };

    let _meshPosRef: Float32Array | null = null;
    let _meshColorRef: Float32Array | null = null;
    let _meshWidthPx = NaN;
    let _meshDpr = NaN;
    let _meshCache: LineMesh = {
      positions: new Float32Array(0),
      colors: new Float32Array(0),
      lineCoords: new Float32Array(0),
    };

    const buildFrameData = () => {
      // Reset scratch arrays (backing store retained by V8)
      _scratchPointPos.length = 0;
      _scratchPointSize.length = 0;
      _scratchPointColor.length = 0;
      _scratchPointPulse.length = 0;
      _scratchPointTwinkle.length = 0;
      const pointPositions = _scratchPointPos;
      const pointSizes = _scratchPointSize;
      const pointColors = _scratchPointColor;
      const pointPulses = _scratchPointPulse;
      const pointTwinkles = _scratchPointTwinkle;
      const screenNodes: ScreenNode[] = [];
      const centerNode =
        renderNodesRef.current.find((node) => node.id === selfNodeId) ?? renderNodesRef.current[renderNodesRef.current.length - 1];
      const now = performance.now();
      const hoveredNodeId = hoveredNodeIdRef.current;
      _pointHighlightedSet.clear();
      if (hoveredNodeId) _pointHighlightedSet.add(hoveredNodeId);
      _baseLineHighlightedSet.clear();
      _activeIds.clear();
      for (const node of renderNodesRef.current) _activeIds.add(node.id);
      const pointHighlightedSet = _pointHighlightedSet;
      const baseLineHighlightedSet = _baseLineHighlightedSet;
      const activeIds = _activeIds;
      // When an active transition is in its pre-reveal phase, the user sees
      // lines rendered from the snapshot — not the current live nodes.
      // Derive the diff from what is actually displayed so that mid-fade
      // pairs are properly tracked instead of snapping instantly.
      const previousScreenNodes = (() => {
        const active = lineTransitionRef.current;
        if (
          active &&
          now < active.revealStartedAt &&
          active.snapshotScreenNodes.length > 0
        ) {
          return active.snapshotScreenNodes;
        }
        return lineScreenNodesRef.current.length > 0
          ? lineScreenNodesRef.current
          : sortStableLineNodes(screenNodesRef.current);
      })();

      const buildPreviewScreenNodes = (nodesToRender: RenderNode[]) => {
        const currentZoom = zoomRef.current;
        const currentPan = panRef.current;
        return nodesToRender.map<ScreenNode>((node) => ({
          ...node,
          px: node.x * cssWidth * currentZoom + currentPan.x,
          py: node.y * cssHeight * currentZoom + currentPan.y,
          hitSize: node.size,
          lineRevealProgress: 1,
        }));
      };

      const pendingTransition = pendingLineTransitionRef.current;
      if (
        pendingTransition &&
        previousScreenNodes.length > 0
      ) {
        const previousPreview = buildProximityLines({
          screenNodes: previousScreenNodes,
          centerNodeId: centerNode?.id,
          highlightedNodeIds: baseLineHighlightedSet,
          devicePixelRatio,
          lineTailAlpha: renderVariant.lineTailAlpha,
        });
        const currentPreview = buildProximityLines({
          screenNodes: buildPreviewScreenNodes(renderNodesRef.current),
          centerNodeId: centerNode?.id,
          highlightedNodeIds: baseLineHighlightedSet,
          devicePixelRatio,
          lineTailAlpha: renderVariant.lineTailAlpha,
        });
        const effectiveRemovedNodeIds = new Set(pendingTransition.removedNodeIds);
        for (const node of previousScreenNodes) {
          if (!activeIds.has(node.id)) {
            effectiveRemovedNodeIds.add(node.id);
          }
        }
        const effectiveAddedNodeIds = new Set(pendingTransition.addedNodeIds);
        const replacingTransition = lineTransitionRef.current;
        if (replacingTransition) {
          for (const id of replacingTransition.addedNodeIds) {
            effectiveAddedNodeIds.add(id);
          }
        }

        const { outgoingPairKeys, incomingPairKeys } =
          narrowPairDiffToChangedComponent({
            previousPairKeys: previousPreview.pairKeys ?? [],
            currentPairKeys: currentPreview.pairKeys ?? [],
            previousPairRouteSignatures: previousPreview.pairRouteSignatures,
            currentPairRouteSignatures: currentPreview.pairRouteSignatures,
            addedNodeIds: effectiveAddedNodeIds,
            removedNodeIds: effectiveRemovedNodeIds,
          });

        // Detect repositioning nodes (surviving nodes that will move >18px)
        // and include their edges in outgoing/incoming so they fade out before motion.
        const repositioningNodeIds = new Set<string>();
      for (const node of renderNodesRef.current) {
        const currentZoom = zoomRef.current;
          if (node.id === selfNodeId) continue;
          const prev = lastScreenPositionsRef.current.get(node.id);
          if (prev) {
            const nextX = node.x * cssWidth;
            const nextY = node.y * cssHeight;
            if (Math.hypot(nextX - prev.x, nextY - prev.y) > 18) {
              repositioningNodeIds.add(node.id);
            }
          }
        }
        if (repositioningNodeIds.size > 0) {
          const allPairKeys = new Set([
            ...(previousPreview.pairKeys ?? []),
            ...(currentPreview.pairKeys ?? []),
          ]);
          for (const pairKey of allPairKeys) {
            const sep = pairKey.indexOf("::");
            if (sep < 0) continue;
            if (
              repositioningNodeIds.has(pairKey.slice(0, sep)) ||
              repositioningNodeIds.has(pairKey.slice(sep + 2))
            ) {
              outgoingPairKeys.add(pairKey);
              incomingPairKeys.add(pairKey);
            }
          }
        }

        const stableVisiblePairKeys = new Set<string>([
          ...(previousPreview.pairKeys ?? []),
          ...(currentPreview.pairKeys ?? []),
        ]);
        for (const pairKey of outgoingPairKeys) {
          stableVisiblePairKeys.delete(pairKey);
        }
        for (const pairKey of incomingPairKeys) {
          stableVisiblePairKeys.delete(pairKey);
        }

        const effectiveHopDelay = hopDelayMs ?? 60;
        const staggerEnabled =
          effectiveHopDelay > 0 &&
          incomingPairKeys.size >= STAGGER_THRESHOLD &&
          centerNode != null;
        const hopStagger = staggerEnabled
          ? computePairHopDelays(
              currentPreview.pairKeys ?? [],
              centerNode!.id,
              effectiveHopDelay,
            )
          : undefined;

        const outgoingPairStartAlphas = (() => {
          const old = lineTransitionRef.current;
          if (!old || outgoingPairKeys.size === 0) return undefined;
          const oldFadeProgress = clamp(
            (now - old.startedAt) / LINE_REVEAL_DURATION_MS,
            0,
            1,
          );
          const oldOutgoingAlpha = 1 - easeOutCubic(oldFadeProgress);
          const alphas = new Map<string, number>();
          for (const pairKey of outgoingPairKeys) {
            if (old.outgoingPairKeys.has(pairKey)) {
              const base = old.outgoingPairStartAlphas?.get(pairKey) ?? 1;
              alphas.set(pairKey, base * oldOutgoingAlpha);
            }
          }
          return alphas.size > 0 ? alphas : undefined;
        })();

        lineTransitionRef.current =
          outgoingPairKeys.size > 0 || incomingPairKeys.size > 0
            ? {
                snapshotScreenNodes: previousScreenNodes.map((node) => ({
                  ...node,
                })),
                stableVisiblePairKeys,
                outgoingPairKeys,
                incomingPairKeys,
                addedNodeIds: effectiveAddedNodeIds,
                startedAt: now,
                entryStartedAt:
                  now + (outgoingPairKeys.size > 0 ? LINE_REVEAL_DURATION_MS : 0),
                revealStartedAt:
                  now +
                  (outgoingPairKeys.size > 0 ? LINE_REVEAL_DURATION_MS : 0) +
                  (effectiveAddedNodeIds.size > 0 ||
                  effectiveRemovedNodeIds.size > 0
                    ? ENTRY_ANIMATION_DURATION_MS
                    : 0),
                outgoingPairStartAlphas,
                pairHopDelays: hopStagger?.pairHopDelays,
                maxHopDelay: hopStagger?.maxHopDelay,
              }
            : null;
        pendingLineTransitionRef.current = null;
      }

      const activeLineTransition =
        lineTransitionRef.current;
      if (
        activeLineTransition &&
        now - activeLineTransition.revealStartedAt >
          LINE_REVEAL_DURATION_MS + (activeLineTransition.maxHopDelay ?? 0) + 32
      ) {
        lineTransitionRef.current = null;
      }

      for (const key of animationRef.current.keys()) {
        if (!activeIds.has(key)) {
          animationRef.current.delete(key);
        }
      }
      for (const [key, exit] of exitAnimationRef.current.entries()) {
        if (activeIds.has(key)) {
          exitAnimationRef.current.delete(key);
          continue;
        }
        if (now - exit.startedAt > 320) {
          exitAnimationRef.current.delete(key);
        }
      }
      for (const [key, twinkle] of twinkleAnimationRef.current.entries()) {
        if (!activeIds.has(key) || now - twinkle.startedAt > 1100) {
          twinkleAnimationRef.current.delete(key);
        }
      }
      for (const [key, reveal] of lineRevealRef.current.entries()) {
        if (!activeIds.has(key) || now - reveal.startedAt > LINE_REVEAL_DURATION_MS + 32) {
          lineRevealRef.current.delete(key);
        }
      }

      for (const node of renderNodesRef.current) {
        const currentZoom = zoomRef.current;
        const currentPan = panRef.current;
        const shouldHoldAddedNode =
          activeLineTransition?.addedNodeIds.has(node.id) &&
          now < activeLineTransition.entryStartedAt;
        if (shouldHoldAddedNode) {
          continue;
        }
        if (
          node.id !== selfNodeId &&
          !seenNodeIdsRef.current.has(node.id) &&
          !animationRef.current.has(node.id)
        ) {
          const stagedEntryStart = activeLineTransition?.addedNodeIds.has(node.id)
            ? activeLineTransition.entryStartedAt
            : now;
          const approachAngle = hashString(`${node.id}:entry-angle`) * TAU;
          const entryRadius =
            Math.max(cssWidth, cssHeight) *
            (0.72 + hashString(`${node.id}:entry-radius`) * 0.28);
          const fromX = cssWidth * 0.5 + Math.cos(approachAngle) * entryRadius;
          const fromY = cssHeight * 0.5 + Math.sin(approachAngle) * entryRadius;
          const toX = node.x * cssWidth;
          const toY = node.y * cssHeight;
          animationRef.current.set(
            node.id,
            createTravelAnimation(
              node.id,
              fromX,
              fromY,
              toX,
              toY,
              stagedEntryStart,
              `${node.id}:entry`,
              true,
            ),
          );
        } else if (node.id !== selfNodeId && !animationRef.current.has(node.id)) {
          const nextX = node.x * cssWidth;
          const nextY = node.y * cssHeight;
          const previousPosition = lastScreenPositionsRef.current.get(node.id);
          if (previousPosition) {
            const moveDistance = Math.hypot(
              nextX - previousPosition.x,
              nextY - previousPosition.y,
            );
            if (moveDistance > 18) {
              const stagedStart =
                activeLineTransition &&
                activeLineTransition.entryStartedAt > now
                  ? activeLineTransition.entryStartedAt
                  : now;
              animationRef.current.set(
                node.id,
                createTravelAnimation(
                  node.id,
                  previousPosition.x,
                  previousPosition.y,
                  nextX,
                  nextY,
                  stagedStart,
                  `${node.id}:move`,
                  lineTransitionRef.current != null,
                  true,
                ),
              );
            }
          }
        }

        let px = node.x * cssWidth;
        let py = node.y * cssHeight;
        let entryOpacity = 1;
        let entryScale = 1;
        let lineRevealProgress = 1;
        const entry = animationRef.current.get(node.id);
        if (entry) {
          entry.toX = node.x * cssWidth;
          entry.toY = node.y * cssHeight;
          if (now < entry.startedAt) {
            px = entry.fromX;
            py = entry.fromY;
            if (entry.isReposition) {
              entryOpacity = 1;
              entryScale = 1;
            } else {
              entryOpacity = 0;
              entryScale = 0.72;
            }
            if (entry.hideLinksUntilSettled) {
              lineRevealProgress = 0;
            }
          } else {
            const progress = clamp(
              (now - entry.startedAt) / ENTRY_ANIMATION_DURATION_MS,
              0,
              1,
            );
            const eased = easeOutCubic(progress);
            px = cubicBezier(entry.fromX, entry.control1X, entry.control2X, entry.toX, eased);
            py = cubicBezier(entry.fromY, entry.control1Y, entry.control2Y, entry.toY, eased);
            const meanderEnvelope = Math.sin(progress * Math.PI);
            const meanderOffset =
              Math.sin(progress * Math.PI * entry.meanderCycles + entry.meanderPhase) *
              entry.meanderAmplitude *
              meanderEnvelope;
            px += entry.normalX * meanderOffset;
            py += entry.normalY * meanderOffset;
            entryOpacity = 0.2 + eased * 0.8;
            entryScale = 0.72 + eased * 0.28;
            if (entry.hideLinksUntilSettled) {
              lineRevealProgress = 0;
            }
            if (progress >= 1) {
              animationRef.current.delete(node.id);
              px = entry.toX;
              py = entry.toY;
              entryOpacity = 1;
              entryScale = 1;
              if (entry.hideLinksUntilSettled) {
                lineRevealRef.current.set(node.id, { startedAt: now });
              }
            }
          }
        }
        const lineReveal = lineRevealRef.current.get(node.id);
        if (lineReveal) {
          const progress = clamp(
            (now - lineReveal.startedAt) / LINE_REVEAL_DURATION_MS,
            0,
            1,
          );
          lineRevealProgress = easeOutCubic(progress);
          if (progress >= 1) {
            lineRevealRef.current.delete(node.id);
          }
        }
        seenNodeIdsRef.current.add(node.id);
        px = px * currentZoom + currentPan.x;
        py = py * currentZoom + currentPan.y;
        const isHighlighted = pointHighlightedSet.has(node.id);
        const twinkle = twinkleAnimationRef.current.get(node.id);
        const twinkleProgress = twinkle
          ? clamp((now - twinkle.startedAt) / 1100, 0, 1)
          : 1;
        const twinkleStrength = twinkle
          ? Math.sin(twinkleProgress * Math.PI * 3.2) * (1 - twinkleProgress) * 0.42
          : 0;
        const size = node.size * entryScale;
        const hitSize =
          (node.size + (isHighlighted ? 6 : 0) + twinkleStrength * 16) * entryScale;
        const colorBoost = isHighlighted ? 0.18 : 0;
        const twinkleBoost = Math.max(0, twinkleStrength);
        pointPositions.push(px * devicePixelRatio, py * devicePixelRatio);
        pointSizes.push(hitSize * devicePixelRatio);
        pointColors.push(
          clamp(node.color[0] + colorBoost + twinkleBoost * 0.72, 0, 1),
          clamp(node.color[1] + colorBoost + twinkleBoost * 0.72, 0, 1),
          clamp(node.color[2] + colorBoost + twinkleBoost * 0.78, 0, 1),
          clamp(node.color[3] * entryOpacity + twinkleBoost * 0.34, 0, 1),
        );
        pointPulses.push(node.pulse + (isHighlighted ? 0.35 : 0) + twinkleBoost * 2.2);
        pointTwinkles.push(twinkleBoost);
        screenNodes.push({
          ...node,
          px,
          py,
          hitSize,
          size,
          lineRevealProgress,
        });
        lastScreenPositionsRef.current.set(node.id, {
          x: node.x * cssWidth,
          y: node.y * cssHeight,
        });
      }

      for (const { node, startedAt } of exitAnimationRef.current.values()) {
        const progress = clamp((now - startedAt) / 320, 0, 1);
        const flash = Math.sin(progress * Math.PI);
        const shockwave = Math.sin(progress * Math.PI * 0.92);
        const shockFront = Math.max(0, Math.sin(progress * Math.PI * 1.35 - 0.35));
        const collapse = 1 - Math.pow(progress, 1.55);
        const currentZoom = zoomRef.current;
        const currentPan = panRef.current;
        const px = node.x * cssWidth * currentZoom + currentPan.x;
        const py = node.y * cssHeight * currentZoom + currentPan.y;
        const exitScale = 1 + flash * 1.15 + shockFront * 0.42 + progress * 0.22;
        const exitAlpha = collapse * (0.28 + flash * 1.1 + shockFront * 0.18);
        const whiteCore = Math.max(0, Math.sin(progress * Math.PI * 1.8)) * (1 - progress);
        const hotCore = flash * 0.68 + whiteCore * 0.42;
        const coolFade = Math.max(0, 1 - progress * 1.25);

        pointPositions.push(px * devicePixelRatio, py * devicePixelRatio);
        pointSizes.push(node.size * exitScale * devicePixelRatio);
        pointColors.push(
          clamp(node.color[0] + hotCore * 1.25 + 0.26, 0, 1),
          clamp(node.color[1] + hotCore * 0.82 + 0.16, 0, 1),
          clamp(
            node.color[2] + coolFade * 0.34 + shockwave * 0.16 + whiteCore * 0.12,
            0,
            1,
          ),
          clamp(node.color[3] * exitAlpha, 0, 1),
        );
        pointPulses.push(node.pulse + flash * 2.8 + shockwave * 0.9 + shockFront * 0.65);
        pointTwinkles.push(Math.min(1, whiteCore * 1.35 + shockFront * 0.3));
      }

      const stableLineScreenNodes = sortStableLineNodes(screenNodes);
      const hitTestScreenNodes = [...screenNodes].sort(
        (left, right) => right.z - left.z || right.hitSize - left.hitSize,
      );
      lineScreenNodesRef.current = stableLineScreenNodes;
      screenNodesRef.current = hitTestScreenNodes;
      for (const key of lastScreenPositionsRef.current.keys()) {
        if (!activeIds.has(key)) {
          lastScreenPositionsRef.current.delete(key);
          seenNodeIdsRef.current.delete(key);
        }
      }

      const activeTransition =
        lineTransitionRef.current;

      let lineFp = stableLineScreenNodes.length;
      const centerId = centerNode?.id ?? "";
      for (let i = 0; i < centerId.length; i++) {
        lineFp = ((lineFp * 0x01000193) ^ centerId.charCodeAt(i)) | 0;
      }
      lineFp = ((lineFp * 0x01000193) ^ ((devicePixelRatio * 100) | 0)) | 0;
      for (const n of stableLineScreenNodes) {
        lineFp = ((lineFp * 0x01000193) ^ ((n.px * 100) | 0)) | 0;
        lineFp = ((lineFp * 0x01000193) ^ ((n.py * 100) | 0)) | 0;
        lineFp = ((lineFp * 0x01000193) ^ ((n.lineRevealProgress * 1000) | 0)) | 0;
      }

      const lineInputChanged = lineFp !== _lineFp;
      if (lineInputChanged) {
        _lineCache = renderVariant.buildLines({
          screenNodes: stableLineScreenNodes,
          centerNodeId: centerNode?.id,
          highlightedNodeIds: baseLineHighlightedSet,
          devicePixelRatio,
          lineTailAlpha: renderVariant.lineTailAlpha,
        });
        _lineFp = lineFp;
      }
      const currentBaseLinePreview = _lineCache;

      const baseLineOutput =
        activeTransition
          ? (() => {
              const fadeOutProgress = clamp(
                (now - activeTransition.startedAt) / LINE_REVEAL_DURATION_MS,
                0,
                1,
              );
              const outgoingAlpha = 1 - easeOutCubic(fadeOutProgress);
              const hasAddedNodes = activeTransition.addedNodeIds.size > 0;
              const currentVisiblePairKeys = new Set<string>();
              const currentPairAlphaOverrides = new Map<string, number>();
              const previousPairAlphaOverrides = new Map<string, number>();

              for (const pairKey of activeTransition.outgoingPairKeys) {
                const startAlpha = activeTransition.outgoingPairStartAlphas?.get(pairKey) ?? 1;
                previousPairAlphaOverrides.set(pairKey, startAlpha * outgoingAlpha);
              }

              if (hasAddedNodes && now < activeTransition.revealStartedAt) {
                return buildProximityLines({
                  screenNodes: activeTransition.snapshotScreenNodes,
                  centerNodeId: centerNode?.id,
                  highlightedNodeIds: baseLineHighlightedSet,
                  devicePixelRatio,
                  lineTailAlpha: renderVariant.lineTailAlpha,
                  pairAlphaOverrides: previousPairAlphaOverrides,
                });
              }

              if (!hasAddedNodes && now < activeTransition.revealStartedAt) {
                return buildProximityLines({
                  screenNodes: activeTransition.snapshotScreenNodes,
                  centerNodeId: centerNode?.id,
                  highlightedNodeIds: baseLineHighlightedSet,
                  devicePixelRatio,
                  lineTailAlpha: renderVariant.lineTailAlpha,
                  pairAlphaOverrides: previousPairAlphaOverrides,
                });
              }

              for (const pairKey of activeTransition.stableVisiblePairKeys) {
                currentVisiblePairKeys.add(pairKey);
              }

              for (const pairKey of currentBaseLinePreview.pairKeys ?? []) {
                currentVisiblePairKeys.add(pairKey);
                if (activeTransition.incomingPairKeys.has(pairKey)) {
                  const hopDelay = activeTransition.pairHopDelays?.get(pairKey) ?? 0;
                  const pairRevealProgress = clamp(
                    (now - activeTransition.revealStartedAt - hopDelay) / LINE_REVEAL_DURATION_MS,
                    0,
                    1,
                  );
                  currentPairAlphaOverrides.set(pairKey, easeOutCubic(pairRevealProgress));
                }
              }

              return buildProximityLines({
                screenNodes: stableLineScreenNodes,
                centerNodeId: centerNode?.id,
                highlightedNodeIds: baseLineHighlightedSet,
                devicePixelRatio,
                lineTailAlpha: renderVariant.lineTailAlpha,
                visiblePairKeys: currentVisiblePairKeys,
                pairAlphaOverrides: currentPairAlphaOverrides,
              });
            })()
          : currentBaseLinePreview;
      const meshInputSame =
        baseLineOutput.positions === _meshPosRef &&
        baseLineOutput.colors === _meshColorRef &&
        renderVariant.lineWidthPx === _meshWidthPx &&
        devicePixelRatio === _meshDpr;
      const baseLineMesh = meshInputSame
        ? _meshCache
        : buildLineMesh({
            positions: baseLineOutput.positions,
            colors: baseLineOutput.colors,
            lineWidthPx: renderVariant.lineWidthPx,
            devicePixelRatio,
          });
      if (!meshInputSame) {
        _meshPosRef = baseLineOutput.positions;
        _meshColorRef = baseLineOutput.colors;
        _meshWidthPx = renderVariant.lineWidthPx;
        _meshDpr = devicePixelRatio;
        _meshCache = baseLineMesh;
      }

      _pointPosBuf = ensureFloat32(_pointPosBuf, pointPositions.length);
      _pointSizeBuf = ensureFloat32(_pointSizeBuf, pointSizes.length);
      _pointColorBuf = ensureFloat32(_pointColorBuf, pointColors.length);
      _pointPulseBuf = ensureFloat32(_pointPulseBuf, pointPulses.length);
      _pointTwinkleBuf = ensureFloat32(_pointTwinkleBuf, pointTwinkles.length);
      _pointPosBuf.set(pointPositions);
      _pointSizeBuf.set(pointSizes);
      _pointColorBuf.set(pointColors);
      _pointPulseBuf.set(pointPulses);
      _pointTwinkleBuf.set(pointTwinkles);

      return {
        pointPositions: new Float32Array(_pointPosBuf.buffer, 0, pointPositions.length),
        pointSizes: new Float32Array(_pointSizeBuf.buffer, 0, pointSizes.length),
        pointColors: new Float32Array(_pointColorBuf.buffer, 0, pointColors.length),
        pointPulses: new Float32Array(_pointPulseBuf.buffer, 0, pointPulses.length),
        pointTwinkles: new Float32Array(_pointTwinkleBuf.buffer, 0, pointTwinkles.length),
        linePositions: baseLineMesh.positions,
        lineColors: baseLineMesh.colors,
        lineCoords: baseLineMesh.lineCoords,
      };
    };

    const drawLineMesh = (
      positions: Float32Array,
      colors: Float32Array,
      coords: Float32Array,
    ) => {
      if (positions.length === 0) {
        return;
      }

      renderVariant.applyLineBlendMode(gl);
      gl["useProgram"](lineProgram);
      gl.uniform2f(lineResolutionLocation, width, height);
      disableVertexAttributes(pointAttributeLocations);

      _glLinePosCap = uploadToGLBuffer(linePositionBuffer, positions, _glLinePosCap);
      gl.enableVertexAttribArray(linePositionLocation);
      gl.vertexAttribPointer(linePositionLocation, 2, gl.FLOAT, false, 0, 0);

      _glLineColorCap = uploadToGLBuffer(lineColorBuffer, colors, _glLineColorCap);
      gl.enableVertexAttribArray(lineColorLocation);
      gl.vertexAttribPointer(lineColorLocation, 4, gl.FLOAT, false, 0, 0);

      _glLineCoordCap = uploadToGLBuffer(lineCoordBuffer, coords, _glLineCoordCap);
      gl.enableVertexAttribArray(lineCoordLocation);
      gl.vertexAttribPointer(lineCoordLocation, 2, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.TRIANGLES, 0, positions.length / 2);
    };

    let _lastHoveredId: string | null = null;
    let _lastZoom = NaN;
    let _lastPanX = NaN;
    let _lastPanY = NaN;
    let _frameDataCache: ReturnType<typeof buildFrameData> | null = null;

    const render = () => {
      frame += 1;

      const hoveredId = hoveredNodeIdRef.current ?? null;
      const curZoom = zoomRef.current;
      const curPan = panRef.current;
      const needsBuild =
        _frameDataCache === null
        || _resizeDirty
        || hoveredId !== _lastHoveredId
        || curZoom !== _lastZoom
        || curPan.x !== _lastPanX
        || curPan.y !== _lastPanY
        || animationRef.current.size > 0
        || exitAnimationRef.current.size > 0
        || twinkleAnimationRef.current.size > 0
        || lineRevealRef.current.size > 0
        || lineTransitionRef.current !== null
        || pendingLineTransitionRef.current !== null;

      _lastHoveredId = hoveredId;
      _lastZoom = curZoom;
      _lastPanX = curPan.x;
      _lastPanY = curPan.y;
      if (_resizeDirty) _resizeDirty = false;

      if (needsBuild) {
        _frameDataCache = buildFrameData();
      }
      const {
        pointPositions,
        pointSizes,
        pointColors,
        pointPulses,
        pointTwinkles,
        linePositions,
        lineColors,
        lineCoords,
      } = _frameDataCache!;

      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);

      drawLineMesh(linePositions, lineColors, lineCoords);

      renderVariant.applyPointBlendMode(gl);
      gl["useProgram"](pointProgram);
      gl.uniform2f(pointResolutionLocation, width, height);
      gl.uniform1f(pointTimeLocation, frame / 60);
      disableVertexAttributes(lineAttributeLocations);

      _glPointPosCap = uploadToGLBuffer(pointPositionBuffer, pointPositions, _glPointPosCap);
      gl.enableVertexAttribArray(pointPositionLocation);
      gl.vertexAttribPointer(pointPositionLocation, 2, gl.FLOAT, false, 0, 0);

      _glPointSizeCap = uploadToGLBuffer(pointSizeBuffer, pointSizes, _glPointSizeCap);
      gl.enableVertexAttribArray(pointSizeLocation);
      gl.vertexAttribPointer(pointSizeLocation, 1, gl.FLOAT, false, 0, 0);

      _glPointColorCap = uploadToGLBuffer(pointColorBuffer, pointColors, _glPointColorCap);
      gl.enableVertexAttribArray(pointColorLocation);
      gl.vertexAttribPointer(pointColorLocation, 4, gl.FLOAT, false, 0, 0);

      _glPointPulseCap = uploadToGLBuffer(pointPulseBuffer, pointPulses, _glPointPulseCap);
      gl.enableVertexAttribArray(pointPulseLocation);
      gl.vertexAttribPointer(pointPulseLocation, 1, gl.FLOAT, false, 0, 0);

      _glPointTwinkleCap = uploadToGLBuffer(pointTwinkleBuffer, pointTwinkles, _glPointTwinkleCap);
      gl.enableVertexAttribArray(pointTwinkleLocation);
      gl.vertexAttribPointer(pointTwinkleLocation, 1, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.POINTS, 0, pointPositions.length / 2);

      animationFrame = window.requestAnimationFrame(render);
    };

    resize();
    render();

    const observer = new ResizeObserver(() => resize());
    observer.observe(host);

    return () => {
      window.cancelAnimationFrame(animationFrame);
      observer.disconnect();
      disableVertexAttributes(lineAttributeLocations);
      disableVertexAttributes(pointAttributeLocations);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      gl.deleteBuffer(pointPositionBuffer);
      gl.deleteBuffer(pointSizeBuffer);
      gl.deleteBuffer(pointColorBuffer);
      gl.deleteBuffer(pointPulseBuffer);
      gl.deleteBuffer(pointTwinkleBuffer);
      gl.deleteBuffer(linePositionBuffer);
      gl.deleteBuffer(lineColorBuffer);
      gl.deleteBuffer(lineCoordBuffer);
      gl.deleteProgram(pointProgram);
      gl.deleteProgram(lineProgram);
    };
  }, [
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
    renderVariant,
    screenNodesRef,
    seenNodeIdsRef,
    selfNodeId,
    hopDelayMs,
    twinkleAnimationRef,
    zoomRef,
  ]);
}
