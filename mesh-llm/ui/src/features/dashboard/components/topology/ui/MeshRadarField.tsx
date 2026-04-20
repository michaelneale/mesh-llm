import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Clock3,
  Cpu,
  MemoryStick,
  Minus,
  Plus,
  RotateCcw,
  Sparkles,
  Wifi,
} from "lucide-react";

import { formatShortDuration } from "../../../../../lib/format-duration";
import { useResolvedTheme } from "../../../../../lib/resolved-theme";
import { cn } from "../../../../../lib/utils";
import { formatLiveNodeState, shortName } from "../../../../app-shell/lib/status-helpers";
import type { LiveNodeState } from "../../../../app-shell/lib/status-types";
import type { TopologyNode } from "../../../../app-shell/lib/topology-types";

import { nodeUpdateSignature, clamp } from "../helpers";
import { useRadarFieldNodes } from "../layout/useRadarFieldNodes";
import { useTopologyCanvas } from "../render/useTopologyCanvas";
import { RENDER_VARIANTS } from "../theme/render-variants";
import type {
  EntryAnimation,
  ExitAnimation,
  LineTransition,
  LineRevealAnimation,
  MeshRadarFieldProps,
  PendingLineTransition,
  RenderNode,
  ScreenNode,
  UpdateTwinkle,
} from "../types";

function createDebugNode(
  index: number,
  selectedModel: string,
  fallbackModel: string,
): TopologyNode {
  const pattern = index % 3;
  const model =
    selectedModel && selectedModel !== "auto"
      ? selectedModel
      : fallbackModel || "Qwen3-8B";

  if (pattern === 0) {
    return {
      id: `debug-serving-${index}`,
      vram: 24 + (index % 4) * 6,
      self: false,
      host: false,
      client: false,
      serving: model,
      servingModels: [model],
      state: "serving" as LiveNodeState,
      latencyMs: 14 + (index % 5) * 8,
      hostname: `test-serving-${index}`,
      isSoc: false,
      gpus: [{ name: "Synthetic GPU", vram_bytes: 24 * 1024 ** 3 }],
    };
  }

  if (pattern === 1) {
    return {
      id: `debug-worker-${index}`,
      vram: 12 + (index % 3) * 4,
      self: false,
      host: false,
      client: false,
      serving: "",
      servingModels: [],
      state: "standby" as LiveNodeState,
      latencyMs: 24 + (index % 6) * 10,
      hostname: `test-worker-${index}`,
      isSoc: index % 2 === 0,
      gpus:
        index % 2 === 0
          ? []
          : [{ name: "Synthetic GPU", vram_bytes: 12 * 1024 ** 3 }],
    };
  }

  return {
    id: `debug-client-${index}`,
    vram: 0,
    self: false,
    host: false,
    client: true,
      serving: "",
      servingModels: [],
      state: "client" as LiveNodeState,
      latencyMs: 42 + (index % 7) * 12,
    hostname: `test-client-${index}`,
    isSoc: false,
    gpus: [],
  };
}

export function MeshRadarField({
  status,
  nodes,
  selectedModel,
  themeMode,
  onOpenNode,
  highlightedNodeId,
  fullscreen,
  heightClass,
  containerStyle,
}: MeshRadarFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const hostRef = useRef<HTMLDivElement | null>(null);
  const screenNodesRef = useRef<ScreenNode[]>([]);
  const animationRef = useRef<Map<string, EntryAnimation>>(new Map());
  const lineRevealRef = useRef<Map<string, LineRevealAnimation>>(new Map());
  const exitAnimationRef = useRef<Map<string, ExitAnimation>>(new Map());
  const twinkleAnimationRef = useRef<Map<string, UpdateTwinkle>>(new Map());
  const pendingLineTransitionRef =
    useRef<PendingLineTransition | null>(null);
  const lineTransitionRef = useRef<LineTransition | null>(null);
  const previousRenderNodesRef = useRef<Map<string, RenderNode>>(new Map());
  const lastScreenPositionsRef = useRef<Map<string, { x: number; y: number }>>(
    new Map(),
  );
  const previousNodeSignaturesRef = useRef<Map<string, string>>(
    new Map(nodes.map((node) => [node.id, nodeUpdateSignature(node)])),
  );
  const seenNodeIdsRef = useRef<Set<string>>(new Set(nodes.map((node) => node.id)));
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const [hoveredNode, setHoveredNode] = useState<ScreenNode | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string>(() => {
    const selfNode = nodes.find((node) => node.self);
    return highlightedNodeId || selfNode?.id || nodes[0]?.id || "";
  });
  const selectedNodeIdRef = useRef(selectedNodeId);
  const hoveredNodeIdRef = useRef<string | null>(null);
  const debugNodeCounterRef = useRef(0);
  const resolvedTheme = useResolvedTheme(themeMode);
  const renderVariant = RENDER_VARIANTS[resolvedTheme];
  const scene = renderVariant.scene;
  const SelfNodeAccentComponent = renderVariant.SelfNodeAccent;
  const selfNode = useMemo(() => nodes.find((node) => node.self) ?? nodes[0], [nodes]);
  const [tooltipStyle, setTooltipStyle] = useState<CSSProperties | null>(null);
  const [hostSize, setHostSize] = useState({ width: 0, height: 0 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const zoomRef = useRef(1);
  const panRef = useRef({ x: 0, y: 0 });
  const [debugNodes, setDebugNodes] = useState<TopologyNode[]>([]);
  const [hopDelayMs, setHopDelayMs] = useState(60);
  const displayNodes = useMemo(() => [...nodes, ...debugNodes], [debugNodes, nodes]);
  const renderNodes = useRadarFieldNodes(
    displayNodes,
    selectedModel,
    status.model_name ?? "",
    renderVariant,
  );
  const dragRef = useRef<{
    active: boolean;
    originX: number;
    originY: number;
    panX: number;
    panY: number;
    moved: boolean;
  }>({
    active: false,
    originX: 0,
    originY: 0,
    panX: 0,
    panY: 0,
    moved: false,
  });

  useEffect(() => {
    if (highlightedNodeId) {
      setSelectedNodeId(highlightedNodeId);
    }
  }, [highlightedNodeId]);

  useEffect(() => {
    selectedNodeIdRef.current = selectedNodeId;
  }, [selectedNodeId]);

  useEffect(() => {
    hoveredNodeIdRef.current = hoveredNode?.id ?? null;
  }, [hoveredNode?.id]);

  useEffect(() => {
    setDebugNodes((current) =>
      current.filter((node) => !nodes.some((realNode) => realNode.id === node.id)),
    );
  }, [nodes]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const updateSize = () => {
      const rect = host.getBoundingClientRect();
      setHostSize({ width: rect.width, height: rect.height });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(host);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const previousSignatures = previousNodeSignaturesRef.current;
    const nextSignatures = new Map<string, string>();
    const now = performance.now();

    for (const node of displayNodes) {
      const signature = nodeUpdateSignature(node);
      nextSignatures.set(node.id, signature);
      const previousSignature = previousSignatures.get(node.id);
      if (previousSignature && previousSignature !== signature) {
        twinkleAnimationRef.current.set(node.id, { startedAt: now });
      }
    }

    for (const key of twinkleAnimationRef.current.keys()) {
      if (!nextSignatures.has(key)) {
        twinkleAnimationRef.current.delete(key);
      }
    }

    previousNodeSignaturesRef.current = nextSignatures;
  }, [displayNodes]);

  useEffect(() => {
    const previous = previousRenderNodesRef.current;
    const currentIds = new Set(renderNodes.map((node) => node.id));
    const previousIds = new Set(previous.keys());
    const addedNodeIds = new Set([...currentIds].filter((id) => !previousIds.has(id)));
    const removedNodeIds = new Set([...previousIds].filter((id) => !currentIds.has(id)));
    const now = performance.now();

    pendingLineTransitionRef.current =
      addedNodeIds.size > 0 || removedNodeIds.size > 0
        ? { addedNodeIds, removedNodeIds }
        : null;

    for (const [id, node] of previous.entries()) {
      if (!currentIds.has(id) && id !== selfNode?.id && !exitAnimationRef.current.has(id)) {
        exitAnimationRef.current.set(id, {
          node,
          startedAt: now,
        });
      }
    }

    previousRenderNodesRef.current = new Map(renderNodes.map((node) => [node.id, node]));
  }, [renderNodes, selfNode?.id]);

  useEffect(() => {
    if (!hoveredNode || !hostRef.current || !tooltipRef.current) {
      setTooltipStyle(null);
      return;
    }

    const hostRect = hostRef.current.getBoundingClientRect();
    const tooltipRect = tooltipRef.current.getBoundingClientRect();
    const nodeX = hoveredNode.px;
    const nodeY = hoveredNode.py;
    const gutter = 16;

    let left = nodeX + gutter;
    let top = nodeY - 12;
    let transform = "translateY(-100%)";

    if (left + tooltipRect.width > hostRect.width - 8) {
      left = nodeX - tooltipRect.width - gutter;
    }
    if (left < 8) {
      left = 8;
    }

    const topWithTransform = top - tooltipRect.height;
    if (topWithTransform < 8) {
      top = Math.min(nodeY + gutter, hostRect.height - tooltipRect.height - 8);
      transform = "none";
    }

    setTooltipStyle({
      left: `${left}px`,
      top: `${top}px`,
      transform,
    });
  }, [hoveredNode]);

  useTopologyCanvas({
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
    selfNodeId: selfNode?.id,
    hopDelayMs,
  });

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragRef.current.active) {
      const nextPanX = dragRef.current.panX + (event.clientX - dragRef.current.originX);
      const nextPanY = dragRef.current.panY + (event.clientY - dragRef.current.originY);
      if (
        Math.abs(event.clientX - dragRef.current.originX) > 3 ||
        Math.abs(event.clientY - dragRef.current.originY) > 3
      ) {
        dragRef.current.moved = true;
      }
      const nextPan = { x: nextPanX, y: nextPanY };
      panRef.current = nextPan;
      setPan(nextPan);
      setHoveredNode(null);
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const nextNode =
      screenNodesRef.current.find((node) => {
        const dx = x - node.px;
        const dy = y - node.py;
        return Math.hypot(dx, dy) <= node.hitSize * 0.8 + 10;
      }) ?? null;

    setHoveredNode((previous) => (previous?.id === nextNode?.id ? previous : nextNode));
  };

  const handlePointerLeave = () => {
    dragRef.current.active = false;
    setHoveredNode(null);
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    dragRef.current = {
      active: true,
      originX: event.clientX,
      originY: event.clientY,
      panX: panRef.current.x,
      panY: panRef.current.y,
      moved: false,
    };
  };

  const handlePointerUp = () => {
    const shouldSelect = dragRef.current.active && !dragRef.current.moved;
    dragRef.current.active = false;
    if (shouldSelect) {
      handleClick();
    }
  };

  const handleClick = () => {
    if (dragRef.current.moved || !hoveredNode) return;
    setSelectedNodeId(hoveredNode.id);
    onOpenNode?.(hoveredNode.id);
  };

  const zoomAroundPoint = (nextZoom: number, anchorX: number, anchorY: number) => {
    const currentZoom = zoomRef.current;
    const currentPan = panRef.current;
    const worldX = (anchorX - currentPan.x) / currentZoom;
    const worldY = (anchorY - currentPan.y) / currentZoom;
    const nextPan = {
      x: anchorX - worldX * nextZoom,
      y: anchorY - worldY * nextZoom,
    };
    zoomRef.current = nextZoom;
    panRef.current = nextPan;
    setZoom(nextZoom);
    setPan(nextPan);
  };

  const handleWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (!hostRef.current) return;
    const rect = hostRef.current.getBoundingClientRect();
    const pointerX = event.clientX - rect.left;
    const pointerY = event.clientY - rect.top;
    const nextZoom = clamp(zoomRef.current * (event.deltaY > 0 ? 0.92 : 1.08), 0.7, 2.4);
    zoomAroundPoint(nextZoom, pointerX, pointerY);
  };

  const resetView = () => {
    zoomRef.current = 1;
    panRef.current = { x: 0, y: 0 };
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const selfRenderNode =
    renderNodes.find((node) => node.id === selfNode?.id) ?? renderNodes[renderNodes.length - 1];
  const isSelfNodeTooltipActive =
    resolvedTheme === "light" && hoveredNode?.id === selfRenderNode?.id && tooltipStyle != null;
  const selfScreenX = selfRenderNode
    ? selfRenderNode.x * hostSize.width * zoom + pan.x
    : 0.5 * hostSize.width * zoom + pan.x;
  const selfScreenY = selfRenderNode
    ? selfRenderNode.y * hostSize.height * zoom + pan.y
    : 0.52 * hostSize.height * zoom + pan.y;
  const focalPoint = `${selfScreenX}px ${selfScreenY}px`;
  const gridSize = fullscreen ? 64 : 56;
  const selfAccentScale = 0.92 + zoom * 0.08;
  const selfLabelOffset = 38;

  const selfLabelStyle: CSSProperties = {
    left: `${selfScreenX}px`,
    top: `${selfScreenY + selfLabelOffset}px`,
    transform: "translate(-50%, 0)",
  };
  const nebulaStyle: CSSProperties = {
    transformOrigin: focalPoint,
    transform: `translate(${pan.x * 0.06}px, ${pan.y * 0.06}px) scale(${1 + (zoom - 1) * 0.04})`,
  };
  const selfAccentStyle: CSSProperties = {
    left: `${selfScreenX}px`,
    top: `${selfScreenY}px`,
    transform: `translate(-50%, -50%) scale(${selfAccentScale})`,
  };
  const showDebugControls = import.meta.env.DEV;
  const hoveredNodeSecondaryLabel =
    hoveredNode && hoveredNode.label !== hoveredNode.id ? hoveredNode.id : null;
  const hoveredNodeShowsModel =
    hoveredNode != null && !(hoveredNode.role === "Client" && hoveredNode.modelLabel === "API-only");
  const addDebugNode = () => {
    debugNodeCounterRef.current += 1;
    setDebugNodes((current) => [
      ...current,
      createDebugNode(debugNodeCounterRef.current, selectedModel, status.model_name ?? ""),
    ]);
  };

  const removeDebugNode = () => {
    setDebugNodes((current) => current.slice(0, -1));
  };

  const triggerRandomTwinkle = () => {
    const candidates = displayNodes.filter((node) => !node.self);
    if (candidates.length === 0) return;
    const randomIndex = Math.floor(Math.random() * candidates.length);
    const node = candidates[randomIndex];
    twinkleAnimationRef.current.set(node.id, { startedAt: performance.now() });
  };

  return (
    <div
      ref={hostRef}
      className={cn(
        "relative overflow-hidden rounded-[20px] border",
        scene.shellClassName,
        heightClass ?? "h-[360px] md:h-[420px] lg:h-[460px] xl:h-[520px]",
        onOpenNode ? "cursor-crosshair" : undefined,
      )}
      style={containerStyle}
      onPointerMove={handlePointerMove}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerLeave}
      onWheel={handleWheel}
    >
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: scene.surfaceGradient(focalPoint),
        }}
      />
      <div
        className={cn(
          "pointer-events-none absolute inset-[-8%]",
          scene.nebulaBlendClassName,
        )}
        style={{
          ...nebulaStyle,
          opacity: scene.nebulaOpacity,
          background: scene.nebulaGradient,
          filter: "blur(16px)",
        }}
      />
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: `
            linear-gradient(${scene.gridLineColor} 1px, transparent 1px),
            linear-gradient(90deg, ${scene.gridLineColor} 1px, transparent 1px)
          `,
          opacity: scene.gridOpacity,
          backgroundPosition: `${pan.x}px ${pan.y}px`,
          backgroundSize: `${gridSize}px ${gridSize}px`,
          maskImage: "radial-gradient(circle at center, black 20%, transparent 82%)",
        }}
      />
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
      <SelfNodeAccentComponent
        style={selfAccentStyle}
        selectedModelMatch={Boolean(selfRenderNode?.selectedModelMatch)}
      />
      <div className="absolute bottom-4 right-4 flex flex-col gap-2">
        <button
          type="button"
          aria-label="Zoom in"
          title="Zoom in"
          className={cn(
            "flex h-9 w-9 items-center justify-center rounded-full border backdrop-blur",
            scene.chromeClassName,
          )}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation();
            const nextZoom = clamp(zoomRef.current * 1.12, 0.7, 2.4);
            zoomAroundPoint(nextZoom, hostSize.width * 0.5, hostSize.height * 0.5);
          }}
        >
          <Plus className="h-4 w-4" />
        </button>
        <button
          type="button"
          aria-label="Zoom out"
          title="Zoom out"
          className={cn(
            "flex h-9 w-9 items-center justify-center rounded-full border backdrop-blur",
            scene.chromeClassName,
          )}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation();
            const nextZoom = clamp(zoomRef.current * 0.88, 0.7, 2.4);
            zoomAroundPoint(nextZoom, hostSize.width * 0.5, hostSize.height * 0.5);
          }}
        >
          <Minus className="h-4 w-4" />
        </button>
        <button
          type="button"
          aria-label="Reset view"
          title="Reset view"
          className={cn(
            "flex h-9 w-9 items-center justify-center rounded-full border backdrop-blur",
            scene.chromeClassName,
          )}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation();
            resetView();
          }}
        >
          <RotateCcw className="h-4 w-4" />
        </button>
      </div>
      {showDebugControls ? (
        <div className="absolute left-4 top-4 flex gap-2">
          <button
            type="button"
            className={cn(
              "rounded-full border px-3 py-1.5 text-[11px] font-medium uppercase tracking-[0.16em] backdrop-blur",
              scene.chromeClassName,
            )}
            onPointerDown={(event) => event.stopPropagation()}
            onClick={(event) => {
              event.stopPropagation();
              addDebugNode();
            }}
          >
            Add node
          </button>
          <button
            type="button"
            className={cn(
              "rounded-full border px-3 py-1.5 text-[11px] font-medium uppercase tracking-[0.16em] backdrop-blur disabled:cursor-not-allowed disabled:opacity-45",
              scene.chromeClassName,
            )}
            disabled={displayNodes.length <= 1}
            onPointerDown={(event) => event.stopPropagation()}
            onClick={(event) => {
              event.stopPropagation();
              triggerRandomTwinkle();
            }}
          >
            Twinkle
          </button>
          <button
            type="button"
            className={cn(
              "rounded-full border px-3 py-1.5 text-[11px] font-medium uppercase tracking-[0.16em] backdrop-blur disabled:cursor-not-allowed disabled:opacity-45",
              scene.chromeClassName,
            )}
            disabled={debugNodes.length === 0}
            onPointerDown={(event) => event.stopPropagation()}
            onClick={(event) => {
              event.stopPropagation();
              removeDebugNode();
            }}
          >
            Remove node
          </button>
          <div
            className={cn(
              "flex items-center gap-2 rounded-full border px-3 py-1.5 backdrop-blur",
              scene.chromeClassName,
            )}
            onPointerDown={(event) => event.stopPropagation()}
          >
            <span className="text-[11px] font-medium uppercase tracking-[0.16em] whitespace-nowrap">
              Hop {hopDelayMs}ms
            </span>
            <input
              type="range"
              min={0}
              max={200}
              step={10}
              value={hopDelayMs}
              onChange={(event) => setHopDelayMs(Number(event.target.value))}
              className="h-1 w-20 cursor-pointer accent-current"
            />
          </div>
        </div>
      ) : null}

      {selectedModel && selectedModel !== "auto" ? (
        <div
          className={cn(
            "pointer-events-none absolute right-5 top-4 rounded-full border px-2.5 py-1 text-[10px] font-medium uppercase tracking-[0.22em]",
            scene.selectedModelClassName,
          )}
        >
          {shortName(selectedModel)}
        </div>
      ) : null}
      <div
        className={cn(
          "pointer-events-none absolute text-center transition-opacity duration-200 ease-out",
          isSelfNodeTooltipActive ? "opacity-[0.32]" : "opacity-100",
        )}
        style={selfLabelStyle}
      >
        <div
          className={cn(
            "text-[11px] font-medium uppercase tracking-[0.28em]",
            scene.selfLabelPrimaryClassName,
          )}
          style={{ textShadow: scene.selfLabelTextShadow }}
        >
          {selfNode?.hostname || "you / host"}
        </div>
        <div
          className={cn(
            "mt-1 text-[10px] uppercase tracking-[0.22em]",
            scene.selfLabelSecondaryClassName,
          )}
          style={{ textShadow: scene.selfLabelTextShadow }}
        >
          {selfNode ? formatLiveNodeState(selfNode.state) : "connected"}
        </div>
      </div>
      {hoveredNode ? (
        <div
          ref={tooltipRef}
          className={cn(
            "pointer-events-none absolute w-[14rem] rounded-xl border px-3 py-3 shadow-2xl backdrop-blur",
            scene.tooltipClassName,
          )}
          style={tooltipStyle ?? { opacity: 0 }}
        >
          <div className="min-w-0">
            <div className="text-sm font-semibold leading-tight tracking-tight [overflow-wrap:anywhere]">
              {hoveredNode.label}
            </div>
            {hoveredNodeSecondaryLabel ? (
              <div
                className={cn(
                  "mt-1 truncate text-[11px] leading-snug",
                  scene.tooltipSubtleTextClassName,
                )}
                title={hoveredNodeSecondaryLabel}
              >
                {hoveredNodeSecondaryLabel}
              </div>
            ) : null}
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5 text-[10px] font-medium">
            <span
              className={cn(
                "inline-flex items-center rounded-full border bg-black/[0.04] px-2 py-0.5 dark:bg-white/[0.04]",
                scene.tooltipDividerClassName,
              )}
            >
              {hoveredNode.role}
            </span>

          </div>
          <div className={cn("mt-2.5 space-y-2.5 border-t pt-2.5", scene.tooltipDividerClassName)}>
            {hoveredNodeShowsModel ? (
              <div className="flex items-start gap-2">
                <Sparkles
                  className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", scene.tooltipMutedTextClassName)}
                />
                <div className="min-w-0">
                  <div
                    className={cn(
                      "text-[10px] font-semibold uppercase tracking-[0.18em]",
                      scene.tooltipMutedTextClassName,
                    )}
                  >
                    Model
                  </div>
                  <div
                    className="text-[11px] font-medium leading-snug [overflow-wrap:anywhere]"
                    title={hoveredNode.modelLabel}
                  >
                    {hoveredNode.modelLabel}
                  </div>
                </div>
              </div>
            ) : null}
            <div className="grid grid-cols-2 gap-x-3 gap-y-2.5">
              <div className="flex min-w-0 items-start gap-2">
                <Wifi
                  className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", scene.tooltipMutedTextClassName)}
                />
                <div className="min-w-0">
                  <div
                    className={cn(
                      "text-[10px] font-semibold uppercase tracking-[0.18em]",
                      scene.tooltipMutedTextClassName,
                    )}
                  >
                    Latency
                  </div>
                  <div className="text-[11px] font-medium leading-snug">{hoveredNode.latencyLabel}</div>
                </div>
              </div>
              <div className="flex min-w-0 items-start gap-2">
                <Clock3
                  className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", scene.tooltipMutedTextClassName)}
                />
                <div className="min-w-0">
                  <div
                    className={cn(
                      "text-[10px] font-semibold uppercase tracking-[0.18em]",
                      scene.tooltipMutedTextClassName,
                    )}
                  >
                    Age
                  </div>
                  <div className="text-[11px] font-medium leading-snug">
                    {formatShortDuration(hoveredNode.ageSeconds)}
                  </div>
                </div>
              </div>
              <div className="flex min-w-0 items-start gap-2">
                <Cpu
                  className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", scene.tooltipMutedTextClassName)}
                />
                <div className="min-w-0">
                  <div
                    className={cn(
                      "text-[10px] font-semibold uppercase tracking-[0.18em]",
                      scene.tooltipMutedTextClassName,
                    )}
                  >
                    Compute
                  </div>
                  <div className="text-[11px] font-medium leading-snug">{hoveredNode.gpuLabel}</div>
                </div>
              </div>
              <div className="flex min-w-0 items-start gap-2">
                <MemoryStick
                  className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", scene.tooltipMutedTextClassName)}
                />
                <div className="min-w-0">
                  <div
                    className={cn(
                      "text-[10px] font-semibold uppercase tracking-[0.18em]",
                      scene.tooltipMutedTextClassName,
                    )}
                  >
                    VRAM
                  </div>
                  <div className="text-[11px] font-medium leading-snug">{hoveredNode.vramLabel}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
