import {
  type CSSProperties,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  type NodeTypes,
  type ReactFlowInstance,
} from "@xyflow/react";
import {
  Cpu,
  Gauge,
  Gpu,
  Laptop,
  MemoryStick,
  Network,
  Server,
  Sparkles,
  Wifi,
} from "lucide-react";

import { Badge } from "../../../../components/ui/badge";
import { cn } from "../../../../lib/utils";
import { TOPOLOGY_LAYOUTS } from "../../../../topology/layouts";
import { TOPOLOGY_NODE_WIDTH } from "../../../../topology/layouts/constants";
import { estimateTopologyNodeHeight } from "../../../../topology/layouts/elk";
import {
  formatLatency,
  shortName,
  topologyStatusTone,
  topologyStatusTooltip,
  trimGpuVendor,
} from "../../../app-shell/lib/status-helpers";
import type {
  BucketedTopologyNode,
  PositionedTopologyNode,
  TopologyLayoutMode,
  TopologyNode,
  TopologyNodeInfo,
} from "../../../../topology/layouts/types";
import { EmptyPanel, StatusPill } from "../details";

type TopologyStatusPayload = {
  model_name?: string | null;
};

type TopologyFlowNodeData = {
  node: PositionedTopologyNode;
  info: TopologyNodeInfo;
  selected: boolean;
  sameModelAsCurrent: boolean;
  layoutDirection: "horizontal" | "vertical";
};

type TopologyFlowDiagramNode = Node<
  TopologyFlowNodeData,
  "topologyNode" | "clientDot"
>;

function TopologyFlowNode({ data }: NodeProps<TopologyFlowDiagramNode>) {
  const isCenter = data.node.bucket === "center";
  const isHorizontal = data.layoutDirection === "horizontal";
  const dotClass = isCenter
    ? "bg-primary border-primary"
    : "bg-muted border-border";
  const baseHandleStyle = {
    opacity: 0,
    width: 1,
    height: 1,
    border: 0,
    pointerEvents: "none" as const,
  };

  return (
    <div className="relative w-[246px] pt-2">
      <div className={cn("relative mx-auto h-7 w-7 rounded-full border-2", dotClass)}>
        <Handle
          type="target"
          position={isHorizontal ? Position.Left : Position.Top}
          style={baseHandleStyle}
        />
        <Handle
          type="source"
          position={isHorizontal ? Position.Right : Position.Bottom}
          style={baseHandleStyle}
        />
      </div>
      <div className="mt-1 flex items-center justify-center gap-1 text-[10px] leading-3 text-foreground">
        <span className="break-all">{data.node.id}</span>
        {data.node.self ? (
          <Badge className="h-4 rounded-full border-sky-500/45 bg-sky-500/10 px-1.5 text-[9px] font-medium text-sky-700 dark:border-sky-400/55 dark:bg-sky-400/15 dark:text-sky-200">
            You
          </Badge>
        ) : null}
      </div>

      <div
        className={cn(
          "mt-1 rounded-md border bg-card p-2",
          data.sameModelAsCurrent
            ? "border-emerald-500/60 dark:border-emerald-400/70"
            : data.selected
              ? "border-ring"
              : "border-border/90",
          data.selected ? "ring-1 ring-ring/50" : null,
        )}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            {data.info.loadedModels.length > 1 ? (
              <div className="flex flex-col gap-0.5">
                {data.info.loadedModels.map((model) => (
                  <div
                    key={model}
                    className="inline-flex items-center gap-1 text-[11px] font-medium leading-4"
                  >
                    <Sparkles className="h-3 w-3 shrink-0 text-muted-foreground" />
                    <span className="min-w-0 truncate" title={model}>
                      {shortName(model)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="inline-flex items-center gap-1 text-[11px] font-medium leading-4">
                <Sparkles className="h-3 w-3 shrink-0 text-muted-foreground" />
                <span
                  className="min-w-0 truncate"
                  title={data.info.loadedModels[0] || data.info.loadedModel}
                >
                  {data.info.loadedModel}
                </span>
              </div>
            )}
          </div>
          <StatusPill
            className="text-[9px]"
            label={data.info.statusLabel}
            tone={topologyStatusTone(data.info.statusLabel)}
            tooltip={topologyStatusTooltip(data.info.statusLabel)}
          />
        </div>

        <div className="mt-2 flex flex-wrap gap-1.5 text-[10px] leading-3">
          {data.info.hostname && (
            <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
              <Server className="h-3 w-3 text-muted-foreground" />
              <span className="text-muted-foreground">Host</span>
              <span className="font-medium">{data.info.hostname}</span>
            </div>
          )}
          <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
            <Network className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">Role</span>
            <span className="font-medium">{data.info.role}</span>
          </div>
          {!data.node.self ? (
            <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
              <Wifi className="h-3 w-3 text-muted-foreground" />
              <span className="text-muted-foreground">Latency</span>
              <span className="font-medium">{formatLatency(data.info.latencyMs)}</span>
            </div>
          ) : null}
          <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
            <MemoryStick className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">VRAM</span>
            <span className="font-medium">{data.info.vramGb.toFixed(1)} GB</span>
          </div>
          <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
            <Gauge className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">Share</span>
            <span className="whitespace-nowrap font-medium">{data.info.vramSharePct}%</span>
          </div>
          {data.info.gpus?.map((gpu, index) => {
            const duplicateCount =
              data.info.gpus
                ?.slice(0, index)
                .filter(
                  (candidate) =>
                    candidate.name === gpu.name &&
                    candidate.vram_bytes === gpu.vram_bytes,
                ).length ?? 0;
            const lower = gpu.name.toLowerCase();
            const isNvidia = lower.includes("nvidia") || lower.includes("jetson");
            const isAmd = lower.includes("amd");
            const isIntel = lower.includes("intel");
            const iconColor = isNvidia
              ? "#76b900"
              : isAmd
                ? "#ED1C24"
                : isIntel
                  ? "#0071C5"
                  : undefined;
            const model = trimGpuVendor(gpu.name);
            const vramGb = gpu.vram_bytes / (1024 * 1024 * 1024);
            const GpuIcon = data.info.isSoc ? Cpu : Gpu;

            return (
              <div
                key={`${data.node.id}-${gpu.name}-${gpu.vram_bytes}-${duplicateCount}`}
                className="group/gpu inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1"
              >
                <GpuIcon
                  className="h-3 w-3"
                  style={iconColor ? { color: iconColor } : undefined}
                />
                <span className="text-muted-foreground">
                  {data.info.isSoc ? "SoC" : "GPU"}
                </span>
                <span className="relative inline-flex font-medium">
                  <span className="group-hover/gpu:invisible">{model}</span>
                  <span className="absolute left-0 top-0 whitespace-nowrap opacity-0 transition-opacity duration-200 group-hover/gpu:opacity-100">
                    {`${Math.round(vramGb)} GB`}
                  </span>
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function TopologyClientDot({ data }: NodeProps<TopologyFlowDiagramNode>) {
  const isHorizontal = data.layoutDirection === "horizontal";
  const handleStyle = {
    opacity: 0,
    width: 1,
    height: 1,
    border: 0,
    pointerEvents: "none" as const,
  };

  return (
    <div className="relative w-[120px]">
      <div
        className={cn(
          "relative mx-auto h-5 w-5 rounded-full border-2 bg-muted",
          data.selected ? "border-ring" : "border-border",
        )}
      >
        <Handle
          type="target"
          position={isHorizontal ? Position.Left : Position.Top}
          style={handleStyle}
        />
        <Handle
          type="source"
          position={isHorizontal ? Position.Right : Position.Bottom}
          style={handleStyle}
        />
      </div>
      <div
        className={cn(
          "mt-1 rounded-md border bg-card p-1.5",
          data.selected ? "border-ring ring-1 ring-ring/50" : "border-border/90",
        )}
      >
        <div className="flex items-center justify-between gap-1">
          <div className="flex min-w-0 items-center gap-1">
            <Laptop className="h-3 w-3 shrink-0 text-muted-foreground" />
            <span className="truncate text-[10px] font-medium leading-3">
              {data.node.hostname || data.node.id.slice(0, 8)}
            </span>
          </div>
        </div>
        <div className="mt-1 text-[9px] leading-none text-muted-foreground">Client</div>
      </div>
    </div>
  );
}

const topologyNodeTypes = {
  topologyNode: TopologyFlowNode,
  clientDot: TopologyClientDot,
} as NodeTypes;

function positionedTopologyLayoutsEqual(
  left: PositionedTopologyNode[],
  right: PositionedTopologyNode[],
) {
  if (left === right) return true;
  if (left.length !== right.length) return false;

  for (let index = 0; index < left.length; index += 1) {
    const a = left[index];
    const b = right[index];
    if (
      a.id !== b.id ||
      a.bucket !== b.bucket ||
      Math.abs(a.x - b.x) > 0.5 ||
      Math.abs(a.y - b.y) > 0.5
    ) {
      return false;
    }
  }

  return true;
}

function topologyLayoutSignature(
  nodes: Pick<BucketedTopologyNode, "id" | "bucket" | "width" | "height">[],
  nodeRadius: number,
) {
  return `${nodeRadius}:${nodes
    .map((node) => `${node.id}:${node.bucket}:${node.width}:${node.height}`)
    .join(",")}`;
}

function positionedTopologySignature(nodes: PositionedTopologyNode[]) {
  return nodes
    .map(
      (node) =>
        `${node.id}:${node.bucket}:${Math.round(node.x)}:${Math.round(node.y)}`,
    )
    .join(",");
}

export function MeshTopologyDiagram({
  status,
  nodes,
  selectedModel,
  layoutMode,
  themeMode,
  onOpenNode,
  highlightedNodeId,
  fullscreen = false,
  heightClass,
  containerStyle,
}: {
  status: TopologyStatusPayload | null;
  nodes: TopologyNode[];
  selectedModel: string;
  layoutMode: TopologyLayoutMode;
  themeMode: "light" | "dark" | "auto";
  onOpenNode?: (nodeId: string) => void;
  highlightedNodeId?: string;
  fullscreen?: boolean;
  heightClass?: string;
  containerStyle?: CSSProperties;
}) {
  if (!status) {
    return <EmptyPanel text="No topology data yet." />;
  }
  if (!nodes.length) {
    return <EmptyPanel text="No host or worker nodes visible yet." />;
  }

  return (
    <MeshTopologyFlow
      status={status}
      nodes={nodes}
      selectedModel={selectedModel}
      layoutMode={layoutMode}
      themeMode={themeMode}
      onOpenNode={onOpenNode}
      highlightedNodeId={highlightedNodeId}
      fullscreen={fullscreen}
      heightClass={heightClass}
      containerStyle={containerStyle}
    />
  );
}

function MeshTopologyFlow({
  status,
  nodes,
  selectedModel,
  layoutMode,
  themeMode,
  onOpenNode,
  highlightedNodeId,
  fullscreen,
  heightClass,
  containerStyle,
}: {
  status: TopologyStatusPayload;
  nodes: TopologyNode[];
  selectedModel: string;
  layoutMode: TopologyLayoutMode;
  themeMode: "light" | "dark" | "auto";
  onOpenNode?: (nodeId: string) => void;
  highlightedNodeId?: string;
  fullscreen: boolean;
  heightClass?: string;
  containerStyle?: CSSProperties;
}) {
  const { center, serving, workers, clients, nodeRadius, meshVramGb } = useMemo(() => {
    const center = nodes.find((node) => node.host) || nodes.find((node) => node.self) || nodes[0];
    const others = nodes
      .filter((node) => node.id !== center.id)
      .sort((a, b) => b.vram - a.vram || a.id.localeCompare(b.id));
    const focusModel = selectedModel || status.model_name || "";
    const serving = others.filter(
      (node) => !node.client && !!node.serving && (!focusModel || node.serving === focusModel),
    );
    const servingIds = new Set(serving.map((node) => node.id));
    const clients = others.filter((node) => node.client);
    const workers = others.filter((node) => !node.client && !servingIds.has(node.id));

    const total = nodes.length;
    const nodeRadius =
      total >= 500
        ? 3.6
        : total >= 280
          ? 4.8
          : total >= 160
            ? 6.2
            : total >= 90
              ? 7.4
              : total >= 50
                ? 8.8
                : 10.4;
    const meshVramGb = nodes
      .filter((node) => !node.client)
      .reduce((sum, node) => sum + Math.max(0, node.vram), 0);

    return {
      center,
      serving,
      workers,
      clients,
      nodeRadius,
      meshVramGb,
    };
  }, [nodes, selectedModel, status.model_name]);

  const currentNodeServingModel = useMemo(() => {
    const current = nodes.find((node) => node.self);
    if (!current || current.client || !current.serving || current.serving === "(idle)") {
      return "";
    }
    return current.serving;
  }, [nodes]);

  const [selectedNodeId, setSelectedNodeId] = useState(center.id);

  useEffect(() => {
    setSelectedNodeId((previous) =>
      nodes.some((node) => node.id === previous) ? previous : center.id,
    );
  }, [nodes, center.id]);

  const nodeIdsRef = useRef(new Set(nodes.map((node) => node.id)));
  useEffect(() => {
    nodeIdsRef.current = new Set(nodes.map((node) => node.id));
  }, [nodes]);
  useEffect(() => {
    if (highlightedNodeId && nodeIdsRef.current.has(highlightedNodeId)) {
      setSelectedNodeId(highlightedNodeId);
    }
  }, [highlightedNodeId]);

  const nodeInfoById = useMemo(() => {
    const output = new Map<string, TopologyNodeInfo>();
    for (const node of nodes) {
      const servingModel =
        !node.client && node.serving && node.serving !== "(idle)" ? node.serving : "";
      const servingModels = !node.client
        ? node.servingModels.filter((model) => model && model !== "(idle)")
        : [];
      const role = node.client
        ? "Client"
        : node.host
          ? "Host"
          : servingModel
            ? "Worker"
            : "Idle";
      const vramSharePct =
        !node.client && meshVramGb > 0
          ? Math.round((Math.max(0, node.vram) / meshVramGb) * 100)
          : 0;
      output.set(node.id, {
        role,
        statusLabel: node.statusLabel,
        latencyMs: node.latencyMs ?? null,
        loadedModel: node.client
          ? "n/a"
          : servingModels.length > 0
            ? servingModels.map(shortName).join(", ")
            : servingModel
              ? shortName(servingModel)
              : "idle",
        loadedModels: node.client
          ? []
          : servingModels.length > 0
            ? servingModels
            : servingModel
              ? [servingModel]
              : [],
        vramGb: Math.max(0, node.vram),
        vramSharePct,
        hostname: node.hostname,
        isSoc: node.isSoc,
        gpus: node.gpus,
      });
    }
    return output;
  }, [nodes, meshVramGb]);

  const activeLayout = TOPOLOGY_LAYOUTS[layoutMode];
  const topologyLayoutNodes = useMemo<BucketedTopologyNode[]>(() => {
    const toBucketedNode = (
      node: TopologyNode,
      bucket: BucketedTopologyNode["bucket"],
    ): BucketedTopologyNode => ({
      ...node,
      bucket,
      width: node.client ? 120 : TOPOLOGY_NODE_WIDTH,
      height: node.client ? 64 : estimateTopologyNodeHeight(node, nodeInfoById.get(node.id)),
    });

    return [
      toBucketedNode(center, "center"),
      ...serving.map((node) => toBucketedNode(node, "serving")),
      ...workers.map((node) => toBucketedNode(node, "worker")),
      ...clients.map((node) => toBucketedNode(node, "client")),
    ];
  }, [center, clients, nodeInfoById, serving, workers]);

  const layoutContext = useMemo(
    () => ({
      center,
      serving,
      workers,
      clients,
      nodeRadius,
      nodes: topologyLayoutNodes,
    }),
    [center, clients, nodeRadius, serving, topologyLayoutNodes, workers],
  );
  const layoutInputSignature = useMemo(
    () => topologyLayoutSignature(topologyLayoutNodes, nodeRadius),
    [nodeRadius, topologyLayoutNodes],
  );
  const layoutInputSignatureRef = useRef(layoutInputSignature);
  layoutInputSignatureRef.current = layoutInputSignature;
  const layoutContextRef = useRef(layoutContext);
  layoutContextRef.current = layoutContext;
  const [positioned, setPositioned] = useState<PositionedTopologyNode[]>([]);

  useEffect(() => {
    let cancelled = false;
    const runSignature = layoutInputSignature;
    const nextLayoutContext = layoutContextRef.current;

    const immediateLayout = activeLayout.getImmediateLayout(nextLayoutContext);
    setPositioned((previous) =>
      positionedTopologyLayoutsEqual(previous, immediateLayout) ? previous : immediateLayout,
    );

    void activeLayout
      .getLayout(nextLayoutContext)
      .then((next) => {
        if (!cancelled && layoutInputSignatureRef.current === runSignature) {
          setPositioned((previous) =>
            positionedTopologyLayoutsEqual(previous, next) ? previous : next,
          );
        }
      })
      .catch(() => {
        if (!cancelled && layoutInputSignatureRef.current === runSignature) {
          setPositioned((previous) =>
            positionedTopologyLayoutsEqual(previous, immediateLayout)
              ? previous
              : immediateLayout,
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeLayout, layoutInputSignature]);

  const positionedIdsKey = useMemo(
    () => positioned.map((node) => node.id).sort().join(","),
    [positioned],
  );
  const expectedPositionIdsKey = useMemo(
    () => topologyLayoutNodes.map((node) => node.id).sort().join(","),
    [topologyLayoutNodes],
  );
  const layoutReady = positioned.length > 0 && positionedIdsKey === expectedPositionIdsKey;
  const [resolvedColorMode, setResolvedColorMode] = useState<"light" | "dark">(() =>
    themeMode === "dark" ? "dark" : "light",
  );

  useEffect(() => {
    if (themeMode !== "auto") {
      setResolvedColorMode(themeMode);
      return;
    }
    if (typeof document === "undefined") {
      setResolvedColorMode("light");
      return;
    }

    const root = document.documentElement;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const updateColorMode = () => {
      setResolvedColorMode(root.classList.contains("dark") ? "dark" : "light");
    };

    updateColorMode();
    const observer = new MutationObserver(updateColorMode);
    observer.observe(root, { attributes: true, attributeFilter: ["class"] });
    media.addEventListener("change", updateColorMode);

    return () => {
      observer.disconnect();
      media.removeEventListener("change", updateColorMode);
    };
  }, [themeMode]);

  const layoutFitSignature = useMemo(
    () => `${fullscreen ? "fs" : "std"}:${layoutMode}:${positionedTopologySignature(positioned)}`,
    [fullscreen, layoutMode, positioned],
  );
  const flowContainerRef = useRef<HTMLDivElement | null>(null);
  const flowInstanceRef = useRef<ReactFlowInstance<TopologyFlowDiagramNode, Edge> | null>(null);
  const [flowReady, setFlowReady] = useState(false);
  const [containerReady, setContainerReady] = useState(false);
  const [containerSizeSignature, setContainerSizeSignature] = useState("");
  const fitViewOptions = useMemo(() => ({ padding: 0.12, maxZoom: 1.45 }), []);
  const fitDuration = fullscreen ? 220 : 0;

  useEffect(() => {
    if (
      !flowInstanceRef.current ||
      !flowReady ||
      !layoutReady ||
      !containerReady ||
      !layoutFitSignature ||
      !containerSizeSignature
    ) {
      return;
    }

    const frame = window.requestAnimationFrame(() => {
      flowInstanceRef.current?.fitView({
        ...fitViewOptions,
        duration: fitDuration,
      });
    });

    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [
    containerReady,
    containerSizeSignature,
    fitDuration,
    fitViewOptions,
    flowReady,
    layoutFitSignature,
    layoutReady,
  ]);

  useEffect(() => {
    const container = flowContainerRef.current;
    if (!container) return;

    const update = () => {
      const rect = container.getBoundingClientRect();
      const ready = rect.width > 8 && rect.height > 8;
      const size = ready ? `${Math.round(rect.width)}x${Math.round(rect.height)}` : "";

      setContainerReady((previous) => (previous === ready ? previous : ready));
      setContainerSizeSignature((previous) => (previous === size ? previous : size));
    };

    update();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", update);
      return () => window.removeEventListener("resize", update);
    }

    const observer = new ResizeObserver(update);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (containerReady && layoutReady) return;
    flowInstanceRef.current = null;
    setFlowReady(false);
  }, [containerReady, layoutReady]);

  const flowNodes = useMemo<TopologyFlowDiagramNode[]>(() => {
    const isHorizontal = activeLayout.direction === "horizontal";
    return positioned.map((node) => ({
      id: node.id,
      type: node.client ? "clientDot" : "topologyNode",
      position: { x: node.x, y: node.y },
      origin: [0.5, 0],
      sourcePosition: isHorizontal ? Position.Right : Position.Bottom,
      targetPosition: isHorizontal ? Position.Left : Position.Top,
      data: {
        node,
        layoutDirection: activeLayout.direction,
        info: nodeInfoById.get(node.id) ?? {
          role: "Node",
          statusLabel: "n/a",
          latencyMs: null,
          loadedModel: "idle",
          loadedModels: [],
          vramGb: 0,
          vramSharePct: 0,
        },
        selected: node.id === selectedNodeId,
        sameModelAsCurrent:
          !!currentNodeServingModel && !node.client && node.serving === currentNodeServingModel,
      },
      draggable: false,
      selectable: false,
      connectable: false,
      zIndex: node.id === center.id ? 10 : 1,
    }));
  }, [
    activeLayout.direction,
    positioned,
    nodeInfoById,
    selectedNodeId,
    center.id,
    currentNodeServingModel,
  ]);

  const flowEdges = useMemo<Edge[]>(() => {
    return positioned
      .filter((node) => node.id !== center.id)
      .map((node) => {
        const isClient = node.bucket === "client";
        const stroke = isClient
          ? "rgba(160,160,160,0.15)"
          : node.bucket === "serving"
            ? "rgba(34,197,94,0.35)"
            : "rgba(56,189,248,0.3)";
        return {
          id: `edge-${center.id}-${node.id}`,
          source: center.id,
          target: node.id,
          type: activeLayout.edgeType,
          className: `mesh-edge mesh-edge--${node.bucket}`,
          animated: false,
          pathOptions:
            activeLayout.edgeType === "smoothstep"
              ? { borderRadius: 18, offset: 24 }
              : undefined,
          style: {
            stroke,
            strokeWidth: isClient ? 1 : 2.4,
            strokeDasharray: isClient ? "1 4" : "2 6",
          },
        };
      });
  }, [activeLayout.edgeType, positioned, center.id]);

  return (
    <div
      className={cn(
        "overflow-hidden rounded-lg border",
        heightClass ?? "h-[360px] md:h-[420px] lg:h-[460px] xl:h-[520px]",
      )}
      ref={flowContainerRef}
      style={containerStyle}
    >
      {containerReady && layoutReady ? (
        <ReactFlow<TopologyFlowDiagramNode, Edge>
          className="h-full w-full"
          style={{ width: "100%", height: "100%" }}
          nodes={flowNodes}
          edges={flowEdges}
          nodeTypes={topologyNodeTypes}
          colorMode={resolvedColorMode}
          minZoom={0.2}
          maxZoom={1.6}
          zoomOnScroll={false}
          zoomOnPinch={false}
          panOnScroll={false}
          panOnDrag
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          onInit={(instance) => {
            flowInstanceRef.current = instance;
            setFlowReady(true);
          }}
          onNodeClick={(_, node) => {
            setSelectedNodeId(node.id);
            onOpenNode?.(node.id);
          }}
          proOptions={{ hideAttribution: true }}
        >
          <Background
            variant={BackgroundVariant.Dots}
            gap={18}
            size={1}
            color="hsl(var(--border))"
          />
          <Controls showInteractive={false} position="bottom-right" />
        </ReactFlow>
      ) : (
        <div className="flex h-full w-full items-center justify-center text-sm text-muted-foreground">
          Preparing topology view...
        </div>
      )}
    </div>
  );
}
