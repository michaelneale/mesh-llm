import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { createPortal } from "react-dom";
import {
  AlertTriangle,
  Copy,
  Cpu,
  FolderTree,
  Gauge,
  Hash,
  Info,
  Loader2,
  Maximize2,
  Minimize2,
  Network,
  Shield,
  Sparkles,
} from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "../../../components/ui/alert";
import { Button } from "../../../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../../../components/ui/card";
import { ScrollArea } from "../../../components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../../components/ui/select";
import {
  Sheet,
  SheetContent,
} from "../../../components/ui/sheet";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../../../components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "../../../components/ui/tooltip";
import { cn } from "../../../lib/utils";
import {
  formatLatency,
  localRoutableModels,
  meshGpuVram,
  modelDisplayName,
  overviewVramGb,
  ownershipPrimaryLabel,
  ownershipStatusLabel,
  ownershipTone,
  peerAssignedModels,
  peerPrimaryModel,
  peerRoutableModels,
  peerStatusLabel,
  shortName,
  topologyNodeRole,
  topologyStatusTone,
  topologyStatusTooltip,
  uniqueModels,
} from "../../app-shell/lib/status-helpers";
import type {
  MeshModel,
  Ownership,
  StatusPayload,
  ThemeMode,
} from "../../app-shell/lib/status-types";
import {
  TOPOLOGY_LAYOUT_OPTIONS,
  isTopologyLayoutMode,
} from "../../../topology/layouts";
import type {
  TopologyLayoutMode,
  TopologyNode,
} from "../../../topology/layouts/types";
import {
  DashboardPanelEmpty,
  ModelSidebar,
  NodeSidebar,
  StatusPill,
} from "./details";
import { MeshTopologyDiagram } from "./topology";
import { useDashboardDetailStack } from "../hooks/useDashboardDetailStack";

const DOCS_URL = "https://docs.anarchai.org";

type ActivePeerRow = {
  id: string;
  latencyLabel: string;
  vramLabel: string;
  shareLabel: string;
};

type NodeSidebarRecord = {
  id: string;
  title: string;
  hostname?: string;
  self: boolean;
  role: string;
  statusLabel: string;
  latencyLabel: string;
  vramGb: number;
  vramSharePct: number | null;
  isSoc?: boolean;
  gpus: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
  hostedModels: string[];
  hotModels: string[];
  servingModels: string[];
  requestedModels: string[];
  availableModels: string[];
  version?: string;
  latestVersion?: string | null;
  llamaReady?: boolean;
  apiPort?: number;
  inflightRequests?: number;
  owner: Ownership;
  privacyLimited: boolean;
};

export function DashboardPage({
  status,
  meshModels,
  modelsLoading,
  topologyNodes,
  selectedModel,
  meshModelByName,
  themeMode,
  isPublicMesh,
  inviteToken,
  isLocalhost,
}: {
  status: StatusPayload | null;
  meshModels: MeshModel[];
  modelsLoading: boolean;
  topologyNodes: TopologyNode[];
  selectedModel: string;
  meshModelByName: Record<string, MeshModel>;
  themeMode: ThemeMode;
  isPublicMesh: boolean;
  inviteToken: string;
  isLocalhost: boolean;
}) {
  const [modelFilter, setModelFilter] = useState<"all" | "warm" | "cold">("all");
  const [isMeshOverviewFullscreen, setIsMeshOverviewFullscreen] = useState(false);
  const [selectedTopologyNodeId, setSelectedTopologyNodeId] = useState<string>("");
  const [meshTopologyLayoutMode, setMeshTopologyLayoutMode] =
    useState<TopologyLayoutMode>("elk");
  const {
    activeDetail,
    closeDetailPanel,
    detailPanelStack,
    goBackDetailPanel,
    openModelDetail,
    openNodeDetail: openNodeDetailFromStack,
  } = useDashboardDetailStack({
    isMeshOverviewFullscreen,
    meshModelByName,
  });

  const openNodeDetail = useCallback(
    (nodeId: string) => {
      setSelectedTopologyNodeId(nodeId);
      openNodeDetailFromStack(nodeId);
    },
    [openNodeDetailFromStack],
  );

  const highlightedNodeId =
    activeDetail?.kind === "node" ? activeDetail.nodeId : selectedTopologyNodeId;

  const topologyDiagramNodes = topologyNodes;
  const filteredModels = useMemo(() => {
    const models = meshModels;
    return [...models]
      .filter((model) => (modelFilter === "all" ? true : model.status === modelFilter))
      .sort((a, b) => b.node_count - a.node_count || a.name.localeCompare(b.name));
  }, [meshModels, modelFilter]);
  const totalMeshVramGb = useMemo(() => meshGpuVram(status), [status]);
  const distinctMeshVersions = useMemo(() => {
    const versions = new Set<string>();
    if (status?.version) versions.add(status.version);
    status?.peers?.forEach((peer) => {
      if (peer.version) versions.add(peer.version);
    });
    return versions;
  }, [status]);
  const sortedPeers = useMemo(() => {
    return [...(status?.peers ?? [])].sort((a, b) => {
      const bOverviewVramGb = overviewVramGb(b.role === "Client", b.vram_gb);
      const aOverviewVramGb = overviewVramGb(a.role === "Client", a.vram_gb);
      return bOverviewVramGb - aOverviewVramGb || a.id.localeCompare(b.id);
    });
  }, [status?.peers]);
  const peerRows = useMemo(() => {
    return sortedPeers.map((peer) => {
      const statusLabel = peer.role === "Client" ? "Client" : peerStatusLabel(peer);
      const primaryModel = peerPrimaryModel(peer);
      const modelLabel =
        primaryModel && primaryModel !== "(idle)" ? shortName(primaryModel) : "idle";
      const latencyLabel = formatLatency(peer.rtt_ms);
      const displayVramGb = overviewVramGb(peer.role === "Client", peer.vram_gb);
      const sharePct =
        peer.role !== "Client" && totalMeshVramGb > 0
          ? Math.round((displayVramGb / totalMeshVramGb) * 100)
          : null;
      return {
        ...peer,
        displayVramGb,
        statusLabel,
        modelLabel,
        latencyLabel,
        shareLabel: sharePct == null ? "n/a" : `${sharePct}%`,
      };
    });
  }, [sortedPeers, totalMeshVramGb]);
  const activePeerRows = useMemo(() => {
    const activeModelName = activeDetail?.kind === "model" ? activeDetail.modelName : null;
    const selectedCatalogModel = activeModelName
      ? meshModelByName[activeModelName] ?? null
      : null;
    if (!selectedCatalogModel || selectedCatalogModel.status !== "warm" || !status) {
      return [] as ActivePeerRow[];
    }
    const targetModel = selectedCatalogModel.name;
    const totalModelVram = selectedCatalogModel.mesh_vram_gb ?? 0;
    const rows: ActivePeerRow[] = [];
    const localServing = localRoutableModels(status).includes(targetModel);
    if (localServing && !status.is_client) {
      const localVram = overviewVramGb(status.is_client, status.my_vram_gb);
      rows.push({
        id: status.node_id,
        latencyLabel: "local",
        vramLabel: `${localVram.toFixed(1)} GB`,
        shareLabel:
          totalModelVram > 0
            ? `${Math.round((localVram / totalModelVram) * 100)}%`
            : "n/a",
      });
    }
    for (const peer of peerRows) {
      const servesTarget =
        peerRoutableModels(peer).includes(targetModel) ||
        peerAssignedModels(peer).includes(targetModel);
      if (!servesTarget || peer.role === "Client") continue;
      rows.push({
        id: peer.id,
        latencyLabel: peer.latencyLabel,
        vramLabel: `${peer.displayVramGb.toFixed(1)} GB`,
        shareLabel:
          totalModelVram > 0
            ? `${Math.round((peer.displayVramGb / totalModelVram) * 100)}%`
            : "n/a",
      });
    }
    return rows;
  }, [activeDetail, meshModelByName, peerRows, status]);

  const activeModel =
    activeDetail?.kind === "model" ? meshModelByName[activeDetail.modelName] ?? null : null;
  const activeNode = useMemo(() => {
    if (!status || activeDetail?.kind !== "node") return null;
    const topologyNode = topologyNodes.find((node) => node.id === activeDetail.nodeId);
    if (!topologyNode) return null;
    const peer = topologyNode.self
      ? null
      : status.peers.find((candidate) => candidate.id === topologyNode.id);
    const hostedModels = topologyNode.self
      ? uniqueModels(localRoutableModels(status))
      : uniqueModels(peer ? peerRoutableModels(peer) : []);
    const servingModels = topologyNode.self
      ? uniqueModels(status.serving_models)
      : uniqueModels(peer ? peerAssignedModels(peer) : []);
    const requestedModels = topologyNode.self
      ? uniqueModels(status.requested_models)
      : uniqueModels(peer?.requested_models);
    return {
      id: topologyNode.id,
      title: topologyNode.hostname || topologyNode.id,
      hostname: topologyNode.hostname,
      self: topologyNode.self,
      role: topologyNodeRole(topologyNode),
      statusLabel: topologyNode.statusLabel,
      latencyLabel: topologyNode.self ? "local" : formatLatency(topologyNode.latencyMs),
      vramGb: Math.max(0, topologyNode.vram),
      vramSharePct: topologyNode.client
        ? null
        : totalMeshVramGb <= 0
          ? 0
          : Math.round((Math.max(0, topologyNode.vram) / totalMeshVramGb) * 100),
      isSoc: topologyNode.isSoc,
      gpus: topologyNode.gpus ?? [],
      hostedModels,
      hotModels: uniqueModels(hostedModels, servingModels, requestedModels),
      servingModels,
      requestedModels,
      availableModels: topologyNode.self
        ? uniqueModels(status.available_models)
        : uniqueModels(peer?.available_models),
      version: topologyNode.self ? status.version : peer?.version,
      latestVersion: topologyNode.self ? status.latest_version : undefined,
      llamaReady: topologyNode.self ? status.llama_ready : undefined,
      apiPort: topologyNode.self ? status.api_port : undefined,
      inflightRequests: topologyNode.self ? status.inflight_requests : undefined,
      owner: topologyNode.self
        ? status.owner ?? { status: "unsigned", verified: false }
        : peer?.owner ?? { status: "unsigned", verified: false },
      privacyLimited:
        !topologyNode.self && !topologyNode.hostname && !(topologyNode.gpus?.length ?? 0),
    } satisfies NodeSidebarRecord;
  }, [activeDetail, status, topologyNodes, totalMeshVramGb]);

  useEffect(() => {
    const prevOverflow = document.body.style.overflow;
    if (isMeshOverviewFullscreen) document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prevOverflow;
    };
  }, [isMeshOverviewFullscreen]);

  useEffect(() => {
    if (!isMeshOverviewFullscreen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setIsMeshOverviewFullscreen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isMeshOverviewFullscreen]);

  function toggleMeshOverviewFullscreen() {
    setIsMeshOverviewFullscreen((prev) => !prev);
  }

  const meshTopologyLayoutControl = (
    <Select
      value={meshTopologyLayoutMode}
      onValueChange={(value) => {
        if (isTopologyLayoutMode(value)) setMeshTopologyLayoutMode(value);
      }}
    >
      <SelectTrigger
        className="h-8 w-[118px] gap-1.5 px-2 text-xs"
        aria-label="Select topology layout"
        title="Select topology layout"
      >
        <FolderTree className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        <SelectValue placeholder="Layout" />
      </SelectTrigger>
      <SelectContent align="end" className={isMeshOverviewFullscreen ? "z-[130]" : undefined}>
        {TOPOLOGY_LAYOUT_OPTIONS.map((option) => (
          <SelectItem key={option.value} value={option.value} className="text-xs">
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );

  const gpuNodeCount = topologyDiagramNodes.filter((node) => !node.client).length;
  const clientCount = topologyDiagramNodes.filter((node) => node.client).length;

  return (
    <div className="space-y-4">
      <Alert className="border-primary/20 bg-primary/5">
        <Network className="h-4 w-4" />
        <AlertTitle className="text-sm font-medium">
          {isPublicMesh ? "Welcome to the public mesh" : "Your private mesh"}
        </AlertTitle>
        <AlertDescription className="text-xs text-muted-foreground">
          {isPublicMesh
            ? "Mesh LLM is a project to let people contribute spare compute, build private personal AI, using open models."
            : "Mesh LLM lets you build private personal AI, using open models. Pool machines across your home, office, or friends, no cloud needed."}{" "}
          <a
            href={DOCS_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-foreground"
          >
            Learn more →
          </a>
          {" · "}
          <a
            href="https://github.com/Mesh-LLM/mesh-llm"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 underline hover:text-foreground"
          >
            GitHub
          </a>
        </AlertDescription>
      </Alert>
      {distinctMeshVersions.size >= 2 && (
        <Alert
          data-testid="mixed-version-banner"
          className="border-amber-500/30 bg-amber-500/5"
        >
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle className="text-sm font-medium">Mesh has mixed versions</AlertTitle>
          <AlertDescription className="text-xs text-muted-foreground">
            Detected {distinctMeshVersions.size} distinct mesh-llm versions: {" "}
            {[...distinctMeshVersions].join(", ")}. Functionality may vary between nodes.
          </AlertDescription>
        </Alert>
      )}
      {(status?.local_instances?.length ?? 0) >= 2 && (
        <Alert
          data-testid="multi-instance-banner"
          className="border-blue-500/30 bg-blue-500/5"
        >
          <Info className="h-4 w-4" />
          <AlertTitle className="text-sm font-medium">
            Multiple mesh-llm instances on this host
          </AlertTitle>
          <AlertDescription className="text-xs text-muted-foreground">
            Detected {status!.local_instances!.length} local mesh-llm processes sharing this
            machine. Each runs in an isolated scope.
          </AlertDescription>
        </Alert>
      )}
      {modelsLoading && meshModels.length === 0 ? (
        <Alert className="border-border/60 bg-card/80">
          <Loader2 className="h-4 w-4 animate-spin" />
          <AlertTitle className="text-sm font-medium">Loading model catalog</AlertTitle>
          <AlertDescription className="space-y-2">
            <div className="text-xs text-muted-foreground">
              Scanning local models and assembling mesh metadata.
            </div>
          </AlertDescription>
        </Alert>
      ) : null}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-6">
        <StatCard
          title="Node ID"
          value={status?.node_id ?? "n/a"}
          valueSuffix={
            <StatusPill
              label={status?.node_status ?? "n/a"}
              tone={topologyStatusTone(status?.node_status ?? "n/a")}
            />
          }
          icon={<Hash className="h-4 w-4" />}
          tooltip={`Current node identifier in this mesh. ${topologyStatusTooltip(status?.node_status ?? "n/a")}`}
        />
        <StatCard
          title="Owner"
          value={ownershipPrimaryLabel(status?.owner)}
          valueSuffix={
            <StatusPill
              label={ownershipStatusLabel(status?.owner?.status)}
              tone={ownershipTone(status?.owner?.status)}
            />
          }
          icon={<Shield className="h-4 w-4" />}
          tooltip="Stable owner identity from the keystore, if this node is attested. Ownership certificate state for this node."
        />
        <StatCard
          title="Active Models"
          value={`${meshModels.filter((model) => model.status === "warm").length}`}
          icon={<Sparkles className="h-4 w-4" />}
          tooltip="Models currently loaded and serving across the mesh."
        />
        <StatCard
          title="Mesh VRAM"
          value={`${meshGpuVram(status).toFixed(1)} GB`}
          icon={<Cpu className="h-4 w-4" />}
          tooltip="Total GPU VRAM across non-client nodes in the mesh."
        />
        <StatCard
          title="Nodes"
          value={`${gpuNodeCount}`}
          valueSuffix={
            clientCount > 0 ? (
              <span className="text-xs font-normal text-muted-foreground">
                +{clientCount} client{clientCount === 1 ? "" : "s"}
              </span>
            ) : undefined
          }
          icon={<Network className="h-4 w-4" />}
          tooltip="GPU nodes in the mesh, plus connected clients."
        />
        <StatCard
          title="Inflight"
          value={`${status?.inflight_requests ?? 0}`}
          icon={<Gauge className="h-4 w-4" />}
          tooltip="Current in-flight request count."
        />
      </div>

      <div className="grid items-start gap-4 lg:grid-cols-7">
        <div className="lg:col-span-5">
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between gap-2">
                <CardTitle className="text-sm">Mesh Overview</CardTitle>
                <div className="flex items-center gap-2">
                  {meshTopologyLayoutControl}
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5"
                    onClick={() => void toggleMeshOverviewFullscreen()}
                  >
                    <Maximize2 className="h-3.5 w-3.5" />
                    Fullscreen
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              {isMeshOverviewFullscreen ? (
                <div className="flex h-[360px] items-center justify-center rounded-lg border border-dashed bg-muted/20 text-sm text-muted-foreground md:h-[420px] lg:h-[460px] xl:h-[520px]">
                  Mesh Overview is open in fullscreen.
                </div>
              ) : (
                <MeshTopologyDiagram
                  status={status}
                  nodes={topologyDiagramNodes}
                  selectedModel={selectedModel}
                  layoutMode={meshTopologyLayoutMode}
                  themeMode={themeMode}
                  onOpenNode={openNodeDetail}
                  highlightedNodeId={highlightedNodeId}
                  fullscreen={false}
                  heightClass="h-[360px] md:h-[420px] lg:h-[460px] xl:h-[520px]"
                />
              )}
            </CardContent>
          </Card>
        </div>

        <Card className="lg:col-span-2">
          <CardHeader className="pb-2">
            <div className="flex flex-wrap items-center gap-2">
              <CardTitle className="text-sm">Model Catalog</CardTitle>
              <div className="ml-auto flex shrink-0 items-center gap-2">
                <span className="text-xs text-muted-foreground">Filter</span>
                <Select
                  value={modelFilter}
                  onValueChange={(value) => setModelFilter(value as "all" | "warm" | "cold")}
                >
                  <SelectTrigger className="h-8 w-[110px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    <SelectItem value="warm">Warm</SelectItem>
                    <SelectItem value="cold">Cold</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            {filteredModels.length > 0 ? (
              <div className="h-[360px] overflow-y-auto pr-2 md:h-[420px] lg:h-[460px] xl:h-[520px]">
                <div className="space-y-2">
                  {filteredModels.map((model) => (
                    <button
                      key={model.name}
                      type="button"
                      onClick={() => openModelDetail(model.name)}
                      className="block w-full rounded-md border p-3 text-left transition-colors hover:border-primary/35 hover:bg-muted/30"
                    >
                      <div className="flex flex-col items-start gap-2 sm:flex-row sm:items-start">
                        <div className="flex h-7 w-7 items-center justify-center rounded-md border bg-muted/40 text-muted-foreground">
                          <Sparkles className="h-3.5 w-3.5" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <div className="text-sm font-medium leading-5 [overflow-wrap:anywhere]">
                              {shortName(modelDisplayName(model))}
                            </div>
                            <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
                              {model.vision ? (
                                <span role="img" aria-label="Vision">
                                  👁
                                </span>
                              ) : null}
                              {model.reasoning ? (
                                <span role="img" aria-label="Reasoning">
                                  🧠
                                </span>
                              ) : null}
                              {model.moe ? (
                                <span role="img" aria-label="Mixture of Experts">
                                  🧩
                                </span>
                              ) : null}
                            </div>
                          </div>
                          <div className="text-xs leading-4 text-muted-foreground [overflow-wrap:anywhere]">
                            {model.name}
                          </div>
                        </div>
                        <StatusPill
                          className="self-start"
                          label={
                            model.status === "warm"
                              ? "Warm"
                              : model.status === "cold"
                                ? "Cold"
                                : model.status
                          }
                          tone={
                            model.status === "warm"
                              ? "warm"
                              : model.status === "cold"
                                ? "cold"
                                : "neutral"
                          }
                          dot
                        />
                      </div>
                      <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
                        <span>
                          {model.node_count} node{model.node_count === 1 ? "" : "s"}
                        </span>
                        <span className="flex items-center gap-2">
                          {model.vision && (
                            <span role="img" aria-label="Vision">
                              👁
                            </span>
                          )}
                          {model.size_gb.toFixed(1)} GB
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-[360px] md:h-[420px] lg:h-[460px] xl:h-[520px]">
                <DashboardPanelEmpty
                  icon={<Sparkles className="h-4 w-4" />}
                  title={meshModels.length > 0 ? `No ${modelFilter} models` : "No model catalog data"}
                  description={
                    meshModels.length > 0
                      ? "Try changing the model filter."
                      : modelsLoading
                        ? "The model catalog will appear once the local scan completes."
                        : "Model metadata will appear once the mesh reports available models."
                  }
                />
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Connected Peers</CardTitle>
        </CardHeader>
        <CardContent className="min-h-0 pt-0">
          {peerRows.length > 0 ? (
            <ScrollArea horizontal className="max-h-[18rem] md:max-h-[20rem]">
              <div className="pr-3">
                <Table className="min-w-[920px]">
                  <TableHeader>
                    <TableRow>
                      <TableHead>ID</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead>Version</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Model</TableHead>
                      <TableHead className="text-right">Latency</TableHead>
                      <TableHead className="text-right">VRAM</TableHead>
                      <TableHead className="text-right whitespace-nowrap">Share</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {peerRows.map((peer) => (
                      <TableRow
                        key={peer.id}
                        data-id={peer.id}
                        tabIndex={0}
                        className={cn(
                          "cursor-pointer focus-visible:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                          peer.id === highlightedNodeId && "bg-muted/50 hover:bg-muted/60",
                        )}
                        onClick={() => setSelectedTopologyNodeId(peer.id)}
                        onKeyDown={(event) => {
                          if (event.target !== event.currentTarget) return;
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            setSelectedTopologyNodeId(peer.id);
                          }
                        }}
                      >
                        <TableCell className="font-mono text-xs">
                          <button
                            type="button"
                            className="rounded-sm text-left underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                            onClick={(event) => {
                              event.stopPropagation();
                              setSelectedTopologyNodeId(peer.id);
                              openNodeDetail(peer.id);
                            }}
                          >
                            {peer.id}
                          </button>
                        </TableCell>
                        <TableCell>{peer.role}</TableCell>
                        <TableCell className="font-mono text-xs">
                          {peer.version ?? (
                            <span className="text-muted-foreground">unknown</span>
                          )}
                        </TableCell>
                        <TableCell>{peer.statusLabel}</TableCell>
                        <TableCell className="max-w-[180px] truncate">
                          {peer.modelLabel}
                        </TableCell>
                        <TableCell className="text-right">{peer.latencyLabel}</TableCell>
                        <TableCell className="text-right">
                          {peer.role === "Client" ? "n/a" : `${peer.displayVramGb.toFixed(1)} GB`}
                        </TableCell>
                        <TableCell className="text-right whitespace-nowrap">
                          {peer.shareLabel}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </ScrollArea>
          ) : (
            <DashboardPanelEmpty
              icon={<Network className="h-4 w-4" />}
              title="No peers connected"
              description="Invite another node to join this mesh to see connected peers."
            />
          )}
        </CardContent>
      </Card>

      {isMeshOverviewFullscreen && typeof document !== "undefined"
        ? createPortal(
            <div className="fixed inset-0 z-[120] bg-black/55 backdrop-blur-sm">
              <div className="h-full w-full p-3 md:p-4">
                <Card className="flex h-full min-h-0 w-full flex-col shadow-2xl shadow-black/65">
                  <CardHeader className="shrink-0 pb-2">
                    <div className="flex items-center justify-between gap-2">
                      <CardTitle className="text-sm">Mesh Overview</CardTitle>
                      <div className="flex items-center gap-2">
                        {meshTopologyLayoutControl}
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          className="h-8 gap-1.5"
                          onClick={() => void toggleMeshOverviewFullscreen()}
                        >
                          <Minimize2 className="h-3.5 w-3.5" />
                          Exit Fullscreen
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="flex min-h-0 flex-1 p-0">
                    <MeshTopologyDiagram
                      status={status}
                      nodes={topologyDiagramNodes}
                      selectedModel={selectedModel}
                      layoutMode={meshTopologyLayoutMode}
                      themeMode={themeMode}
                      onOpenNode={openNodeDetail}
                      highlightedNodeId={highlightedNodeId}
                      fullscreen
                      heightClass="min-h-[420px]"
                      containerStyle={{
                        width: "100%",
                        height: "calc(100dvh - 8rem)",
                        minHeight: 420,
                      }}
                    />
                  </CardContent>
                </Card>
              </div>
            </div>,
            document.body,
          )
        : null}

      <Sheet
        open={detailPanelStack.length > 0 && !isMeshOverviewFullscreen}
        onOpenChange={(open) => !open && closeDetailPanel()}
      >
        <SheetContent
          side="right"
          className="w-full overflow-y-auto border-l bg-background/95 p-0 backdrop-blur sm:max-w-2xl"
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            (event.currentTarget as HTMLElement).focus();
          }}
        >
          {activeDetail?.kind === "node" && activeNode ? (
            <NodeSidebar
              node={activeNode}
              meshModelByName={meshModelByName}
              onOpenModel={openModelDetail}
              onBack={detailPanelStack.length > 1 ? goBackDetailPanel : undefined}
            />
          ) : activeDetail?.kind === "node" && !activeNode ? (
            <MissingDetailState
              canGoBack={detailPanelStack.length > 1}
              label="This node is no longer available."
              onAction={detailPanelStack.length > 1 ? goBackDetailPanel : closeDetailPanel}
            />
          ) : null}
          {activeDetail?.kind === "model" && activeModel ? (
            <ModelSidebar
              model={activeModel}
              activePeers={activePeerRows}
              onOpenNode={openNodeDetail}
              onBack={detailPanelStack.length > 1 ? goBackDetailPanel : undefined}
            />
          ) : activeDetail?.kind === "model" && !activeModel ? (
            <MissingDetailState
              canGoBack={detailPanelStack.length > 1}
              label="This model is no longer available."
              onAction={detailPanelStack.length > 1 ? goBackDetailPanel : closeDetailPanel}
            />
          ) : null}
        </SheetContent>
      </Sheet>

      <Card className="mt-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Connect</CardTitle>
          <div className="text-xs text-muted-foreground">
            Run mesh-llm on your machine to get a local OpenAI-compatible API and contribute
            compute to the mesh.
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1.5">
            <div className="text-xs font-medium">1. Install</div>
            <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
              <a
                href="https://docs.anarchai.org/#install"
                target="_blank"
                rel="noopener noreferrer"
                className="min-w-0 flex-1 text-xs text-primary underline hover:text-foreground"
              >
                docs.anarchai.org/#install
              </a>
            </div>
          </div>
          <div className="space-y-1.5">
            <div className="text-xs font-medium">2. Run</div>
            {(() => {
              const cmd = isPublicMesh
                ? "mesh-llm --auto"
                : `mesh-llm --auto --join ${inviteToken || "(token)"}`;
              return (
                <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                  <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                    {cmd}
                  </code>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="h-6 w-6 shrink-0"
                    aria-label="Copy"
                    onClick={() => void navigator.clipboard.writeText(cmd)}
                  >
                    <Copy className="h-3 w-3" />
                  </Button>
                </div>
              );
            })()}
            <div className="text-xs text-muted-foreground">
              This auto-selects a model for your hardware, joins the mesh, and serves a local
              API at <code className="text-[0.7rem]">http://127.0.0.1:9337/v1</code>
            </div>
          </div>
          {isLocalhost ? null : (
            <div className="text-xs text-muted-foreground">
              <a
                href={DOCS_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-foreground"
              >
                Full docs →
              </a>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex items-center justify-center gap-3 py-2 text-xs text-muted-foreground">
        <a
          href={DOCS_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="underline-offset-2 hover:text-foreground hover:underline"
        >
          Docs
        </a>
        <span>·</span>
        <a
          href={`${DOCS_URL}/#agents`}
          target="_blank"
          rel="noopener noreferrer"
          className="underline-offset-2 hover:text-foreground hover:underline"
        >
          Agents
        </a>
        <span>·</span>
        <a
          href={`${DOCS_URL}/#models`}
          target="_blank"
          rel="noopener noreferrer"
          className="underline-offset-2 hover:text-foreground hover:underline"
        >
          Models
        </a>
        <span>·</span>
        <a
          href={`${DOCS_URL}/#running`}
          target="_blank"
          rel="noopener noreferrer"
          className="underline-offset-2 hover:text-foreground hover:underline"
        >
          Common patterns
        </a>
      </div>
    </div>
  );
}

function MissingDetailState({
  canGoBack,
  label,
  onAction,
}: {
  canGoBack: boolean;
  label: string;
  onAction: () => void;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 p-8 text-center">
      <p className="text-sm text-muted-foreground">{label}</p>
      <button
        type="button"
        className="text-xs underline hover:text-foreground"
        onClick={onAction}
      >
        {canGoBack ? "Go back" : "Close"}
      </button>
    </div>
  );
}

function StatCard({
  title,
  value,
  valueSuffix,
  icon,
  tooltip,
}: {
  title: string;
  value: string;
  valueSuffix?: ReactNode;
  icon: ReactNode;
  tooltip?: string;
}) {
  const card = (
    <Card>
      <CardContent className="p-3">
        <div className="mb-2 flex items-center gap-2 text-muted-foreground">
          {icon}
          <span className="text-xs">{title}</span>
        </div>
        <div className="flex min-w-0 items-center gap-2 text-sm font-semibold text-foreground">
          <span className="truncate">{value}</span>
          {valueSuffix}
        </div>
      </CardContent>
    </Card>
  );
  if (!tooltip) return card;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="block w-full rounded-lg bg-transparent text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {card}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" align="center" sideOffset={8}>
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}
