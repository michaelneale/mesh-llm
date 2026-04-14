import { useMemo } from "react";

import {
  ArrowLeft,
  Cpu,
  Gauge,
  Gpu,
  Info,
  MemoryStick,
  Server,
  Shield,
  Sparkles,
  Wifi,
} from "lucide-react";

import { Badge } from "../../../../components/ui/badge";
import { Button } from "../../../../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../../../../components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../../../../components/ui/table";
import {
  formatGpuMemory,
  modelDisplayName,
  modelStatusTooltip,
  ownershipStatusLabel,
  shortName,
  topologyStatusTone,
  topologyStatusTooltip,
  trimGpuVendor,
  uniqueModels,
} from "../../../app-shell/lib/status-helpers";
import type {
  MeshModel,
  Ownership,
} from "../../../app-shell/lib/status-types";
import {
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "../../../../components/ui/sheet";

import { ModelFactCard } from "./ModelFactCard";
import { ModelMetaItem } from "./ModelMetaItem";
import { StatusPill } from "./StatusPill";

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

export function NodeSidebar({
  node,
  meshModelByName,
  onOpenModel,
  onBack,
}: {
  node: NodeSidebarRecord;
  meshModelByName: Record<string, MeshModel>;
  onOpenModel: (modelName: string) => void;
  onBack?: () => void;
}) {
  const modelRows = useMemo(() => {
    const order = new Map<string, number>();
    node.hotModels.forEach((name, index) => {
      order.set(name, index);
    });
    node.servingModels.forEach((name, index) => {
      if (!order.has(name)) order.set(name, node.hotModels.length + index);
    });
    node.requestedModels.forEach((name, index) => {
      if (!order.has(name)) {
        order.set(name, node.hotModels.length + node.servingModels.length + index);
      }
    });
    return uniqueModels(node.hotModels, node.servingModels, node.requestedModels)
      .map((name) => ({
        name,
        flags: [
          node.servingModels.includes(name) ? "Serving" : null,
          node.hostedModels.includes(name) ? "Hosted" : null,
          node.requestedModels.includes(name) ? "Requested" : null,
        ].filter(Boolean) as string[],
        meshStatus: meshModelByName[name]?.status ?? "unknown",
      }))
      .sort(
        (a, b) =>
          (order.get(a.name) ?? Number.MAX_SAFE_INTEGER) -
          (order.get(b.name) ?? Number.MAX_SAFE_INTEGER),
      );
  }, [meshModelByName, node]);

  return (
    <div className="flex min-h-full flex-col">
      <div className="border-b bg-gradient-to-br from-emerald-50 via-background to-background px-6 pb-3 pt-3 dark:from-emerald-950/20">
        <SheetHeader className="space-y-2 text-left">
          <div className="flex items-start gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border bg-background text-primary shadow-sm">
              <Server className="h-3.5 w-3.5" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <SheetTitle className="text-lg font-semibold leading-tight tracking-tight [overflow-wrap:anywhere] sm:text-xl">
                  {node.title}
                </SheetTitle>
                {node.self ? (
                  <Badge className="h-5 rounded-full border-sky-500/45 bg-sky-500/10 px-2 text-[10px] font-medium text-sky-700 dark:border-sky-400/55 dark:bg-sky-400/15 dark:text-sky-200">
                    You
                  </Badge>
                ) : null}
                <StatusPill
                  label={node.role}
                  tone={nodeRoleTone(node.role)}
                  tooltip={nodeRoleTooltip(node.role)}
                />
                <StatusPill
                  label={node.statusLabel}
                  tone={topologyStatusTone(node.statusLabel)}
                  dot
                  tooltip={topologyStatusTooltip(node.statusLabel)}
                />
              </div>
              <SheetDescription className="mt-1.5 text-sm text-muted-foreground [overflow-wrap:anywhere]">
                {node.id}
              </SheetDescription>
            </div>
            {onBack ? (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-8 gap-1.5"
                onClick={onBack}
              >
                <ArrowLeft className="h-3.5 w-3.5" />
                Back
              </Button>
            ) : null}
          </div>
        </SheetHeader>
      </div>

      <div className="flex-1 space-y-5 px-6 py-5">
        <div className="grid gap-3 sm:grid-cols-4">
          <ModelFactCard
            title="Latency"
            value={node.latencyLabel}
            icon={<Wifi className="h-4 w-4" />}
            tooltip={nodeLatencyTooltip(node.self)}
          />
          <ModelFactCard
            title="Node VRAM"
            value={`${node.vramGb.toFixed(1)} GB`}
            icon={<MemoryStick className="h-4 w-4" />}
            tooltip={nodeVramTooltip(node.role)}
          />
          <ModelFactCard
            title="Mesh Share"
            value={node.vramSharePct != null ? `${node.vramSharePct}%` : "n/a"}
            icon={<Gauge className="h-4 w-4" />}
            tooltip={nodeMeshShareTooltip(node.role)}
          />
          <ModelFactCard
            title="Models"
            value={`${modelRows.length}`}
            icon={<Sparkles className="h-4 w-4" />}
            tooltip={nodeHotModelsTooltip()}
          />
        </div>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Sparkles className="h-4 w-4 text-muted-foreground" />
              <span>Models</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            {modelRows.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Model</TableHead>
                    <TableHead>Role</TableHead>
                    <TableHead className="text-right">Mesh</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {modelRows.map((row) => {
                    const modelExists = !!meshModelByName[row.name];
                    return (
                      <TableRow key={row.name}>
                        <TableCell className="max-w-[260px]">
                          {modelExists ? (
                            <button
                              type="button"
                              className="truncate rounded-sm text-left text-sm font-medium underline-offset-4 hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                              onClick={() => onOpenModel(row.name)}
                              title={row.name}
                            >
                              {shortName(modelDisplayName(meshModelByName[row.name]) || row.name)}
                            </button>
                          ) : (
                            <span className="truncate text-sm font-medium" title={row.name}>
                              {shortName(row.name)}
                            </span>
                          )}
                        </TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {row.flags.map((flag) => (
                              <StatusPill
                                key={flag}
                                label={flag}
                                tone={flag === "Serving" ? "good" : flag === "Requested" ? "warn" : "info"}
                                tooltip={nodeModelFlagTooltip(flag)}
                              />
                            ))}
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          <StatusPill
                            className="justify-end"
                            label={
                              row.meshStatus === "warm"
                                ? "Warm"
                                : row.meshStatus === "cold"
                                  ? "Cold"
                                  : row.meshStatus
                            }
                            tone={
                              row.meshStatus === "warm"
                                ? "warm"
                                : row.meshStatus === "cold"
                                  ? "cold"
                                  : "neutral"
                            }
                            dot={row.meshStatus === "warm" || row.meshStatus === "cold"}
                            tooltip={modelStatusTooltip(row.meshStatus)}
                          />
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            ) : (
              <div className="text-sm text-muted-foreground">
                No active model assignments on this node.
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Cpu className="h-4 w-4 text-muted-foreground" />
              <span>Hardware</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 pt-0">
            {node.hostname ? (
              <ModelMetaItem
                label="Hostname"
                value={node.hostname}
                icon={<Server className="h-3.5 w-3.5" />}
              />
            ) : null}
            <ModelMetaItem
              label="Version"
              value={node.version ? `v${node.version}` : "unknown"}
              icon={<Info className="h-3.5 w-3.5" />}
            />
            {node.gpus.length > 0 ? (
              <div className="grid gap-3">
                {node.gpus.map((gpu, index) => (
                  <ModelMetaItem
                    key={`${node.id}-${gpu.name}-${gpu.vram_bytes}-${gpu.bandwidth_gbps ?? "unknown"}`}
                    label={node.isSoc ? `SoC ${index + 1}` : `GPU ${index + 1}`}
                    value={`${trimGpuVendor(gpu.name) || gpu.name} · ${formatGpuMemory(gpu.vram_bytes)}${gpu.bandwidth_gbps ? ` · ${gpu.bandwidth_gbps.toFixed(0)} GB/s` : ""}`}
                    icon={node.isSoc ? <Cpu className="h-3.5 w-3.5" /> : <Gpu className="h-3.5 w-3.5" />}
                  />
                ))}
              </div>
            ) : node.privacyLimited ? (
              <p className="text-sm leading-6 text-muted-foreground">
                Hardware details are hidden by this peer&apos;s privacy settings.
              </p>
            ) : (
              <p className="text-sm leading-6 text-muted-foreground">
                No hardware details reported for this node.
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Shield className="h-4 w-4 text-muted-foreground" />
              <span>Ownership</span>
            </CardTitle>
            <p className="text-sm leading-6 text-muted-foreground">
              Shows whether this node identity is cryptographically bound to a stable owner.
            </p>
          </CardHeader>
          <CardContent className="grid gap-3 pt-0 sm:grid-cols-2">
            <ModelMetaItem
              label="Ownership"
              value={ownershipStatusLabel(node.owner.status)}
            />
            <ModelMetaItem label="Node" value={node.id} copyValue={node.id} />
            <ModelMetaItem
              label="Owner"
              value={node.owner.owner_id ?? "Unsigned"}
              copyValue={node.owner.owner_id}
            />
            {node.owner.node_label ? (
              <ModelMetaItem label="Node Label" value={node.owner.node_label} />
            ) : null}
            {node.owner.hostname_hint ? (
              <ModelMetaItem label="Hostname Hint" value={node.owner.hostname_hint} />
            ) : null}
          </CardContent>
        </Card>

        {node.self ? (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Info className="h-4 w-4 text-muted-foreground" />
                <span>Runtime</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-2">
              {node.version ? (
                <ModelMetaItem label="Version" value={`v${node.version}`} />
              ) : null}
              {node.latestVersion ? (
                <ModelMetaItem label="Latest" value={`v${node.latestVersion}`} />
              ) : null}
              {node.llamaReady != null ? (
                <ModelMetaItem label="Llama Ready" value={node.llamaReady ? "Yes" : "No"} />
              ) : null}
              {node.apiPort != null ? (
                <ModelMetaItem label="API Port" value={`${node.apiPort}`} />
              ) : null}
              {node.inflightRequests != null ? (
                <ModelMetaItem label="Inflight" value={`${node.inflightRequests}`} />
              ) : null}
            </CardContent>
          </Card>
        ) : null}

        {node.availableModels.length > 0 && modelRows.length === 0 ? (
          <div className="px-1 text-xs text-muted-foreground">
            Available locally: {node.availableModels.map(shortName).join(", ")}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function nodeRoleTone(role: string): "good" | "info" | "neutral" {
  if (role === "Host") return "good";
  if (role === "Worker" || role === "Client") return "info";
  return "neutral";
}

function nodeRoleTooltip(role: string) {
  if (role === "Host") {
    return "Coordinates requests and mesh routing for this node.";
  }
  if (role === "Worker") {
    return "Contributes VRAM and compute capacity to the mesh.";
  }
  if (role === "Client") {
    return "Sends requests, but does not contribute VRAM.";
  }
  return "Connected to the mesh, but not actively serving a model.";
}

function nodeLatencyTooltip(self: boolean) {
  if (self) {
    return "This is the local node.";
  }
  return "Observed round-trip latency from your node to this peer.";
}

function nodeVramTooltip(role: string) {
  if (role === "Client") {
    return "Clients do not contribute serving VRAM to the mesh.";
  }
  return "Serving VRAM contributed by this node to the mesh.";
}

function nodeMeshShareTooltip(role: string) {
  if (role === "Client") {
    return "Clients do not contribute serving VRAM, so they have no mesh share.";
  }
  return "Approximate share of total serving VRAM contributed by this node.";
}

function nodeHotModelsTooltip() {
  return "Models this node is hosting, serving, or requesting right now.";
}

function nodeModelFlagTooltip(flag: string) {
  if (flag === "Serving") {
    return "This node is actively serving requests for this model.";
  }
  if (flag === "Hosted") {
    return "This node has the model available locally for routing.";
  }
  if (flag === "Requested") {
    return "This model has been requested on this node, but is not active yet.";
  }
  return "Current model role on this node.";
}
