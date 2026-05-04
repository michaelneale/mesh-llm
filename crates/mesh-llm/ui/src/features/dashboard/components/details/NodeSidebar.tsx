import { useMemo } from "react";

import {
  ArrowLeft,
  Activity,
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
  formatLiveNodeState,
  formatGpuMemory,
  modelDisplayName,
  modelRefLabel,
  modelStatusTooltip,
  ownershipStatusLabel,
  topologyStatusTone,
  topologyStatusTooltip,
  trimGpuVendor,
  uniqueModels,
} from "../../../app-shell/lib/status-helpers";
import type {
  LlamaRuntimePayload,
  LlamaRuntimeEndpointStatus,
  LlamaRuntimeMetricItem,
  LlamaRuntimeSlotItem,
  LiveNodeState,
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
  state: LiveNodeState;
  role: string;
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
  llamaRuntime,
  llamaRuntimeLoading = false,
  llamaRuntimeError = null,
  onOpenModel,
  onBack,
}: {
  node: NodeSidebarRecord;
  meshModelByName: Record<string, MeshModel>;
  llamaRuntime?: LlamaRuntimePayload | null;
  llamaRuntimeLoading?: boolean;
  llamaRuntimeError?: string | null;
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
                  label={formatLiveNodeState(node.state)}
                  tone={topologyStatusTone(node.state)}
                  dot
                  tooltip={topologyStatusTooltip(node.state)}
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
                              {modelRefLabel(modelDisplayName(meshModelByName[row.name]) || row.name)}
                            </button>
                          ) : (
                            <span className="truncate text-sm font-medium" title={row.name}>
                              {modelRefLabel(row.name)}
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
              <div className="sm:col-span-2">
                <LlamaRuntimeSummary
                  runtime={llamaRuntime}
                  loading={llamaRuntimeLoading}
                  error={llamaRuntimeError}
                />
              </div>
            </CardContent>
          </Card>
        ) : null}

        {node.availableModels.length > 0 && modelRows.length === 0 ? (
          <div className="px-1 text-xs text-muted-foreground">
            Available locally: {node.availableModels.map(modelRefLabel).join(", ")}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function LlamaRuntimeSummary({
  runtime,
  loading,
  error,
}: {
  runtime?: LlamaRuntimePayload | null;
  loading: boolean;
  error: string | null;
}) {
  const metricItems = runtime?.items?.metrics ?? runtime?.metrics.samples ?? [];
  const slotItems = runtime?.items?.slots ?? legacySlotItems(runtime?.slots.slots ?? []);
  const slotsTotal = runtime?.items?.slots_total ?? slotItems.length;
  const slotsBusy = runtime?.items?.slots_busy ?? slotItems.filter((slot) => slot.is_processing).length;
  const metricsError = runtime?.metrics.error ?? error;
  const slotsError = runtime?.slots.error ?? error;
  const metricsStale = endpointHasStalePayload(runtime?.metrics);
  const slotsStale = endpointHasStalePayload(runtime?.slots);

  return (
    <div className="space-y-3 rounded-lg border bg-muted/20 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
          <Activity className="h-3.5 w-3.5" />
          Llama runtime
        </div>
        <StatusPill
          label={endpointStatusLabel("Metrics", runtime?.metrics.status, loading, metricsStale)}
          tone={endpointStatusTone(runtime?.metrics.status, loading, metricsError)}
          dot
          tooltip={metricsError ?? "Live llama.cpp /metrics snapshot."}
        />
        <StatusPill
          label={endpointStatusLabel("Slots", runtime?.slots.status, loading, slotsStale)}
          tone={endpointStatusTone(runtime?.slots.status, loading, slotsError)}
          dot={runtime?.slots.status === "ready"}
          tooltip={slotsError ?? "Live llama.cpp /slots snapshot."}
        />
        <StatusPill
          label={`${slotsBusy}/${slotsTotal} slots busy`}
          tone={slotsBusy > 0 ? "warm" : "neutral"}
          tooltip="Busy llama.cpp slots out of total reported slots."
        />
      </div>
      {metricsError ? <p className="text-xs text-rose-600 dark:text-rose-300">Metrics: {metricsError}</p> : null}
      {slotsError && slotsError !== metricsError ? (
        <p className="text-xs text-rose-600 dark:text-rose-300">Slots: {slotsError}</p>
      ) : null}
      {runtime?.metrics.last_success_unix_ms ? (
        <p className="text-xs text-muted-foreground">
          {metricsStale ? "Last metrics update" : "Updated"} {formatRuntimeTimestamp(runtime.metrics.last_success_unix_ms)}
        </p>
      ) : null}
      {metricItems.length > 0 ? (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Metric</TableHead>
              <TableHead className="text-right">Value</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {metricItems.map((item) => (
              <TableRow key={metricItemKey(item)}>
                <TableCell className="max-w-[260px] truncate text-xs" title={metricItemTitle(item)}>
                  {formatMetricName(item)}
                </TableCell>
                <TableCell className="text-right font-mono text-xs">
                  {formatMetricValue(item.value)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      ) : (
        <p className="text-sm text-muted-foreground">
          {loading ? "Loading live llama.cpp metrics…" : "No llama.cpp metric samples reported yet."}
        </p>
      )}
      {slotItems.length > 0 ? (
        <LlamaSlotContextSegments slots={slotItems} slotsBusy={slotsBusy} slotsTotal={slotsTotal} />
      ) : null}
    </div>
  );
}

function LlamaSlotContextSegments({
  slots,
  slotsBusy,
  slotsTotal,
}: {
  slots: LlamaRuntimeSlotItem[];
  slotsBusy: number;
  slotsTotal: number;
}) {
  return (
    <div className="space-y-2 rounded-md border bg-background/65 p-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <div>
          <div className="font-medium text-foreground">Slot context map</div>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-500/75 ring-1 ring-emerald-700/20 dark:bg-emerald-400/75" />
            Available
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-amber-400/90 ring-1 ring-amber-700/20 dark:bg-amber-300/85" />
            Active
          </span>
          <span className="font-mono text-[11px] text-foreground">
            {slotsBusy}/{slotsTotal}
          </span>
        </div>
      </div>
      <ul
        aria-label={`Llama slot context map. ${slotsBusy} of ${slotsTotal} slots active.`}
        className="flex min-h-8 list-none gap-px overflow-hidden rounded-md border bg-background/80"
      >
        {slots.map((slot) => {
          const stateLabel = slot.is_processing ? "Active" : "Available";
          const contextLabel = formatSlotContext(slot);
          const title = `${formatSlotIndex(slot)} · ${stateLabel} · context ${contextLabel}`;

          return (
            <li
              key={slot.id ?? slot.index}
              className="min-w-[48px]"
              style={{ flexBasis: 0, flexGrow: slotContextWeight(slot) }}
            >
              <button
                type="button"
                aria-label={title}
                className={`flex h-full min-h-7 w-full items-center justify-center overflow-hidden px-2 font-mono text-[11px] font-semibold tabular-nums transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring ${
                  slot.is_processing
                    ? "bg-amber-400/90 text-amber-950 dark:bg-amber-300/85 dark:text-amber-950"
                    : "bg-emerald-500/80 text-white dark:bg-emerald-400/75 dark:text-emerald-950"
                }`}
              >
                <span className="truncate">
                  {formatSlotIndex(slot)} · {contextLabel}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function slotContextWeight(slot: LlamaRuntimeSlotItem) {
  return typeof slot.n_ctx === "number" && Number.isFinite(slot.n_ctx) && slot.n_ctx > 0 ? slot.n_ctx : 1;
}

function endpointStatusLabel(
  label: "Metrics" | "Slots",
  status: LlamaRuntimeEndpointStatus | undefined,
  loading: boolean,
  stale: boolean,
) {
  if (loading && !status) return `${label} • Loading`;
  if (stale) return `${label} • Stale`;
  if (status === "ready") return `${label} • Live`;
  if (status === "error") return `${label} • Error`;
  if (status === "unavailable") return `${label} • Unavailable`;
  return status ? `${label} • ${status}` : `${label} • Unknown`;
}

function endpointStatusTone(
  status: LlamaRuntimeEndpointStatus | undefined,
  loading: boolean,
  error: string | null,
): "warm" | "cold" | "good" | "info" | "warn" | "bad" | "neutral" {
  if (error || status === "error") return "bad";
  if (status === "ready") return "good";
  if (status === "unavailable") return "warn";
  if (loading) return "info";
  return "neutral";
}

function formatRuntimeTimestamp(timestampMs: number) {
  return new Date(timestampMs).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function endpointHasStalePayload(endpoint?: {
  status?: LlamaRuntimeEndpointStatus;
  last_attempt_unix_ms?: number;
  last_success_unix_ms?: number;
}) {
  return (
    endpoint?.status !== undefined &&
    endpoint.status !== "ready" &&
    typeof endpoint.last_attempt_unix_ms === "number" &&
    typeof endpoint.last_success_unix_ms === "number" &&
    endpoint.last_attempt_unix_ms > endpoint.last_success_unix_ms
  );
}

function legacySlotItems(slots: Array<{ id?: number; id_task?: number; n_ctx?: number; is_processing?: boolean }>) {
  return slots.map((slot, index) => ({
    index,
    id: slot.id,
    id_task: slot.id_task,
    n_ctx: slot.n_ctx,
    is_processing: slot.is_processing ?? false,
  }));
}

function metricItemKey(item: LlamaRuntimeMetricItem) {
  return `${item.name}:${JSON.stringify(item.labels ?? {})}`;
}

function metricItemTitle(item: LlamaRuntimeMetricItem) {
  const labels = Object.entries(item.labels ?? {})
    .map(([key, value]) => `${key}=${value}`)
    .join(", ");
  return labels ? `${item.name} (${labels})` : item.name;
}

function formatMetricName(item: LlamaRuntimeMetricItem) {
  const suffix = Object.values(item.labels ?? {}).filter(Boolean).join(" · ");
  const name = item.name.replace(/^llamacpp:/, "").replace(/^llama_/, "").replace(/_/g, " ");
  return suffix ? `${name} · ${suffix}` : name;
}

function formatMetricValue(value: number) {
  if (!Number.isFinite(value)) return `${value}`;
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function formatSlotContext(slot: LlamaRuntimeSlotItem) {
  if (slot.n_ctx == null) return "n/a";
  return slot.id_task != null ? `${slot.n_ctx} · task ${slot.id_task}` : `${slot.n_ctx}`;
}

function formatSlotIndex(slot: LlamaRuntimeSlotItem) {
  return slot.id != null && slot.id !== slot.index ? `#${slot.index} · id ${slot.id}` : `#${slot.index}`;
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
