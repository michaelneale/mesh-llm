import {
  ArrowLeft,
  Boxes,
  Brain,
  Braces,
  Check,
  Cpu,
  ExternalLink,
  FolderTree,
  Gauge,
  ImagePlus,
  Info,
  MemoryStick,
  MessageSquarePlus,
  Network,
  Sparkles,
  TextCursorInput,
} from "lucide-react";

import { Button } from "../../../../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../../../../components/ui/card";
import {
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "../../../../components/ui/sheet";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../../../../components/ui/table";
import { formatShortDuration } from "../../../../lib/format-duration";
import { cn } from "../../../../lib/utils";
import { modelStatusTooltip } from "../../../app-shell/lib/status-helpers";
import type { MeshModel } from "../../../app-shell/lib/status-types";

import { CapabilityBadge } from "./CapabilityBadge";
import { ModelFactCard } from "./ModelFactCard";
import { ModelMetaItem } from "./ModelMetaItem";
import { ModelMetaLinkItem } from "./ModelMetaLinkItem";
import { StatusPill } from "./StatusPill";

type ActivePeerRow = {
  id: string;
  latencyLabel: string;
  vramLabel: string;
  shareLabel: string;
};

export function ModelSidebar({
  model,
  activePeers,
  onOpenNode,
  onBack,
}: {
  model: MeshModel;
  activePeers: ActivePeerRow[];
  onOpenNode: (nodeId: string) => void;
  onBack?: () => void;
}) {
  const fullFileName = modelFullFileName(model);
  const revisionFileName = modelRevisionFileName(model);

  return (
    <div className="flex min-h-full flex-col">
      <div className="border-b bg-gradient-to-br from-sky-50 via-background to-background px-6 pb-3 pt-3 dark:from-sky-950/20">
        <SheetHeader className="space-y-2 text-left">
          <div className="flex items-start gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border bg-background text-primary shadow-sm">
              <Sparkles className="h-3.5 w-3.5" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <SheetTitle className="text-lg font-semibold leading-tight tracking-tight [overflow-wrap:anywhere] sm:text-xl">
                  {model.name}
                </SheetTitle>
                <StatusPill
                  label={
                    model.status === "warm"
                      ? "Warm"
                      : model.status === "cold"
                        ? "Cold"
                        : model.status || "Unknown"
                  }
                  tone={
                    model.status === "warm"
                      ? "warm"
                      : model.status === "cold"
                        ? "cold"
                        : "neutral"
                  }
                  dot
                  tooltip={modelStatusTooltip(model.status)}
                />
                <StatusPill
                  label={headerFitLabel(model.fit_label ?? "Unknown")}
                  tone={fitLabelTone(model.fit_label ?? "Unknown")}
                  icon={<Check className="h-3 w-3" />}
                  tooltip={fitLabelTooltip(model.fit_label ?? "Unknown")}
                />
              </div>
              <SheetDescription className="mt-1.5 text-sm text-muted-foreground [overflow-wrap:anywhere]">
                {model.name}
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
        <div
          className={cn(
            "grid gap-3",
            model.context_length ? "sm:grid-cols-4" : "sm:grid-cols-3",
          )}
        >
          <ModelFactCard
            title="Mesh Availability"
            value={`${model.node_count} node${model.node_count === 1 ? "" : "s"}`}
            icon={<Network className="h-4 w-4" />}
            tooltip="Warm nodes currently serving this model."
          />
          <ModelFactCard
            title="Mesh VRAM"
            value={`${(model.mesh_vram_gb ?? 0).toFixed(1)} GB`}
            icon={<MemoryStick className="h-4 w-4" />}
            tooltip="Total available VRAM across nodes serving this model."
          />
          <ModelFactCard
            title="File size"
            value={model.size_gb > 0 ? `${model.size_gb.toFixed(1)} GB` : "Unknown"}
            icon={<MemoryStick className="h-4 w-4" />}
            tooltip={fileSizeTooltip(model.source_file)}
          />
          {model.context_length ? (
            <ModelFactCard
              title="Context"
              value={formatContextLength(model.context_length)}
              icon={<TextCursorInput className="h-4 w-4" />}
              tooltip="Approximate maximum context window."
            />
          ) : null}
        </div>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Sparkles className="h-4 w-4 text-muted-foreground" />
              <span>Capabilities</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="flex flex-wrap gap-2">
              <CapabilityBadge
                label="Text"
                icon={<MessageSquarePlus className="h-3.5 w-3.5" />}
                tooltip="Supports text input and text generation."
              />
              {model.multimodal ? (
                <CapabilityBadge
                  label="Multimodal"
                  icon={<Sparkles className="h-3.5 w-3.5" />}
                  tooltip="Supports one or more media input modalities."
                />
              ) : null}
              {model.vision ? (
                <CapabilityBadge
                  label="Vision"
                  icon={<ImagePlus className="h-3.5 w-3.5" />}
                  tooltip="Can understand image input."
                />
              ) : null}
              {model.audio ? (
                <CapabilityBadge
                  label="Audio"
                  icon={<Sparkles className="h-3.5 w-3.5" />}
                  tooltip="Can understand audio input."
                />
              ) : null}
              {model.reasoning ? (
                <CapabilityBadge
                  label="Reasoning"
                  icon={<Brain className="h-3.5 w-3.5" />}
                  tooltip="Reasoning-oriented model behavior."
                />
              ) : null}
              {model.moe ? (
                <CapabilityBadge
                  label="MoE"
                  icon={<Boxes className="h-3.5 w-3.5" />}
                  tooltip="Mixture-of-experts architecture."
                />
              ) : null}
              {model.tool_use ? (
                <CapabilityBadge
                  label="Tool Use"
                  icon={<Braces className="h-3.5 w-3.5" />}
                  tooltip={toolUseTooltip(model.tool_use_status)}
                />
              ) : null}
            </div>
          </CardContent>
        </Card>

        {(model.description ||
          model.architecture ||
          model.quantization ||
          model.draft_model ||
          model.source_ref ||
          model.source_revision ||
          model.expert_count ||
          model.used_expert_count ||
          model.ranking_source ||
          model.ranking_origin) ? (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Info className="h-4 w-4 text-muted-foreground" />
                <span>Model Details</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {model.description ? (
                <p className="leading-7 text-muted-foreground">{model.description}</p>
              ) : null}
              <div className="grid gap-3 sm:grid-cols-2">
                {model.architecture ? (
                  <ModelMetaItem
                    label="Architecture"
                    value={model.architecture}
                    icon={<Cpu className="h-3.5 w-3.5" />}
                  />
                ) : null}
                {model.quantization ? (
                  <ModelMetaItem
                    label="Quantization"
                    value={model.quantization}
                    icon={<Gauge className="h-3.5 w-3.5" />}
                  />
                ) : null}
                {model.draft_model ? (
                  <ModelMetaItem
                    label="Draft Pair"
                    value={model.draft_model}
                    icon={<Sparkles className="h-3.5 w-3.5" />}
                  />
                ) : null}
                {model.moe && model.expert_count && model.used_expert_count ? (
                  <ModelMetaItem
                    label="MoE Topology"
                    value={`${model.expert_count} experts · top-${model.used_expert_count}`}
                    icon={<Boxes className="h-3.5 w-3.5" />}
                  />
                ) : null}
                {formatMoeRanking(model) ? (
                  <ModelMetaItem
                    label="MoE Ranking"
                    value={formatMoeRanking(model)!}
                    icon={<Boxes className="h-3.5 w-3.5" />}
                  />
                ) : null}
                {formatMoeRankingOrigin(model) ? (
                  <ModelMetaItem
                    label="Ranking Origin"
                    value={formatMoeRankingOrigin(model)!}
                    icon={<Sparkles className="h-3.5 w-3.5" />}
                  />
                ) : null}
                {model.source_page_url ? (
                  <ModelMetaLinkItem
                    label="Model Source"
                    href={model.source_page_url}
                    text={
                      huggingFacePathFromUrl(model.source_page_url) ??
                      model.source_ref ??
                      model.name
                    }
                    icon={
                      isHuggingFaceUrl(model.source_page_url) ? (
                        <span aria-hidden="true" className="text-sm leading-none">
                          🤗
                        </span>
                      ) : (
                        <ExternalLink className="h-3.5 w-3.5" />
                      )
                    }
                  />
                ) : model.source_ref ? (
                  <ModelMetaItem
                    label="Model Source"
                    value={model.source_ref}
                    icon={<ExternalLink className="h-3.5 w-3.5" />}
                  />
                ) : null}
              </div>
            </CardContent>
          </Card>
        ) : null}

        {model.name || fullFileName || revisionFileName ? (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <FolderTree className="h-4 w-4 text-muted-foreground" />
                <span>Model Files</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <p className="text-sm leading-6 text-muted-foreground">
                The same model file shown as the mesh shorthand, repository path, and
                pinned revision.
              </p>
              <div className="grid gap-3">
                <ModelMetaItem
                  label="Shorthand"
                  value={model.name}
                  copyValue={model.name}
                />
                {fullFileName ? (
                  <ModelMetaItem
                    label="Full name"
                    value={fullFileName}
                    copyValue={fullFileName}
                  />
                ) : null}
                {revisionFileName ? (
                  <ModelMetaItem
                    label="Revision"
                    value={revisionFileName}
                    copyValue={revisionFileName}
                  />
                ) : null}
              </div>
            </CardContent>
          </Card>
        ) : null}

        {activePeers.length > 0 ? (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Network className="h-4 w-4 text-muted-foreground" />
                <span>Active Peers</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>ID</TableHead>
                    <TableHead className="text-right">Latency</TableHead>
                    <TableHead className="text-right">VRAM</TableHead>
                    <TableHead className="text-right">Share</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {activePeers.map((peer) => (
                    <TableRow key={peer.id}>
                      <TableCell className="font-mono text-xs">
                        <button
                          type="button"
                          className="rounded-sm text-left underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                          onClick={() => onOpenNode(peer.id)}
                        >
                          {peer.id}
                        </button>
                      </TableCell>
                      <TableCell className="text-right">{peer.latencyLabel}</TableCell>
                      <TableCell className="text-right">{peer.vramLabel}</TableCell>
                      <TableCell className="text-right">{peer.shareLabel}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        ) : null}

        {model.request_count != null || model.last_active_secs_ago != null ? (
          <div className="px-1 text-xs text-muted-foreground">
            {model.request_count != null
              ? `${model.request_count} requests seen`
              : "No request count"}
            {model.last_active_secs_ago != null
              ? ` · active ${model.last_active_secs_ago < 1 ? "just now" : `${formatShortDuration(model.last_active_secs_ago)} ago`}`
              : ""}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function formatMoeRanking(model?: MeshModel | null) {
  if (!model?.ranking_source) return null;
  if (model.ranking_source === "analyze") return "Full analyze";
  if (model.ranking_source !== "micro-analyze") return model.ranking_source;

  const parts = ["Micro-analyze"];
  if (model.ranking_prompt_count) {
    parts.push(`${model.ranking_prompt_count} prompt${model.ranking_prompt_count === 1 ? "" : "s"}`);
  }
  if (model.ranking_tokens) {
    parts.push(`${model.ranking_tokens} tokens`);
  }
  if (model.ranking_layer_scope) {
    parts.push(model.ranking_layer_scope === "all" ? "all layers" : "first layer");
  }
  return parts.join(" · ");
}

function formatMoeRankingOrigin(model?: MeshModel | null) {
  switch (model?.ranking_origin) {
    case "local-full-analyze":
      return "Local full analyze";
    case "local-micro-analyze":
      return "Local micro-analyze";
    case "peer-import":
      return "Peer import";
    case "legacy-cache":
      return "Legacy cache";
    default:
      return null;
  }
}

function huggingFacePathFromUrl(url?: string) {
  if (!url) return null;
  return url.replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/$/, "");
}

function isHuggingFaceUrl(url?: string) {
  return !!url && /^https?:\/\/huggingface\.co\//.test(url);
}

function modelFullFileName(model?: MeshModel | null) {
  if (!model) return null;
  if (model.source_ref && model.source_file) {
    return `${model.source_ref}/${model.source_file}`;
  }
  return model.source_file || model.source_ref || null;
}

function modelRevisionFileName(model?: MeshModel | null) {
  if (!model?.source_revision) return null;
  if (model.source_ref && model.source_file) {
    return `${model.source_ref}@${model.source_revision}/${model.source_file}`;
  }
  const fullName = modelFullFileName(model);
  if (!fullName) return null;
  return `${fullName}@${model.source_revision}`;
}


function formatContextLength(value?: number) {
  if (!value || !Number.isFinite(value)) return "Unknown";
  if (value >= 1000) {
    const rounded = value / 1000;
    return Number.isInteger(rounded) ? `${rounded}K` : `${rounded.toFixed(1)}K`;
  }
  return `${value}`;
}

function headerFitLabel(label?: string) {
  if (label === "Likely comfortable" || label === "Likely fits") {
    return "Suitable for this node";
  }
  if (label === "Possible with tradeoffs") {
    return "May fit this node";
  }
  if (label === "Likely too large") {
    return "Too large for this node";
  }
  return "Check fit";
}

function fitLabelTone(
  label?: string,
): "good" | "info" | "warn" | "bad" | "neutral" {
  if (label === "Likely comfortable") return "good";
  if (label === "Likely fits") return "good";
  if (label === "Possible with tradeoffs") return "warn";
  if (label === "Likely too large") return "bad";
  return "neutral";
}

function fitLabelTooltip(label?: string) {
  if (label === "Likely comfortable") {
    return "Should run comfortably on this node.";
  }
  if (label === "Likely fits") {
    return "Should fit on this node, but tightly.";
  }
  if (label === "Possible with tradeoffs") {
    return "May fit, with tighter memory or performance tradeoffs.";
  }
  if (label === "Likely too large") {
    return "Likely too large for this node alone.";
  }
  return "Estimated fit for this node.";
}

function fileSizeTooltip(fileName?: string) {
  const ext = fileName?.split(".").pop()?.toUpperCase();
  if (ext === "GGUF") return "GGUF model file size on disk.";
  if (ext) return `${ext} model file size on disk.`;
  return "Primary model file size on disk.";
}

function toolUseTooltip(status?: string) {
  if (status === "supported") {
    return "Supports tool or function calling.";
  }
  if (status === "likely") {
    return "Likely supports tool or function calling.";
  }
  return "Tool or function calling support.";
}
