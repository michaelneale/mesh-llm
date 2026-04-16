import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Minus, Plus, RotateCcw } from "lucide-react";

import {
  useResolvedTheme,
  type ResolvedTheme,
  type ThemeMode,
} from "../../../../lib/resolved-theme";
import { cn } from "../../../../lib/utils";
import {
  formatLatency,
  shortName,
} from "../../../app-shell/lib/status-helpers";
import type { TopologyNode } from "../../../app-shell/lib/topology-types";
import { EmptyPanel } from "../details";

type TopologyStatusPayload = {
  model_name?: string | null;
};

type TopologyThemeMode = ThemeMode;

type NodePalette = {
  fill: string;
  fillAlpha: number;
  line: string;
  lineAlpha: number;
};

type ScenePalette = {
  shellClassName: string;
  chromeClassName: string;
  selectedModelClassName: string;
  tooltipClassName: string;
  tooltipSubtleTextClassName: string;
  tooltipMutedTextClassName: string;
  tooltipDividerClassName: string;
  selfLabelPrimaryClassName: string;
  selfLabelSecondaryClassName: string;
  selfLabelTextShadow: string;
  surfaceGradient: (focalPoint: string) => string;
  nebulaBlendClassName: string;
  nebulaOpacity: number;
  nebulaGradient: string;
  gridOpacity: number;
  gridLineColor: string;
  selfHaloGradient: string;
  additiveBlend: boolean;
  lineTailAlpha: number;
  nodes: {
    client: NodePalette;
    worker: NodePalette;
    active: NodePalette;
    serving: NodePalette;
    self: NodePalette & {
      selectedLine: string;
      selectedLineAlpha: number;
    };
  };
};

type RenderNode = {
  id: string;
  label: string;
  subtitle: string;
  hostname?: string;
  role: string;
  statusLabel: string;
  latencyLabel: string;
  vramLabel: string;
  modelLabel: string;
  gpuLabel: string;
  x: number;
  y: number;
  size: number;
  color: [number, number, number, number];
  lineColor: [number, number, number, number];
  pulse: number;
  selectedModelMatch: boolean;
  z: number;
};

type ScreenNode = RenderNode & {
  px: number;
  py: number;
};
type EntryAnimation = {
  fromX: number;
  fromY: number;
  control1X: number;
  control1Y: number;
  control2X: number;
  control2Y: number;
  toX: number;
  toY: number;
  normalX: number;
  normalY: number;
  meanderAmplitude: number;
  meanderCycles: number;
  meanderPhase: number;
  startedAt: number;
};
type ExitAnimation = {
  node: RenderNode;
  startedAt: number;
};
type UpdateTwinkle = {
  startedAt: number;
};

const TAU = Math.PI * 2;

function hashString(value: string) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

function color(hex: string, alpha = 1): [number, number, number, number] {
  const normalized = hex.replace("#", "");
  const expanded =
    normalized.length === 3
      ? normalized
          .split("")
          .map((part) => `${part}${part}`)
          .join("")
      : normalized;
  const r = parseInt(expanded.slice(0, 2), 16) / 255;
  const g = parseInt(expanded.slice(2, 4), 16) / 255;
  const b = parseInt(expanded.slice(4, 6), 16) / 255;
  return [r, g, b, alpha];
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

const SCENE_PALETTES: Record<ResolvedTheme, ScenePalette> = {
  dark: {
    shellClassName:
      "border-white/10 bg-[#06111f] shadow-[inset_0_1px_0_rgba(255,255,255,0.06),0_20px_60px_rgba(0,0,0,0.28)]",
    chromeClassName: "border-white/10 bg-slate-950/72 text-slate-100 hover:bg-slate-900/80",
    selectedModelClassName:
      "border-amber-300/20 bg-amber-300/10 text-amber-100/90",
    tooltipClassName:
      "border-white/10 bg-slate-950/88 text-white shadow-2xl shadow-black/50",
    tooltipSubtleTextClassName: "text-slate-300",
    tooltipMutedTextClassName: "text-slate-500",
    tooltipDividerClassName: "border-white/10 text-slate-400",
    selfLabelPrimaryClassName: "text-white",
    selfLabelSecondaryClassName: "text-slate-200",
    selfLabelTextShadow: "0 1px 10px rgba(0,0,0,0.72)",
    surfaceGradient: (focalPoint) => `
      radial-gradient(circle at ${focalPoint}, rgba(59,130,246,0.12), transparent 22%),
      radial-gradient(circle at 72% 34%, rgba(250,204,21,0.10), transparent 26%),
      radial-gradient(circle at 30% 58%, rgba(56,189,248,0.08), transparent 32%),
      radial-gradient(circle at ${focalPoint}, rgba(255,255,255,0.035) 0, transparent 38%),
      linear-gradient(180deg, rgba(12,24,40,0.98), rgba(6,17,31,1))
    `,
    nebulaBlendClassName: "mix-blend-screen",
    nebulaOpacity: 0.48,
    nebulaGradient: `
      radial-gradient(40% 28% at 28% 60%, rgba(56,189,248,0.11), transparent 72%),
      radial-gradient(34% 26% at 72% 32%, rgba(250,204,21,0.1), transparent 74%),
      radial-gradient(46% 32% at 50% 54%, rgba(99,102,241,0.08), transparent 78%)
    `,
    gridOpacity: 0.1,
    gridLineColor: "rgba(255,255,255,0.04)",
    selfHaloGradient:
      "radial-gradient(circle, rgba(255,255,255,0.18) 0%, rgba(192,132,252,0.14) 18%, rgba(59,130,246,0.08) 42%, transparent 72%)",
    additiveBlend: true,
    lineTailAlpha: 0.02,
    nodes: {
      client: { fill: "#94a3b8", fillAlpha: 0.84, line: "#7c8ba1", lineAlpha: 0 },
      worker: { fill: "#7dd3fc", fillAlpha: 0.96, line: "#38bdf8", lineAlpha: 0 },
      active: { fill: "#4ade80", fillAlpha: 0.95, line: "#4ade80", lineAlpha: 0 },
      serving: { fill: "#f8fafc", fillAlpha: 1, line: "#facc15", lineAlpha: 0.22 },
      self: {
        fill: "#c084fc",
        fillAlpha: 1,
        line: "#c084fc",
        lineAlpha: 0.16,
        selectedLine: "#facc15",
        selectedLineAlpha: 0.34,
      },
    },
  },
  light: {
    shellClassName:
      "border-slate-200/80 bg-slate-50/95 shadow-[inset_0_1px_0_rgba(255,255,255,0.96),0_18px_40px_rgba(148,163,184,0.18)]",
    chromeClassName:
      "border-slate-200/85 bg-white/84 text-slate-700 shadow-sm shadow-slate-200/80 hover:bg-white",
    selectedModelClassName:
      "border-amber-300/70 bg-amber-100/92 text-amber-900",
    tooltipClassName:
      "border-slate-200/90 bg-white/92 text-slate-900 shadow-2xl shadow-slate-200/80",
    tooltipSubtleTextClassName: "text-slate-600",
    tooltipMutedTextClassName: "text-slate-500",
    tooltipDividerClassName: "border-slate-200/90 text-slate-500",
    selfLabelPrimaryClassName: "text-slate-900",
    selfLabelSecondaryClassName: "text-slate-600",
    selfLabelTextShadow: "0 1px 10px rgba(255,255,255,0.92)",
    surfaceGradient: (focalPoint) => `
      radial-gradient(circle at ${focalPoint}, rgba(59,130,246,0.08), transparent 24%),
      radial-gradient(circle at 72% 34%, rgba(245,158,11,0.11), transparent 24%),
      radial-gradient(circle at 30% 58%, rgba(14,165,233,0.07), transparent 31%),
      radial-gradient(circle at ${focalPoint}, rgba(255,255,255,0.58) 0, transparent 34%),
      linear-gradient(180deg, rgba(248,250,252,0.98), rgba(241,245,249,1))
    `,
    nebulaBlendClassName: "mix-blend-multiply",
    nebulaOpacity: 0.22,
    nebulaGradient: `
      radial-gradient(42% 30% at 28% 60%, rgba(14,165,233,0.08), transparent 72%),
      radial-gradient(34% 26% at 72% 32%, rgba(245,158,11,0.09), transparent 74%),
      radial-gradient(46% 32% at 50% 54%, rgba(124,58,237,0.05), transparent 78%)
    `,
    gridOpacity: 0.22,
    gridLineColor: "rgba(51,65,85,0.08)",
    selfHaloGradient:
      "radial-gradient(circle, rgba(255,255,255,0.88) 0%, rgba(196,181,253,0.38) 20%, rgba(125,211,252,0.18) 44%, transparent 74%)",
    additiveBlend: false,
    lineTailAlpha: 0.05,
    nodes: {
      client: { fill: "#64748b", fillAlpha: 0.5, line: "#64748b", lineAlpha: 0 },
      worker: { fill: "#0284c7", fillAlpha: 0.72, line: "#0284c7", lineAlpha: 0 },
      active: { fill: "#16a34a", fillAlpha: 0.74, line: "#16a34a", lineAlpha: 0 },
      serving: { fill: "#0f172a", fillAlpha: 0.9, line: "#f59e0b", lineAlpha: 0.18 },
      self: {
        fill: "#7c3aed",
        fillAlpha: 0.96,
        line: "#8b5cf6",
        lineAlpha: 0.12,
        selectedLine: "#f59e0b",
        selectedLineAlpha: 0.24,
      },
    },
  },
};

function distributePerimeterClients(nodes: TopologyNode[]) {
  const placed: Array<{ x: number; y: number }> = [];

  return nodes.map((node) => {
    const angle = hashString(`${node.id}:angle`) * TAU;
    const edgeBias = 0.74 + hashString(`${node.id}:radius`) * 0.16;
    const ellipseX = 0.5 + Math.cos(angle) * 0.4 * edgeBias;
    const ellipseY = 0.5 + Math.sin(angle) * 0.35 * edgeBias;
    const tangentJitter = (hashString(`${node.id}:tangent`) - 0.5) * 0.06;
    const radialJitter = (hashString(`${node.id}:radial`) - 0.5) * 0.025;
    const tangentAngle = angle + Math.PI / 2;
    let x = clamp(
      ellipseX +
        Math.cos(tangentAngle) * tangentJitter +
        Math.cos(angle) * radialJitter,
      0.08,
      0.92,
    );
    let y = clamp(
      ellipseY +
        Math.sin(tangentAngle) * tangentJitter +
        Math.sin(angle) * radialJitter,
      0.1,
      0.9,
    );

    for (const prior of placed) {
      const dx = x - prior.x;
      const dy = y - prior.y;
      const distance = Math.hypot(dx, dy);
      if (distance > 0 && distance < 0.032) {
        const push = (0.032 - distance) * 0.5;
        x = clamp(x + (dx / distance) * push, 0.08, 0.92);
        y = clamp(y + (dy / distance) * push, 0.1, 0.9);
      }
    }
    placed.push({ x, y });

    return {
      node,
      x,
      y,
    };
  });
}

function normalizedLatency(node: TopologyNode, minLatency: number, maxLatency: number) {
  if (node.latencyMs == null || !Number.isFinite(node.latencyMs)) {
    return 0.45;
  }
  if (maxLatency <= minLatency) {
    return 0.2;
  }
  return clamp((node.latencyMs - minLatency) / (maxLatency - minLatency), 0, 1);
}

function radiusMixFromLatencyMs(
  node: TopologyNode,
  minLatency: number,
  maxLatency: number,
  radiusSeed: number,
  radialBias: number,
) {
  const latencyNorm = Math.pow(normalizedLatency(node, minLatency, maxLatency), 0.9);
  return clamp(latencyNorm * 0.78 + radiusSeed * 0.22 + radialBias, 0, 1);
}

function distributeLatencyBand(
  nodes: TopologyNode[],
  minLatency: number,
  maxLatency: number,
  angleStart: number,
  angleEnd: number,
  innerRadiusX: number,
  outerRadiusX: number,
  innerRadiusY: number,
  outerRadiusY: number,
  armCount = 3,
  radialBias = 0,
  curveLimit = 0.7,
) {
  const angleSpan = angleEnd - angleStart;

  return nodes.map((node) => {
    const latencyNorm = normalizedLatency(node, minLatency, maxLatency);
    const identity = `${node.id}:${node.hostname ?? ""}:${node.serving}`;
    const armSeed = hashString(`${identity}:arm`);
    const armIndex = Math.floor(armSeed * armCount) % armCount;
    const armBandStart = angleStart + (armIndex / armCount) * angleSpan;
    const armBandEnd = angleStart + ((armIndex + 1) / armCount) * angleSpan;
    const angleSeed = hashString(`${identity}:angle`);
    const angle = armBandStart + angleSeed * (armBandEnd - armBandStart);
    const radiusSeed = hashString(`${identity}:radius`);
    const radiusMix = radiusMixFromLatencyMs(
      node,
      minLatency,
      maxLatency,
      radiusSeed,
      radialBias,
    );
    const radiusX = innerRadiusX + (outerRadiusX - innerRadiusX) * radiusMix;
    const radiusY = innerRadiusY + (outerRadiusY - innerRadiusY) * radiusMix;
    const tangentAngle = angle + Math.PI / 2;
    const tangentDrift = (hashString(`${identity}:tangent-drift`) - 0.5) * Math.min(0.05, curveLimit * 0.08);
    const radialDrift = (hashString(`${identity}:radial-drift`) - 0.5) * 0.024;
    const driftX =
      Math.cos(tangentAngle) * tangentDrift + Math.cos(angle) * radialDrift;
    const driftY =
      Math.sin(tangentAngle) * tangentDrift + Math.sin(angle) * radialDrift;
    const x = clamp(0.5 + Math.cos(angle) * radiusX + driftX, 0.12, 0.88);
    const y = clamp(0.52 + Math.sin(angle) * radiusY + driftY, 0.16, 0.84);

    return {
      node,
      latencyNorm,
      x,
      y,
    };
  });
}

function nodeSize(node: TopologyNode, emphasis: number) {
  const base = node.client ? 8 : 10;
  const vramBoost = node.client ? 0 : Math.sqrt(Math.max(0, node.vram)) * 1.55;
  return clamp(base + vramBoost + emphasis, node.client ? 6 : 10, 38);
}

function resolveNodeOverlap(nodes: RenderNode[], fixedNode?: RenderNode) {
  const movableNodes = nodes
    .sort((left, right) => left.id.localeCompare(right.id));
  const placed: Array<{ x: number; y: number; size: number }> = fixedNode
    ? [{ x: fixedNode.x, y: fixedNode.y, size: fixedNode.size }]
    : [];

  const resolved = movableNodes.map((node) => {
    const baseRadius = 0.02 + node.size * 0.0012;

    for (let attempt = 0; attempt < 24; attempt += 1) {
      const angle = hashString(`${node.id}:resolve-angle:${attempt}`) * TAU;
      const offset = attempt === 0 ? 0 : baseRadius * (0.55 + attempt * 0.22);
      const candidateX = clamp(node.x + Math.cos(angle) * offset, 0.08, 0.92);
      const candidateY = clamp(node.y + Math.sin(angle) * offset, 0.1, 0.9);
      const overlaps = placed.some((prior) => {
        const minDistance = 0.012 + (node.size + prior.size) * 0.0014;
        return Math.hypot(candidateX - prior.x, candidateY - prior.y) < minDistance;
      });
      if (!overlaps) {
        placed.push({ x: candidateX, y: candidateY, size: node.size });
        return { ...node, x: candidateX, y: candidateY };
      }
    }

    placed.push({ x: node.x, y: node.y, size: node.size });
    return node;
  });

  return resolved;
}

function nodeUpdateSignature(node: TopologyNode) {
  return JSON.stringify({
    vram: node.vram,
    host: node.host,
    client: node.client,
    serving: node.serving,
    servingModels: node.servingModels,
    statusLabel: node.statusLabel,
    latencyMs: node.latencyMs,
    hostname: node.hostname,
    isSoc: node.isSoc,
    gpus:
      node.gpus?.map((gpu) => ({
        name: gpu.name,
        vram_bytes: gpu.vram_bytes,
        bandwidth_gbps: gpu.bandwidth_gbps,
      })) ?? [],
  });
}

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
      statusLabel: "Serving",
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
      statusLabel: "Standby",
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
    statusLabel: "Client",
    latencyMs: 42 + (index % 7) * 12,
    hostname: `test-client-${index}`,
    isSoc: false,
    gpus: [],
  };
}

function useRadarFieldNodes(
  nodes: TopologyNode[],
  selectedModel: string,
  fallbackModel: string,
  palette: ScenePalette["nodes"],
) {
  return useMemo<RenderNode[]>(() => {
    if (!nodes.length) return [];

    const focusModel = selectedModel || fallbackModel || "";
    const selfNode = nodes.find((node) => node.self) ?? nodes[0];
    const others = nodes.filter((node) => node.id !== selfNode.id);

    const selectedServing = others.filter(
      (node) =>
        !node.client &&
        !!focusModel &&
        node.servingModels.some((model) => model === focusModel),
    );
    const selectedIds = new Set(selectedServing.map((node) => node.id));
    const serving = others.filter(
      (node) =>
        !node.client &&
        !selectedIds.has(node.id) &&
        node.servingModels.some((model) => model && model !== "(idle)"),
    );
    const clients = others.filter((node) => node.client);
    const workers = others.filter(
      (node) => !node.client && !selectedIds.has(node.id) && !serving.some((entry) => entry.id === node.id),
    );

    const latencyNodes = others.filter(
      (node) => !node.client && node.latencyMs != null && Number.isFinite(node.latencyMs),
    );
    const minLatency =
      latencyNodes.length > 0
        ? Math.min(...latencyNodes.map((node) => Number(node.latencyMs)))
        : 0;
    const maxLatency =
      latencyNodes.length > 0
        ? Math.max(...latencyNodes.map((node) => Number(node.latencyMs)))
        : 1;

    const servingNodes = distributeLatencyBand(
      selectedServing,
      minLatency,
      maxLatency,
      0,
      TAU,
      0.08,
      0.18,
      0.07,
      0.15,
      3,
      -0.1,
      0.42,
    );
    const workerNodes = distributeLatencyBand(
      workers,
      minLatency,
      maxLatency,
      0,
      TAU,
      0.18,
      0.34,
      0.15,
      0.28,
      3,
      0.08,
      0.7,
    );
    const clientNodes = distributePerimeterClients(clients);
    const activeNodes = distributeLatencyBand(
      serving,
      minLatency,
      maxLatency,
      0,
      TAU,
      0.19,
      0.32,
      0.15,
      0.25,
      2,
      0.14,
      0.6,
    );

    const output: RenderNode[] = [];

    for (const { node, x, y } of clientNodes) {
      output.push({
        id: node.id,
        label: node.hostname || node.id,
        subtitle: "Client",
        hostname: node.hostname,
        role: "Client",
        statusLabel: node.statusLabel,
        latencyLabel: formatLatency(node.latencyMs),
        vramLabel: "n/a",
        modelLabel: "API-only",
        gpuLabel: "No GPU",
        x,
        y,
        size: nodeSize(node, 1.5),
        color: color(palette.client.fill, palette.client.fillAlpha),
        lineColor: color(palette.client.line, palette.client.lineAlpha),
        pulse: 0.28 + hashString(`${node.id}:pulse`) * 0.24,
        selectedModelMatch: false,
        z: 1,
      });
    }

    for (const { node, x, y, latencyNorm } of workerNodes) {
      const models = node.servingModels.filter((model) => model && model !== "(idle)");
      const glowAlpha = 0.96 - latencyNorm * 0.22;
      output.push({
        id: node.id,
        label: node.hostname || node.id,
        subtitle: node.statusLabel,
        hostname: node.hostname,
        role: node.host ? "Host" : "Worker",
        statusLabel: node.statusLabel,
        latencyLabel: formatLatency(node.latencyMs),
        vramLabel: `${Math.max(0, node.vram).toFixed(1)} GB`,
        modelLabel: models.length > 0 ? models.map(shortName).join(", ") : "idle",
        gpuLabel:
          node.gpus && node.gpus.length > 0
            ? `${node.gpus.length} GPU${node.gpus.length === 1 ? "" : "s"}`
            : node.isSoc
              ? "SoC"
              : "GPU unknown",
        x,
        y,
        size: nodeSize(node, 3),
        color: color(palette.worker.fill, Math.min(glowAlpha, palette.worker.fillAlpha)),
        lineColor: color(palette.worker.line, palette.worker.lineAlpha),
        pulse: 0.42 + (1 - latencyNorm) * 0.34 + hashString(`${node.id}:pulse`) * 0.36,
        selectedModelMatch: false,
        z: 2,
      });
    }

    for (const { node, x, y, latencyNorm } of activeNodes) {
      const models = node.servingModels.filter((model) => model && model !== "(idle)");
      const glowAlpha = 0.95 - latencyNorm * 0.18;
      output.push({
        id: node.id,
        label: node.hostname || node.id,
        subtitle: node.statusLabel,
        hostname: node.hostname,
        role: node.host ? "Host" : "Serving",
        statusLabel: node.statusLabel,
        latencyLabel: formatLatency(node.latencyMs),
        vramLabel: `${Math.max(0, node.vram).toFixed(1)} GB`,
        modelLabel: models.length > 0 ? models.map(shortName).join(", ") : "idle",
        gpuLabel:
          node.gpus && node.gpus.length > 0
            ? `${node.gpus.length} GPU${node.gpus.length === 1 ? "" : "s"}`
            : node.isSoc
              ? "SoC"
              : "GPU unknown",
        x,
        y,
        size: nodeSize(node, 5),
        color: color(palette.active.fill, Math.min(glowAlpha, palette.active.fillAlpha)),
        lineColor: color(palette.active.line, palette.active.lineAlpha),
        pulse: 0.68 + (1 - latencyNorm) * 0.4 + hashString(`${node.id}:pulse`) * 0.34,
        selectedModelMatch: false,
        z: 3,
      });
    }

    for (const { node, x, y, latencyNorm } of servingNodes) {
      const models = node.servingModels.filter((model) => model && model !== "(idle)");
      const glowAlpha = 1 - latencyNorm * 0.12;
      output.push({
        id: node.id,
        label: node.hostname || node.id,
        subtitle: focusModel ? shortName(focusModel) : node.statusLabel,
        hostname: node.hostname,
        role: node.host ? "Host" : "Serving",
        statusLabel: node.statusLabel,
        latencyLabel: formatLatency(node.latencyMs),
        vramLabel: `${Math.max(0, node.vram).toFixed(1)} GB`,
        modelLabel: models.length > 0 ? models.map(shortName).join(", ") : "idle",
        gpuLabel:
          node.gpus && node.gpus.length > 0
            ? `${node.gpus.length} GPU${node.gpus.length === 1 ? "" : "s"}`
            : node.isSoc
              ? "SoC"
              : "GPU unknown",
        x,
        y,
        size: nodeSize(node, 9),
        color: color(palette.serving.fill, Math.min(glowAlpha, palette.serving.fillAlpha)),
        lineColor: color(palette.serving.line, palette.serving.lineAlpha),
        pulse: 1.02 + (1 - latencyNorm) * 0.46 + hashString(`${node.id}:pulse`) * 0.28,
        selectedModelMatch: true,
        z: 4,
      });
    }

    const selfRenderNode: RenderNode = {
      id: selfNode.id,
      label: selfNode.hostname || (selfNode.client ? "this client" : "this node"),
      subtitle: selfNode.client ? "This client" : "This node",
      hostname: selfNode.hostname,
      role: selfNode.client ? "Client" : selfNode.host ? "Host" : "Node",
      statusLabel: selfNode.statusLabel,
      latencyLabel: "local",
      vramLabel: selfNode.client ? "n/a" : `${Math.max(0, selfNode.vram).toFixed(1)} GB`,
      modelLabel: selfNode.client
        ? "API-only"
        : selfNode.servingModels.filter((model) => model && model !== "(idle)").length > 0
          ? selfNode.servingModels
              .filter((model) => model && model !== "(idle)")
              .map(shortName)
              .join(", ")
          : "idle",
      gpuLabel:
        selfNode.client
          ? "No GPU"
          : selfNode.gpus && selfNode.gpus.length > 0
            ? `${selfNode.gpus.length} GPU${selfNode.gpus.length === 1 ? "" : "s"}`
            : selfNode.isSoc
              ? "SoC"
              : "GPU unknown",
      x: 0.5,
      y: 0.52,
      size: nodeSize(selfNode, 15),
      color: color(palette.self.fill, palette.self.fillAlpha),
      lineColor:
        focusModel && selfNode.servingModels.includes(focusModel)
          ? color(palette.self.selectedLine, palette.self.selectedLineAlpha)
          : color(palette.self.line, palette.self.lineAlpha),
      pulse: 1.4,
      selectedModelMatch:
        !!focusModel && selfNode.servingModels.some((model) => model === focusModel),
      z: 5,
    };

    const resolvedOutput = resolveNodeOverlap(output, selfRenderNode);
    resolvedOutput.push(selfRenderNode);

    return resolvedOutput.sort((left, right) => left.z - right.z || left.size - right.size);
  }, [fallbackModel, nodes, palette, selectedModel]);
}

function createShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string,
) {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(
  gl: WebGLRenderingContext,
  vertexSource: string,
  fragmentSource: string,
) {
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  if (!vertexShader || !fragmentShader) return null;

  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

function easeOutCubic(value: number) {
  return 1 - Math.pow(1 - value, 3);
}

function cubicBezier(
  start: number,
  control1: number,
  control2: number,
  end: number,
  t: number,
) {
  const inv = 1 - t;
  return (
    inv * inv * inv * start +
    3 * inv * inv * t * control1 +
    3 * inv * t * t * control2 +
    t * t * t * end
  );
}

function createTravelAnimation(
  nodeId: string,
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  startedAt: number,
  randomSource = `${nodeId}:entry`,
): EntryAnimation {
  const horizontalSpan = toX - fromX;
  const verticalSpan = toY - fromY;
  const distance = Math.max(1, Math.hypot(horizontalSpan, verticalSpan));
  const normalX = -verticalSpan / distance;
  const normalY = horizontalSpan / distance;
  const bendDirection = hashString(`${randomSource}:bend`) > 0.5 ? 1 : -1;
  const bendAmount =
    Math.min(distance * 0.18, 82) *
    (0.6 + hashString(`${randomSource}:arc`) * 0.9) *
    bendDirection;
  const lateralJitter = (hashString(`${randomSource}:lateral`) - 0.5) * 42;

  return {
    fromX,
    fromY,
    control1X: fromX + horizontalSpan * 0.24 + normalX * bendAmount + lateralJitter,
    control1Y: fromY + verticalSpan * 0.18 + normalY * bendAmount,
    control2X:
      fromX +
      horizontalSpan * 0.72 -
      normalX * bendAmount * 0.56 -
      lateralJitter * 0.35,
    control2Y: fromY + verticalSpan * 0.82 - normalY * bendAmount * 0.56,
    toX,
    toY,
    normalX,
    normalY,
    meanderAmplitude: Math.min(distance * 0.06, 28) * (0.6 + hashString(`${randomSource}:meander`) * 0.8),
    meanderCycles: 1.15 + hashString(`${randomSource}:cycles`) * 1.35,
    meanderPhase: hashString(`${randomSource}:phase`) * Math.PI,
    startedAt,
  };
}

const POINT_VERTEX_SHADER = `
attribute vec2 a_position;
attribute float a_size;
attribute vec4 a_color;
attribute float a_pulse;
attribute float a_twinkle;
uniform vec2 u_resolution;
uniform float u_time;
varying vec4 v_color;
varying float v_glow;
varying float v_twinkle;

void main() {
  vec2 zeroToOne = a_position / u_resolution;
  vec2 clip = zeroToOne * 2.0 - 1.0;
  gl_Position = vec4(clip * vec2(1.0, -1.0), 0.0, 1.0);
  float pulse = 0.94 + 0.1 * sin(u_time * (0.55 + a_pulse * 0.18) + a_pulse * 6.2831);
  gl_PointSize = a_size * pulse * (1.0 + a_twinkle * 0.12);
  v_color = a_color;
  v_glow = 0.45 + a_pulse * 0.16;
  v_twinkle = a_twinkle;
}
`;

const POINT_FRAGMENT_SHADER = `
precision mediump float;
varying vec4 v_color;
varying float v_glow;
varying float v_twinkle;

void main() {
  vec2 centered = gl_PointCoord * 2.0 - 1.0;
  float distanceFromCenter = length(centered);
  if (distanceFromCenter > 1.0) {
    discard;
  }

  float glow = smoothstep(1.0, 0.08, distanceFromCenter);
  float core = smoothstep(0.45, 0.0, distanceFromCenter);
  float rim = smoothstep(0.92, 0.58, distanceFromCenter) * 0.42;
  vec2 twinkleCoords = centered;
  twinkleCoords.x *= mix(1.0, 0.82, v_twinkle);
  twinkleCoords.y *= mix(1.0, 1.08, v_twinkle);
  float starAxis = max(
    smoothstep(0.11, 0.0, abs(twinkleCoords.x)) * smoothstep(1.0, 0.14, abs(twinkleCoords.y)),
    smoothstep(0.11, 0.0, abs(twinkleCoords.y)) * smoothstep(1.0, 0.14, abs(twinkleCoords.x))
  );
  float starDiagonal = max(
    smoothstep(0.12, 0.0, abs(twinkleCoords.x - twinkleCoords.y)) *
      smoothstep(1.08, 0.2, abs(twinkleCoords.x + twinkleCoords.y)),
    smoothstep(0.12, 0.0, abs(twinkleCoords.x + twinkleCoords.y)) *
      smoothstep(1.08, 0.2, abs(twinkleCoords.x - twinkleCoords.y))
  );
  float starNeedle = pow(max(abs(centered.x), abs(centered.y)), 0.35);
  float starSpark = smoothstep(0.24, 0.0, distanceFromCenter) * v_twinkle;
  float starFlare = ((starAxis * 1.25 + starDiagonal * 0.55) * (1.0 - starNeedle * 0.42) + starSpark * 0.75) * v_twinkle;
  vec3 colorMix = mix(
    v_color.rgb * 0.62,
    vec3(1.0, 0.98, 0.92),
    core * 0.34 + rim * 0.16 + starFlare * 0.82
  );
  float alpha = glow * v_glow + core * 0.65 + rim + starFlare * 1.18;

  gl_FragColor = vec4(colorMix, alpha * v_color.a);
}
`;

const LINE_VERTEX_SHADER = `
attribute vec2 a_position;
attribute vec4 a_color;
uniform vec2 u_resolution;
varying vec4 v_color;

void main() {
  vec2 zeroToOne = a_position / u_resolution;
  vec2 clip = zeroToOne * 2.0 - 1.0;
  gl_Position = vec4(clip * vec2(1.0, -1.0), 0.0, 1.0);
  v_color = a_color;
}
`;

const LINE_FRAGMENT_SHADER = `
precision mediump float;
varying vec4 v_color;

void main() {
  gl_FragColor = v_color;
}
`;

export function MeshTopologyDiagram({
  status,
  nodes,
  selectedModel,
  themeMode,
  onOpenNode,
  highlightedNodeId,
  fullscreen,
  heightClass,
  containerStyle,
}: {
  status: TopologyStatusPayload | null;
  nodes: TopologyNode[];
  selectedModel: string;
  themeMode: TopologyThemeMode;
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
    <MeshRadarField
      status={status}
      nodes={nodes}
      selectedModel={selectedModel}
      themeMode={themeMode}
      onOpenNode={onOpenNode}
      highlightedNodeId={highlightedNodeId}
      fullscreen={fullscreen ?? false}
      heightClass={heightClass}
      containerStyle={containerStyle}
    />
  );
}

function MeshRadarField({
  status,
  nodes,
  selectedModel,
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
  themeMode: TopologyThemeMode;
  onOpenNode?: (nodeId: string) => void;
  highlightedNodeId?: string;
  fullscreen: boolean;
  heightClass?: string;
  containerStyle?: CSSProperties;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const hostRef = useRef<HTMLDivElement | null>(null);
  const screenNodesRef = useRef<ScreenNode[]>([]);
  const animationRef = useRef<Map<string, EntryAnimation>>(new Map());
  const exitAnimationRef = useRef<Map<string, ExitAnimation>>(new Map());
  const twinkleAnimationRef = useRef<Map<string, UpdateTwinkle>>(new Map());
  const previousRenderNodesRef = useRef<Map<string, RenderNode>>(new Map());
  const lastScreenPositionsRef = useRef<Map<string, { x: number; y: number }>>(new Map());
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
  const scene = useMemo(() => SCENE_PALETTES[resolvedTheme], [resolvedTheme]);
  const selfNode = useMemo(() => nodes.find((node) => node.self) ?? nodes[0], [nodes]);
  const [tooltipStyle, setTooltipStyle] = useState<CSSProperties | null>(null);
  const [hostSize, setHostSize] = useState({ width: 0, height: 0 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const zoomRef = useRef(1);
  const panRef = useRef({ x: 0, y: 0 });
  const [debugNodes, setDebugNodes] = useState<TopologyNode[]>([]);
  const displayNodes = useMemo(() => [...nodes, ...debugNodes], [debugNodes, nodes]);
  const renderNodes = useRadarFieldNodes(
    displayNodes,
    selectedModel,
    status.model_name ?? "",
    scene.nodes,
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
    const now = performance.now();

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

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = hostRef.current;
    if (!canvas || !host) return;

    const gl = canvas.getContext("webgl", {
      alpha: true,
      antialias: true,
      premultipliedAlpha: true,
    });
    if (!gl) return;

    const pointProgram = createProgram(gl, POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER);
    const lineProgram = createProgram(gl, LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER);
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
    const lineResolutionLocation = gl.getUniformLocation(lineProgram, "u_resolution");

    const pointPositionBuffer = gl.createBuffer();
    const pointSizeBuffer = gl.createBuffer();
    const pointColorBuffer = gl.createBuffer();
    const pointPulseBuffer = gl.createBuffer();
    const pointTwinkleBuffer = gl.createBuffer();
    const linePositionBuffer = gl.createBuffer();
    const lineColorBuffer = gl.createBuffer();
    if (
      !pointPositionBuffer ||
      !pointSizeBuffer ||
      !pointColorBuffer ||
      !pointPulseBuffer ||
      !pointTwinkleBuffer ||
      !linePositionBuffer ||
      !lineColorBuffer
    ) {
      return;
    }

    let frame = 0;
    let animationFrame = 0;
    let width = 0;
    let height = 0;
    let cssWidth = 0;
    let cssHeight = 0;
    let devicePixelRatio = 1;

    const resize = () => {
      const rect = host.getBoundingClientRect();
      devicePixelRatio = window.devicePixelRatio || 1;
      cssWidth = Math.max(1, rect.width);
      cssHeight = Math.max(1, rect.height);
      width = Math.max(1, Math.round(rect.width * devicePixelRatio));
      height = Math.max(1, Math.round(rect.height * devicePixelRatio));
      canvas.width = width;
      canvas.height = height;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      gl.viewport(0, 0, width, height);
    };

    const buildFrameData = () => {
      const pointPositions: number[] = [];
      const pointSizes: number[] = [];
      const pointColors: number[] = [];
      const pointPulses: number[] = [];
      const pointTwinkles: number[] = [];
      const linePositions: number[] = [];
      const lineColors: number[] = [];
      const screenNodes: ScreenNode[] = [];
      const centerNode =
        renderNodes.find((node) => node.id === selfNode?.id) ??
        renderNodes[renderNodes.length - 1];
      const now = performance.now();
      const highlightedSet = new Set<string>();
      if (selectedNodeIdRef.current) highlightedSet.add(selectedNodeIdRef.current);
      if (hoveredNodeIdRef.current) highlightedSet.add(hoveredNodeIdRef.current);
      const activeIds = new Set(renderNodes.map((node) => node.id));

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

      for (const node of renderNodes) {
        const currentZoom = zoomRef.current;
        const currentPan = panRef.current;
        if (
          node.id !== selfNode?.id &&
          !seenNodeIdsRef.current.has(node.id) &&
          !animationRef.current.has(node.id)
        ) {
          const approachAngle = hashString(`${node.id}:entry-angle`) * TAU;
          const entryRadius =
            Math.max(cssWidth, cssHeight) * (0.72 + hashString(`${node.id}:entry-radius`) * 0.28);
          const fromX = cssWidth * 0.5 + Math.cos(approachAngle) * entryRadius;
          const fromY = cssHeight * 0.5 + Math.sin(approachAngle) * entryRadius;
          const toX = node.x * cssWidth;
          const toY = node.y * cssHeight;
          animationRef.current.set(
            node.id,
            createTravelAnimation(node.id, fromX, fromY, toX, toY, now),
          );
        } else if (node.id !== selfNode?.id && !animationRef.current.has(node.id)) {
          const nextX = node.x * cssWidth;
          const nextY = node.y * cssHeight;
          const previousPosition = lastScreenPositionsRef.current.get(node.id);
          if (previousPosition) {
            const moveDistance = Math.hypot(nextX - previousPosition.x, nextY - previousPosition.y);
            if (moveDistance > 18) {
              animationRef.current.set(
                node.id,
                createTravelAnimation(
                  node.id,
                  previousPosition.x,
                  previousPosition.y,
                  nextX,
                  nextY,
                  now,
                  `${node.id}:move`,
                ),
              );
            }
          }
        }

        let px = node.x * cssWidth;
        let py = node.y * cssHeight;
        let entryOpacity = 1;
        let entryScale = 1;
        const entry = animationRef.current.get(node.id);
        if (entry) {
          entry.toX = node.x * cssWidth;
          entry.toY = node.y * cssHeight;
          const progress = clamp((now - entry.startedAt) / 1100, 0, 1);
          const eased = easeOutCubic(progress);
          px = cubicBezier(
            entry.fromX,
            entry.control1X,
            entry.control2X,
            entry.toX,
            eased,
          );
          py = cubicBezier(
            entry.fromY,
            entry.control1Y,
            entry.control2Y,
            entry.toY,
            eased,
          );
          const meanderEnvelope = Math.sin(progress * Math.PI);
          const meanderOffset =
            Math.sin(progress * Math.PI * entry.meanderCycles + entry.meanderPhase) *
            entry.meanderAmplitude *
            meanderEnvelope;
          px += entry.normalX * meanderOffset;
          py += entry.normalY * meanderOffset;
          entryOpacity = 0.2 + eased * 0.8;
          entryScale = 0.72 + eased * 0.28;
          if (progress >= 1) {
            animationRef.current.delete(node.id);
            px = entry.toX;
            py = entry.toY;
            entryOpacity = 1;
            entryScale = 1;
          }
        }
        seenNodeIdsRef.current.add(node.id);
        px = px * currentZoom + currentPan.x;
        py = py * currentZoom + currentPan.y;
        const isHighlighted = highlightedSet.has(node.id);
        const twinkle = twinkleAnimationRef.current.get(node.id);
        const twinkleProgress = twinkle
          ? clamp((now - twinkle.startedAt) / 1100, 0, 1)
          : 1;
        const twinkleStrength = twinkle
          ? Math.sin(twinkleProgress * Math.PI * 3.2) *
              (1 - twinkleProgress) *
              0.42
          : 0;
        const size =
          (node.size + (isHighlighted ? 6 : 0) + twinkleStrength * 16) * entryScale;
        const colorBoost = isHighlighted ? 0.18 : 0;
        const twinkleBoost = Math.max(0, twinkleStrength);
        pointPositions.push(px * devicePixelRatio, py * devicePixelRatio);
        pointSizes.push(size * devicePixelRatio);
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
          size,
        });
        lastScreenPositionsRef.current.set(node.id, {
          x: node.x * cssWidth,
          y: node.y * cssHeight,
        });

        if (
          centerNode &&
          node.id !== centerNode.id &&
          (node.selectedModelMatch || highlightedSet.has(node.id)) &&
          node.lineColor[3] > 0
        ) {
          const centerPx = centerNode.x * cssWidth * currentZoom + currentPan.x;
          const centerPy = centerNode.y * cssHeight * currentZoom + currentPan.y;
          linePositions.push(
            centerPx * devicePixelRatio,
            centerPy * devicePixelRatio,
            px * devicePixelRatio,
            py * devicePixelRatio,
          );
          lineColors.push(
            node.lineColor[0],
            node.lineColor[1],
            node.lineColor[2],
            node.lineColor[3],
            node.lineColor[0],
            node.lineColor[1],
            node.lineColor[2],
            scene.lineTailAlpha,
          );
        }
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
          clamp(node.color[2] + coolFade * 0.34 + shockwave * 0.16 + whiteCore * 0.12, 0, 1),
          clamp(node.color[3] * exitAlpha, 0, 1),
        );
        pointPulses.push(node.pulse + flash * 2.8 + shockwave * 0.9 + shockFront * 0.65);
        pointTwinkles.push(Math.min(1, whiteCore * 1.35 + shockFront * 0.3));
      }

      screenNodesRef.current = screenNodes.sort(
        (left, right) => right.z - left.z || right.size - left.size,
      );
      for (const key of lastScreenPositionsRef.current.keys()) {
        if (!activeIds.has(key)) {
          lastScreenPositionsRef.current.delete(key);
        }
      }

      return {
        pointPositions: new Float32Array(pointPositions),
        pointSizes: new Float32Array(pointSizes),
        pointColors: new Float32Array(pointColors),
        pointPulses: new Float32Array(pointPulses),
        pointTwinkles: new Float32Array(pointTwinkles),
        linePositions: new Float32Array(linePositions),
        lineColors: new Float32Array(lineColors),
      };
    };

    const render = () => {
      frame += 1;
      const {
        pointPositions,
        pointSizes,
        pointColors,
        pointPulses,
        pointTwinkles,
        linePositions,
        lineColors,
      } = buildFrameData();

      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      if (scene.additiveBlend) {
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      } else {
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      }

      if (linePositions.length > 0) {
        gl.useProgram(lineProgram);
        gl.uniform2f(lineResolutionLocation, width, height);

        gl.bindBuffer(gl.ARRAY_BUFFER, linePositionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, linePositions, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(linePositionLocation);
        gl.vertexAttribPointer(linePositionLocation, 2, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, lineColorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, lineColors, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(lineColorLocation);
        gl.vertexAttribPointer(lineColorLocation, 4, gl.FLOAT, false, 0, 0);

        gl.lineWidth(1);
        gl.drawArrays(gl.LINES, 0, linePositions.length / 2);
      }

      gl.useProgram(pointProgram);
      gl.uniform2f(pointResolutionLocation, width, height);
      gl.uniform1f(pointTimeLocation, frame / 60);

      gl.bindBuffer(gl.ARRAY_BUFFER, pointPositionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, pointPositions, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(pointPositionLocation);
      gl.vertexAttribPointer(pointPositionLocation, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, pointSizeBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, pointSizes, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(pointSizeLocation);
      gl.vertexAttribPointer(pointSizeLocation, 1, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, pointColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, pointColors, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(pointColorLocation);
      gl.vertexAttribPointer(pointColorLocation, 4, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, pointPulseBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, pointPulses, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(pointPulseLocation);
      gl.vertexAttribPointer(pointPulseLocation, 1, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, pointTwinkleBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, pointTwinkles, gl.DYNAMIC_DRAW);
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
      gl.deleteBuffer(pointPositionBuffer);
      gl.deleteBuffer(pointSizeBuffer);
      gl.deleteBuffer(pointColorBuffer);
      gl.deleteBuffer(pointPulseBuffer);
      gl.deleteBuffer(pointTwinkleBuffer);
      gl.deleteBuffer(linePositionBuffer);
      gl.deleteBuffer(lineColorBuffer);
      gl.deleteProgram(pointProgram);
      gl.deleteProgram(lineProgram);
    };
  }, [renderNodes, scene, selfNode?.id, status.model_name]);

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
        return Math.hypot(dx, dy) <= node.size * 0.8 + 10;
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
    dragRef.current.active = false;
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
  const selfScreenX = selfRenderNode
    ? selfRenderNode.x * hostSize.width * zoom + pan.x
    : 0.5 * hostSize.width * zoom + pan.x;
  const selfScreenY = selfRenderNode
    ? selfRenderNode.y * hostSize.height * zoom + pan.y
    : 0.52 * hostSize.height * zoom + pan.y;
  const focalPoint = `${selfScreenX}px ${selfScreenY}px`;
  const gridSize = fullscreen ? 64 : 56;

  const selfLabelStyle: CSSProperties = {
    left: `${selfScreenX}px`,
    top: `${selfScreenY}px`,
    transform: "translate(-50%, -50%)",
  };
  const nebulaStyle: CSSProperties = {
    transformOrigin: focalPoint,
    transform: `translate(${pan.x * 0.06}px, ${pan.y * 0.06}px) scale(${1 + (zoom - 1) * 0.04})`,
  };
  const selfHaloStyle: CSSProperties = {
    left: `${selfScreenX}px`,
    top: `${selfScreenY}px`,
    transform: `translate(-50%, -50%) scale(${0.92 + zoom * 0.08})`,
  };
  const showDebugControls = import.meta.env.DEV;

  const addDebugNode = () => {
    debugNodeCounterRef.current += 1;
    setDebugNodes((current) => [
      ...current,
      createDebugNode(
        debugNodeCounterRef.current,
        selectedModel,
        status.model_name ?? "",
      ),
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
      onClick={handleClick}
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
      <div
        className="pointer-events-none absolute h-28 w-28 rounded-full"
        style={{
          ...selfHaloStyle,
          background: scene.selfHaloGradient,
          filter: "blur(10px)",
        }}
      />
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
      <div className="absolute bottom-4 right-4 flex flex-col gap-2">
        <button
          type="button"
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
        className="pointer-events-none absolute text-center"
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
          {selfNode?.statusLabel || "connected"}
        </div>
      </div>
      {hoveredNode ? (
        <div
          ref={tooltipRef}
          className={cn(
            "pointer-events-none absolute w-[9rem] rounded-xl border px-3 py-2 shadow-2xl backdrop-blur",
            scene.tooltipClassName,
          )}
          style={tooltipStyle ?? { opacity: 0 }}
        >
          <div className="max-w-[16rem] text-xs font-medium">
            {hoveredNode.label}
          </div>
          <div className={cn("mt-0.5 text-[11px]", scene.tooltipSubtleTextClassName)}>
            {hoveredNode.subtitle}
          </div>
          <div
            className={cn(
              "mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]",
              scene.tooltipSubtleTextClassName,
            )}
          >
            <div className={scene.tooltipMutedTextClassName}>Role</div>
            <div>{hoveredNode.role}</div>
            <div className={scene.tooltipMutedTextClassName}>Status</div>
            <div>{hoveredNode.statusLabel}</div>
            <div className={scene.tooltipMutedTextClassName}>VRAM</div>
            <div>{hoveredNode.vramLabel}</div>
            <div className={scene.tooltipMutedTextClassName}>Latency</div>
            <div>{hoveredNode.latencyLabel}</div>
            {!(hoveredNode.role === "Client" && hoveredNode.modelLabel === "API-only") ? (
              <>
                <div className={scene.tooltipMutedTextClassName}>Model</div>
                <div className="truncate" title={hoveredNode.modelLabel}>
                  {hoveredNode.modelLabel}
                </div>
              </>
            ) : null}
            <div className={scene.tooltipMutedTextClassName}>Compute</div>
            <div>{hoveredNode.gpuLabel}</div>
          </div>
          <div
            className={cn(
              "mt-2 border-t pt-2 text-[10px]",
              scene.tooltipDividerClassName,
            )}
          >
            <div className="truncate" title={hoveredNode.id}>
              {hoveredNode.id}
            </div>
            {hoveredNode.hostname && hoveredNode.hostname !== hoveredNode.label ? (
              <div className="truncate" title={hoveredNode.hostname}>
                {hoveredNode.hostname}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
