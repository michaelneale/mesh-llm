import { LIVE_NODE_STATE_LABELS } from "./status-types";
import type {
  LiveNodeState,
  MeshModel,
  Ownership,
  Peer,
  StatusPayload,
  ThemeMode,
} from "./status-types";
import type { TopologyNode } from "./topology-types";

type GpuInventory = Array<{ vram_bytes: number }>;

export function modelDisplayName(model?: MeshModel | null) {
  if (!model) return "";
  return model.display_name || model.name;
}

export function modelRefLabel(name: string) {
  return name || "";
}

export function peerAssignedModels(peer: Peer): string[] {
  return peer.serving_models?.filter(Boolean) ?? [];
}

export function peerRoutableModels(peer: Peer): string[] {
  const hosted = peer.hosted_models?.filter(Boolean) ?? [];
  if (peer.hosted_models_known === false) {
    return hosted.length ? hosted : peerAssignedModels(peer);
  }
  return hosted;
}

export function localRoutableModels(status: StatusPayload | null): string[] {
  if (!status || status.node_state === "client") return [];
  const hosted = status.hosted_models?.filter(Boolean) ?? [];
  if (hosted.length > 0) return hosted;
  const serving = status.serving_models?.filter(Boolean) ?? [];
  if (serving.length > 0) return serving;
  return status.model_name ? [status.model_name] : [];
}

export function peerPrimaryModel(peer: Peer): string {
  return peerRoutableModels(peer)[0] ?? peerAssignedModels(peer)[0] ?? "";
}

export function normalizeVramGb(vramGb?: number | null) {
  return Math.max(0, vramGb ?? 0);
}

export function overviewVramGb(isClient: boolean, vramGb?: number | null) {
  if (isClient) return 0;
  return normalizeVramGb(vramGb);
}

export function gpuInventoryVramGb(gpus?: GpuInventory | null) {
  const bytes =
    gpus?.reduce((sum, gpu) => {
      const vramBytes = Number(gpu.vram_bytes);
      return sum + (Number.isFinite(vramBytes) && vramBytes > 0 ? vramBytes : 0);
    }, 0) ?? 0;
  return bytes > 0 ? bytes / 1024 ** 3 : null;
}

export function displayVramGb(
  isClient: boolean,
  capacityVramGb?: number | null,
  gpus?: GpuInventory | null,
) {
  if (isClient) return 0;
  return gpuInventoryVramGb(gpus) ?? overviewVramGb(false, capacityVramGb);
}

function assertLiveNodeState(state: LiveNodeState | undefined | null): LiveNodeState | null {
  if (!state || !(state in LIVE_NODE_STATE_LABELS)) {
    console.warn("Invalid or missing live node state:", state);
    return null;
  }
  return state;
}

export function formatLiveNodeState(state: LiveNodeState | undefined | null): string {
  const validated = assertLiveNodeState(state);
  if (!validated) return "Unknown";
  return LIVE_NODE_STATE_LABELS[validated];
}

export function meshGpuVram(status: StatusPayload | null) {
  if (!status) return 0;
  return (
    displayVramGb(status.node_state === "client", status.my_vram_gb, status.gpus) +
    (status.peers || []).reduce(
      (sum, peer) => sum + displayVramGb(peer.state === "client", peer.vram_gb, peer.gpus),
      0,
    )
  );
}

export function ownershipTone(status?: string): "good" | "warn" | "bad" | "neutral" {
  switch (status) {
    case "verified":
      return "good";
    case "expired":
    case "untrusted_owner":
      return "warn";
    case "invalid_signature":
    case "mismatched_node_id":
    case "revoked_owner":
    case "revoked_cert":
    case "revoked_node_id":
      return "bad";
    default:
      return "neutral";
  }
}

export function ownershipStatusLabel(status?: string) {
  if (!status) return "Unknown";
  return status
    .split("_")
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(" ");
}

export function shortIdentity(value?: string | null, size = 12) {
  if (!value) return "n/a";
  return value.length <= size ? value : value.slice(0, size);
}

export function ownershipPrimaryLabel(owner?: Ownership | null) {
  if (!owner) return "Unsigned";
  if (owner.node_label) return owner.node_label;
  if (owner.owner_id) return shortIdentity(owner.owner_id, 16);
  return ownershipStatusLabel(owner.status);
}

export function formatLatency(value?: number | null) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const ms = Math.round(Number(value));
  if (ms <= 0) return "<1 ms";
  return `${ms} ms`;
}

export function topologyStatusTone(
  state: LiveNodeState | undefined | null,
): "good" | "info" | "warn" | "bad" | "neutral" {
  const validated = assertLiveNodeState(state);
  if (!validated) return "neutral";
  switch (validated) {
    case "serving":
      return "good";
    case "client":
      return "info";
    case "loading":
      return "warn";
    case "standby":
      return "neutral";
  }
}

export function topologyStatusTooltip(state: LiveNodeState | undefined | null): string {
  const validated = assertLiveNodeState(state);
  if (!validated) return "Node state unavailable";
  switch (validated) {
    case "serving":
      return "Actively serving a model.";
    case "loading":
      return "Initializing model work before it can serve requests.";
    case "client":
      return "Sends requests, but does not contribute VRAM.";
    case "standby":
      return "Connected, but not currently serving a model.";
  }
}

export function modelStatusTooltip(status?: string) {
  if (status === "warm") {
    return "Loaded and serving in the mesh.";
  }
  if (status === "cold") {
    return "Downloaded locally, but not currently serving.";
  }
  return "Current model availability in the mesh.";
}

export function uniqueModels(...groups: Array<string[] | undefined>): string[] {
  return [
    ...new Set(
      groups
        .flatMap((group) => group ?? [])
        .filter((model) => !!model && model !== "(idle)"),
    ),
  ];
}

export function formatGpuMemory(bytes?: number | null) {
  if (!bytes || !Number.isFinite(bytes)) return "Unknown";
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(0)} GB`;
}

export function trimGpuVendor(name: string) {
  return name
    .replace(/^NVIDIA GeForce\s+/i, "")
    .replace(/^NVIDIA Quadro\s+/i, "")
    .replace(/^NVIDIA\s+/i, "")
    .replace(/^AMD Radeon\s+/i, "")
    .replace(/^AMD\s+/i, "")
    .replace(/^Intel Arc\s+/i, "")
    .replace(/^Intel\s+/i, "")
    .replace(/^Apple\s+/i, "")
    .trim();
}

export function topologyNodeRole(node: Pick<TopologyNode, "client" | "host">): string {
  if (node.client) return "Client";
  if (node.host) return "Host";
  return "Worker";
}

export function readThemeMode(storageKey: string): ThemeMode {
  if (typeof window === "undefined") return "auto";
  const stored = window.localStorage.getItem(storageKey);
  return stored === "light" || stored === "dark" || stored === "auto"
    ? stored
    : "auto";
}

export function applyThemeMode(mode: ThemeMode) {
  if (typeof window === "undefined") return;
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const dark = mode === "dark" || (mode === "auto" && media.matches);
  document.documentElement.classList.toggle("dark", dark);
  document.documentElement.style.colorScheme =
    mode === "auto" ? "light dark" : dark ? "dark" : "light";
}
