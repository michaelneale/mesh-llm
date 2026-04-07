import {
  type CSSProperties,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
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
  ArrowLeft,
  Bot,
  Braces,
  ChevronDown,
  Check,
  Copy,
  Cpu,
  File,
  FolderTree,
  FileAudio,
  Gauge,
  Gpu,
  Hash,
  ImagePlus,
  Info,
  Laptop,
  Loader2,
  MessageSquarePlus,
  Maximize2,
  MemoryStick,
  Minimize2,
  Moon,
  Network,
  ExternalLink,
  Pencil,
  Paperclip,
  RotateCcw,
  Send,
  Server,
  Square,
  Sparkles,
  Brain,
  Boxes,
  TextCursorInput,
  Sun,
  Trash2,
  UserPlus,
  User,
  Wifi,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "./components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "./components/ui/alert";
import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  navigationMenuTriggerStyle,
} from "./components/ui/navigation-menu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "./components/ui/popover";
import { ScrollArea } from "./components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import { Separator } from "./components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./components/ui/table";
import { Textarea } from "./components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./components/ui/tooltip";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "./components/ui/sheet";
import { BrandIcon } from "./components/brand-icon";
import { MeshLlmWordmark } from "./components/mesh-llm-wordmark";
import {
  getAttachmentSendIssue,
  validateAttachmentFile,
} from "./lib/attachments";
import { createRafBatcher } from "./lib/streaming";
import { cn } from "./lib/utils";
import {
  TOPOLOGY_LAYOUT_OPTIONS,
  TOPOLOGY_LAYOUTS,
  isTopologyLayoutMode,
} from "./topology/layouts";
import { TOPOLOGY_NODE_WIDTH } from "./topology/layouts/constants";
import { estimateTopologyNodeHeight } from "./topology/layouts/elk";
import type {
  BucketedTopologyNode,
  PositionedTopologyNode,
  TopologyLayoutMode,
  TopologyNode,
  TopologyNodeInfo,
} from "./topology/layouts/types";
import githubBlackLogo from "./assets/icons/github-invertocat-black.svg";
import githubWhiteLogo from "./assets/icons/github-invertocat-white.svg";

const DOCS_URL = "https://docs.anarchai.org";
const FLY_DOMAINS = [
  "mesh-llm-console.fly.dev",
  "www.mesh-llm.com",
  "www.anarchai.org",
];

type MeshModel = {
  name: string;
  display_name?: string;
  status: "warm" | "cold" | string;
  node_count: number;
  mesh_vram_gb?: number;
  size_gb: number;
  architecture?: string;
  context_length?: number;
  quantization?: string;
  description?: string;
  multimodal?: boolean;
  multimodal_status?: "supported" | "none" | string;
  vision?: boolean;
  vision_status?: "supported" | "likely" | "none" | string;
  audio?: boolean;
  audio_status?: "supported" | "likely" | "none" | string;
  reasoning?: boolean;
  reasoning_status?: "supported" | "likely" | "none" | string;
  tool_use?: boolean;
  tool_use_status?: "supported" | "likely" | "none" | string;
  moe?: boolean;
  expert_count?: number;
  used_expert_count?: number;
  ranking_source?: string;
  ranking_origin?: string;
  ranking_prompt_count?: number;
  ranking_tokens?: number;
  ranking_layer_scope?: string;
  draft_model?: string;
  source_page_url?: string;
  fit_label?: string;
  fit_detail?: string;
  download_command?: string;
  run_command?: string;
  auto_command?: string;
  request_count?: number;
  last_active_secs_ago?: number;
  source_ref?: string;
  source_revision?: string;
  source_file?: string;
  active_nodes?: string[];
};

type ActivePeerRow = {
  id: string;
  latencyLabel: string;
  vramLabel: string;
  shareLabel: string;
};

type DetailPanelEntry =
  | { kind: "node"; nodeId: string }
  | { kind: "model"; modelName: string };

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
  privacyLimited: boolean;
};

function modelDisplayName(model?: MeshModel | null) {
  if (!model) return "";
  return model.display_name || model.name;
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

function visionBadge(model?: MeshModel | null) {
  if (!model) return null;
  if (model.vision) return { icon: "👁", title: "Vision — understands images" };
  if (model.vision_status === "likely") {
    return {
      icon: "👁?",
      title: "Vision likely — inferred from model metadata",
    };
  }
  return null;
}

function multimodalBadge(model?: MeshModel | null) {
  if (!model) return null;
  if (model.multimodal)
    return { icon: "🎛️", title: "Multimodal — supports media inputs" };
  return null;
}

function audioBadge(model?: MeshModel | null) {
  if (!model) return null;
  if (model.audio) return { icon: "🔊", title: "Audio — understands audio input" };
  if (model.audio_status === "likely") {
    return {
      icon: "🔊?",
      title: "Audio likely — inferred from model metadata",
    };
  }
  return null;
}

function reasoningBadge(model?: MeshModel | null) {
  if (!model) return null;
  if (model.reasoning) return { icon: "🧠", title: "Reasoning-oriented model" };
  if (model.reasoning_status === "likely") {
    return {
      icon: "🧠?",
      title: "Reasoning likely — inferred from model metadata",
    };
  }
  return null;
}

type Peer = {
  id: string;
  role: string;
  models: string[];
  available_models?: string[];
  requested_models?: string[];
  vram_gb: number;
  serving_models?: string[];
  hosted_models?: string[];
  hosted_models_known?: boolean;
  rtt_ms?: number | null;
  hostname?: string;
  is_soc?: boolean;
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
};

type StatusPayload = {
  version?: string;
  latest_version?: string | null;
  node_id: string;
  token: string;
  node_status: string;
  is_host: boolean;
  is_client: boolean;
  llama_ready: boolean;
  model_name: string;
  models?: string[];
  available_models?: string[];
  requested_models?: string[];
  serving_models?: string[];
  hosted_models?: string[];
  api_port: number;
  my_vram_gb: number;
  model_size_gb: number;
  mesh_name?: string | null;
  peers: Peer[];
  inflight_requests: number;
  launch_pi?: string | null;
  launch_goose?: string | null;
  nostr_discovery?: boolean;
  my_hostname?: string;
  my_is_soc?: boolean;
  gpus?: { name: string; vram_bytes: number; bandwidth_gbps?: number }[];
};

type ModelsPayload = {
  mesh_models: MeshModel[];
};

type ChatAttachmentKind = "image" | "audio" | "file";
type ChatAttachmentStatus = "pending" | "uploading" | "failed";

type ChatAttachment = {
  id: string;
  kind: ChatAttachmentKind;
  dataUrl: string;
  mimeType: string;
  fileName?: string;
  status?: ChatAttachmentStatus;
  error?: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  model?: string;
  stats?: string;
  error?: boolean;
  /** Legacy attachment fields retained for persisted chat compatibility */
  image?: string;
  audio?: {
    dataUrl: string;
    mimeType: string;
    fileName?: string;
  };
  attachments?: ChatAttachment[];
};

type ChatConversation = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
};

type ChatState = {
  conversations: ChatConversation[];
  activeConversationId: string;
};

type ModelServingStat = {
  nodes: number;
  vramGb: number;
};

type TopSection = "dashboard" | "chat";
type AppRoute = {
  section: TopSection;
  chatId: string | null;
};

type ThemeMode = "auto" | "light" | "dark";

const THEME_STORAGE_KEY = "mesh-llm-theme";
const CHAT_CLIENT_ID_STORAGE_KEY = "mesh-llm-chat-client-id";
const DEFAULT_CHAT_TITLE = "New chat";
const CHAT_DB_NAME = "mesh-llm-chat-db";
const CHAT_DB_STORE = "state";
const CHAT_DB_KEY = "chat-state";
const CHAT_SAVE_DEBOUNCE_MS = 500;
const CHAT_MAX_CONVERSATIONS = 80;
const CHAT_MAX_MESSAGES_PER_CONVERSATION = 240;
const CHAT_MAX_TEXT_CHARS = 12000;

function peerAssignedModels(peer: Peer): string[] {
  return peer.serving_models?.filter(Boolean) ?? [];
}

function peerRoutableModels(peer: Peer): string[] {
  const hosted = peer.hosted_models?.filter(Boolean) ?? [];
  if (peer.hosted_models_known === false)
    return hosted.length ? hosted : peerAssignedModels(peer);
  return hosted;
}

function localRoutableModels(status: StatusPayload | null): string[] {
  if (!status || status.is_client) return [];
  const hosted = status.hosted_models?.filter(Boolean) ?? [];
  if (hosted.length > 0) return hosted;
  const serving = status.serving_models?.filter(Boolean) ?? [];
  if (serving.length > 0) return serving;
  return status.model_name ? [status.model_name] : [];
}

function peerPrimaryModel(peer: Peer): string {
  return peerRoutableModels(peer)[0] ?? peerAssignedModels(peer)[0] ?? "";
}

function peerStatusLabel(peer: Peer): string {
  if (peer.role === "Client") return "Client";
  if (peerRoutableModels(peer).some((model) => model !== "(idle)"))
    return "Serving";
  if (peerAssignedModels(peer).some((model) => model !== "(idle)"))
    return "Assigned";
  if (peer.role === "Host") return "Host";
  return "Idle";
}

function topologyNodeRole(node: Pick<TopologyNode, "client" | "host" | "serving">): string {
  if (node.client) return "Client";
  if (node.host) return "Host";
  if (node.serving && node.serving !== "(idle)") return "Worker";
  return "Idle";
}

function nodeRoleTone(role: string): "good" | "info" | "neutral" {
  if (role === "Host") return "good";
  if (role === "Worker" || role === "Client") return "info";
  return "neutral";
}

function uniqueModels(...groups: Array<string[] | undefined>): string[] {
  return [
    ...new Set(
      groups
        .flatMap((group) => group ?? [])
        .filter((model) => !!model && model !== "(idle)"),
    ),
  ];
}

function formatGpuMemory(bytes?: number | null) {
  if (!bytes || !Number.isFinite(bytes)) return "Unknown";
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(0)} GB`;
}

function trimGpuVendor(name: string) {
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

function sectionFromPathname(pathname: string): TopSection | null {
  if (pathname === "/dashboard" || pathname === "/dashboard/")
    return "dashboard";
  if (
    pathname === "/chat" ||
    pathname === "/chat/" ||
    pathname.startsWith("/chat/")
  )
    return "chat";
  return null;
}

function isMobileViewport(): boolean {
  return typeof window !== "undefined" && window.innerWidth < 768;
}

function readRouteFromLocation(): AppRoute {
  if (typeof window === "undefined")
    return { section: "dashboard", chatId: null };
  const pathname = window.location.pathname;
  if (pathname === "/dashboard" || pathname === "/dashboard/")
    return { section: "dashboard", chatId: null };
  if (pathname === "/chat" || pathname === "/chat/")
    return { section: "chat", chatId: null };
  if (pathname.startsWith("/chat/")) {
    const raw = pathname.slice("/chat/".length);
    const chatId = raw ? decodeURIComponent(raw.split("/")[0]) : null;
    return { section: "chat", chatId };
  }
  // Default: chat on mobile, dashboard on desktop
  return { section: isMobileViewport() ? "chat" : "dashboard", chatId: null };
}

function pathnameForRoute(route: AppRoute): string {
  if (route.section === "dashboard") return "/dashboard";
  return route.chatId ? `/chat/${encodeURIComponent(route.chatId)}` : "/chat";
}

function pushRoute(route: AppRoute) {
  if (typeof window === "undefined") return;
  const nextPath = pathnameForRoute(route);
  if (window.location.pathname !== nextPath)
    window.history.pushState({}, "", nextPath);
}

function replaceRoute(route: AppRoute) {
  if (typeof window === "undefined") return;
  const nextPath = pathnameForRoute(route);
  if (window.location.pathname !== nextPath)
    window.history.replaceState({}, "", nextPath);
}

function readThemeMode(): ThemeMode {
  if (typeof window === "undefined") return "auto";
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return stored === "light" || stored === "dark" || stored === "auto"
    ? stored
    : "auto";
}

function applyThemeMode(mode: ThemeMode) {
  if (typeof window === "undefined") return;
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const dark = mode === "dark" || (mode === "auto" && media.matches);
  document.documentElement.classList.toggle("dark", dark);
  document.documentElement.style.colorScheme =
    mode === "auto" ? "light dark" : dark ? "dark" : "light";
}

function randomId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function")
    return crypto.randomUUID();
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function readOrCreateChatClientId(): string {
  if (typeof window === "undefined") return randomId();
  const stored = window.localStorage.getItem(CHAT_CLIENT_ID_STORAGE_KEY);
  if (stored && stored.trim()) return stored;
  const created = randomId();
  window.localStorage.setItem(CHAT_CLIENT_ID_STORAGE_KEY, created);
  return created;
}

function deriveConversationTitle(input: string): string {
  const compact = input.replace(/\s+/g, " ").trim();
  if (!compact) return DEFAULT_CHAT_TITLE;
  return compact.length > 52 ? `${compact.slice(0, 52).trimEnd()}...` : compact;
}

function createConversation(
  seed?: Partial<Pick<ChatConversation, "title" | "messages">>,
): ChatConversation {
  const now = Date.now();
  return {
    id: randomId(),
    title: seed?.title || DEFAULT_CHAT_TITLE,
    createdAt: now,
    updatedAt: now,
    messages: seed?.messages || [],
  };
}

function createInitialChatState(): ChatState {
  return { conversations: [], activeConversationId: "" };
}

function clampText(
  text: string | undefined,
  maxChars = CHAT_MAX_TEXT_CHARS,
): string | undefined {
  if (typeof text !== "string") return undefined;
  if (!text.length) return undefined;
  return text.length > maxChars
    ? `${text.slice(0, maxChars).trimEnd()}...`
    : text;
}

function parseDataUrl(dataUrl: string): { mimeType: string; base64: string } | null {
  const match = /^data:([^;,]+);base64,(.+)$/s.exec(dataUrl);
  if (!match) return null;
  return { mimeType: match[1], base64: match[2] };
}

function sanitizeAttachment(raw: unknown): ChatAttachment | null {
  if (!raw || typeof raw !== "object") return null;
  const item = raw as Record<string, unknown>;
  const kind = item.kind;
  const dataUrl = item.dataUrl;
  const mimeType = item.mimeType;
  if (
    (kind !== "image" && kind !== "audio" && kind !== "file") ||
    typeof dataUrl !== "string" ||
    typeof mimeType !== "string"
  ) {
    return null;
  }
  return {
    id: typeof item.id === "string" && item.id ? item.id : randomId(),
    kind,
    dataUrl,
    mimeType,
    fileName: typeof item.fileName === "string" ? item.fileName : undefined,
    status:
      item.status === "pending" ||
      item.status === "uploading" ||
      item.status === "failed"
        ? item.status
        : undefined,
    error: typeof item.error === "string" ? item.error : undefined,
  };
}

function messageAttachments(message: ChatMessage): ChatAttachment[] {
  if (Array.isArray(message.attachments) && message.attachments.length > 0) {
    return message.attachments;
  }
  const attachments: ChatAttachment[] = [];
  if (message.image) {
    attachments.push({
      id: `${message.id}-image`,
      kind: "image",
      dataUrl: message.image,
      mimeType: parseDataUrl(message.image)?.mimeType || "image/jpeg",
      fileName: "image.jpg",
    });
  }
  if (message.audio) {
    attachments.push({
      id: `${message.id}-audio`,
      kind: "audio",
      dataUrl: message.audio.dataUrl,
      mimeType: message.audio.mimeType,
      fileName: message.audio.fileName,
    });
  }
  return attachments;
}

function sanitizeMessages(raw: unknown): ChatMessage[] {
  if (!Array.isArray(raw)) return [];
  const sanitized = raw.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const role = (item as { role?: unknown }).role;
    const content = (item as { content?: unknown }).content;
    if (
      (role !== "user" && role !== "assistant") ||
      typeof content !== "string"
    )
      return [];
    const safeRole: ChatMessage["role"] = role;
    const attachments = Array.isArray((item as { attachments?: unknown }).attachments)
      ? ((item as { attachments: unknown[] }).attachments
          .map(sanitizeAttachment)
          .filter(Boolean) as ChatAttachment[])
      : [];
    const legacyImage =
      typeof (item as { image?: unknown }).image === "string"
        ? (item as { image: string }).image
        : undefined;
    const legacyAudio =
      typeof (item as { audio?: unknown }).audio === "object" &&
      (item as { audio?: unknown }).audio &&
      typeof ((item as { audio: { dataUrl?: unknown } }).audio.dataUrl) ===
        "string" &&
      typeof ((item as { audio: { mimeType?: unknown } }).audio.mimeType) ===
        "string"
        ? {
            dataUrl: (item as { audio: { dataUrl: string } }).audio.dataUrl,
            mimeType: (item as { audio: { mimeType: string } }).audio.mimeType,
            fileName:
              typeof ((item as { audio: { fileName?: unknown } }).audio.fileName) ===
              "string"
                ? (item as { audio: { fileName: string } }).audio.fileName
                : undefined,
          }
        : undefined;
    if (attachments.length === 0) {
      if (legacyImage) {
        attachments.push({
          id: randomId(),
          kind: "image",
          dataUrl: legacyImage,
          mimeType: parseDataUrl(legacyImage)?.mimeType || "image/jpeg",
          fileName: "image.jpg",
        });
      }
      if (legacyAudio) {
        attachments.push({
          id: randomId(),
          kind: "audio",
          dataUrl: legacyAudio.dataUrl,
          mimeType: legacyAudio.mimeType,
          fileName: legacyAudio.fileName,
        });
      }
    }
    return [
      {
        id:
          typeof (item as { id?: unknown }).id === "string"
            ? (item as { id: string }).id
            : randomId(),
        role: safeRole,
        content: clampText(content, CHAT_MAX_TEXT_CHARS) ?? "",
        reasoning: clampText(
          typeof (item as { reasoning?: unknown }).reasoning === "string"
            ? (item as { reasoning: string }).reasoning
            : undefined,
        ),
        model: clampText(
          typeof (item as { model?: unknown }).model === "string"
            ? (item as { model: string }).model
            : undefined,
          256,
        ),
        stats: clampText(
          typeof (item as { stats?: unknown }).stats === "string"
            ? (item as { stats: string }).stats
            : undefined,
          256,
        ),
        error: Boolean((item as { error?: unknown }).error),
        image: legacyImage,
        audio: legacyAudio,
        attachments: attachments.length > 0 ? attachments : undefined,
      },
    ];
  });
  return sanitized.slice(-CHAT_MAX_MESSAGES_PER_CONVERSATION);
}

function sanitizeChatState(raw: unknown): ChatState {
  const fallback = createInitialChatState();
  if (!raw || typeof raw !== "object") return fallback;

  const parsed = raw as {
    conversations?: unknown;
    activeConversationId?: unknown;
  };
  if (!Array.isArray(parsed.conversations)) return fallback;

  const now = Date.now();
  const conversations = parsed.conversations
    .flatMap((item) => {
      if (!item || typeof item !== "object") return [];
      const obj = item as Record<string, unknown>;
      const id = typeof obj.id === "string" && obj.id ? obj.id : randomId();
      const titleRaw =
        typeof obj.title === "string" && obj.title.trim()
          ? obj.title
          : DEFAULT_CHAT_TITLE;
      return [
        {
          id,
          title: clampText(titleRaw, 140) ?? DEFAULT_CHAT_TITLE,
          createdAt: typeof obj.createdAt === "number" ? obj.createdAt : now,
          updatedAt: typeof obj.updatedAt === "number" ? obj.updatedAt : now,
          messages: sanitizeMessages(obj.messages),
        },
      ];
    })
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, CHAT_MAX_CONVERSATIONS);

  return {
    conversations,
    // On reopen, always start at the most recently updated chat.
    activeConversationId: conversations[0]?.id ?? "",
  };
}

function openChatDb(): Promise<IDBDatabase | null> {
  if (typeof window === "undefined" || !("indexedDB" in window))
    return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(CHAT_DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(CHAT_DB_STORE))
        db.createObjectStore(CHAT_DB_STORE, { keyPath: "id" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error("Failed to open chat DB"));
  });
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

function transactionToPromise(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () =>
      reject(tx.error ?? new Error("IndexedDB transaction failed"));
    tx.onabort = () =>
      reject(tx.error ?? new Error("IndexedDB transaction aborted"));
  });
}

async function readChatStateFromDb(): Promise<ChatState | null> {
  let db: IDBDatabase | null = null;
  try {
    db = await openChatDb();
    if (!db) return null;
    const tx = db.transaction(CHAT_DB_STORE, "readonly");
    const store = tx.objectStore(CHAT_DB_STORE);
    const record = await requestToPromise(
      store.get(CHAT_DB_KEY) as IDBRequest<
        { id: string; state?: unknown } | undefined
      >,
    );
    await transactionToPromise(tx);
    if (!record?.state) return null;
    return sanitizeChatState(record.state);
  } catch (err) {
    console.warn("Failed to read chat state from IndexedDB:", err);
    return null;
  } finally {
    db?.close();
  }
}

async function writeChatStateToDb(state: ChatState): Promise<void> {
  let db: IDBDatabase | null = null;
  try {
    db = await openChatDb();
    if (!db) return;
    const tx = db.transaction(CHAT_DB_STORE, "readwrite");
    const store = tx.objectStore(CHAT_DB_STORE);
    store.put({ id: CHAT_DB_KEY, state, updatedAt: Date.now() });
    await transactionToPromise(tx);
  } catch (err) {
    console.warn("Failed to write chat state to IndexedDB:", err);
  } finally {
    db?.close();
  }
}

async function loadPersistedChatState(): Promise<ChatState> {
  const fromDb = await readChatStateFromDb();
  return fromDb ?? createInitialChatState();
}

function findLastUserMessageIndex(messages: ChatMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === "user") return i;
  }
  return -1;
}

function updateConversationList(
  conversations: ChatConversation[],
  conversationId: string,
  updater: (conversation: ChatConversation) => ChatConversation,
): ChatConversation[] {
  const index = conversations.findIndex(
    (conversation) => conversation.id === conversationId,
  );
  if (index < 0) return conversations;
  const updated = updater(conversations[index]);
  if (index === 0) return [updated, ...conversations.slice(1)];
  return [
    updated,
    ...conversations.slice(0, index),
    ...conversations.slice(index + 1),
  ];
}

export function App() {
  const [section, setSection] = useState<TopSection>(
    () => readRouteFromLocation().section,
  );
  const [routedChatId, setRoutedChatId] = useState<string | null>(
    () => readRouteFromLocation().chatId,
  );
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => readThemeMode());
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [modelsPayload, setModelsPayload] = useState<ModelsPayload | null>(
    null,
  );
  const [modelsLoading, setModelsLoading] = useState(false);
  const [chatState, setChatState] = useState<ChatState>(() =>
    createInitialChatState(),
  );
  const [chatStateHydrated, setChatStateHydrated] = useState(false);
  const [input, setInput] = useState("");
  const [pendingAttachments, setPendingAttachments] = useState<ChatAttachment[]>(
    [],
  );
  const [selectedModel, setSelectedModel] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [composerError, setComposerError] = useState<string | null>(null);
  const [reasoningOpen, setReasoningOpen] = useState<Record<string, boolean>>(
    {},
  );
  const chatClientIdRef = useRef<string>(readOrCreateChatClientId());
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const currentAbortRef = useRef<AbortController | null>(null);
  const queuedInputRef = useRef<string | null>(null);
  const [queuedText, setQueuedText] = useState<string | null>(null);
  const activeConversationId = chatState.activeConversationId;
  const conversations = chatState.conversations;
  const activeConversation = useMemo(
    () =>
      conversations.find(
        (conversation) => conversation.id === activeConversationId,
      ) ?? conversations[0],
    [conversations, activeConversationId],
  );
  const messages = activeConversation?.messages ?? [];
  const lastMessageId = messages[messages.length - 1]?.id ?? "";
  const meshModels = modelsPayload?.mesh_models ?? [];

  const warmModels = useMemo(() => {
    const list = meshModels
      .filter((m) => m.status === "warm")
      .map((m) => m.name);
    if (!list.length && status?.model_name) list.push(status.model_name);
    return list;
  }, [meshModels, status?.model_name]);
  const modelStatsByName = useMemo<Record<string, ModelServingStat>>(() => {
    const stats: Record<string, ModelServingStat> = {};
    for (const model of warmModels) stats[model] = { nodes: 0, vramGb: 0 };
    if (!status) return stats;

    const addServingNode = (modelName: string, vramGb: number) => {
      if (!stats[modelName]) stats[modelName] = { nodes: 0, vramGb: 0 };
      stats[modelName].nodes += 1;
      stats[modelName].vramGb += Math.max(0, vramGb || 0);
    };

    for (const model of new Set(localRoutableModels(status))) {
      if (model && model !== "(idle)") addServingNode(model, status.my_vram_gb);
    }
    for (const peer of status.peers ?? []) {
      if (peer.role === "Client") continue;
      for (const model of new Set(peerRoutableModels(peer))) {
        if (model && model !== "(idle)") addServingNode(model, peer.vram_gb);
      }
    }

    for (const model of meshModels) {
      if (!stats[model.name]) continue;
      if (stats[model.name].nodes === 0)
        stats[model.name].nodes = Math.max(0, model.node_count || 0);
    }

    return stats;
  }, [status, warmModels, meshModels]);
  const selectedChatModel =
    selectedModel || warmModels[0] || status?.model_name || "";
  const visionModels = useMemo(() => {
    const set = new Set<string>();
    for (const m of meshModels) {
      if (m.vision) set.add(m.name);
    }
    return set;
  }, [meshModels]);
  const audioModels = useMemo(() => {
    const set = new Set<string>();
    for (const m of meshModels) {
      if (m.audio) set.add(m.name);
    }
    return set;
  }, [meshModels]);
  const multimodalModels = useMemo(() => {
    const set = new Set<string>();
    for (const m of meshModels) {
      if (m.multimodal) set.add(m.name);
    }
    return set;
  }, [meshModels]);
  const selectedModelVision = useMemo(() => {
    if (selectedModel) return visionModels.has(selectedModel);
    return meshModels.some((m) => m.status === "warm" && m.vision);
  }, [meshModels, selectedModel, visionModels]);
  const selectedModelAudio = useMemo(() => {
    if (selectedModel) return audioModels.has(selectedModel);
    return meshModels.some((m) => m.status === "warm" && m.audio);
  }, [audioModels, meshModels, selectedModel]);
  const selectedModelMultimodal = useMemo(() => {
    if (selectedModel) return multimodalModels.has(selectedModel);
    return meshModels.some((m) => m.status === "warm" && m.multimodal);
  }, [meshModels, multimodalModels, selectedModel]);
  const pendingKinds = useMemo(
    () => new Set(pendingAttachments.map((attachment) => attachment.kind)),
    [pendingAttachments],
  );
  const attachmentSendIssue = useMemo(() => {
    if (!pendingAttachments.length || !status) return null;
    return getAttachmentSendIssue({
      pendingKinds,
      selectedModel,
      warmModels,
      visionModels,
      audioModels,
      multimodalModels,
    });
  }, [
    audioModels,
    multimodalModels,
    pendingAttachments.length,
    pendingKinds,
    selectedModel,
    status,
    visionModels,
    warmModels,
  ]);
  const meshModelByName = useMemo(() => {
    const entries = meshModels.map((model) => [model.name, model] as const);
    return Object.fromEntries(entries) as Record<string, MeshModel>;
  }, [meshModels]);
  const selectedModelStat = selectedChatModel
    ? modelStatsByName[selectedChatModel]
    : undefined;
  const selectedModelNodeCount = selectedModelStat
    ? selectedModelStat.nodes
    : null;
  const selectedModelVramGb = selectedModelStat
    ? selectedModelStat.vramGb
    : null;

  const inviteWithModelName =
    selectedModel || warmModels[0] || status?.model_name || "";
  const inviteWithModelCommand = useMemo(() => {
    const token = status?.token ?? "";
    return token && inviteWithModelName
      ? `mesh-llm --join ${token} --model ${inviteWithModelName}`
      : "";
  }, [inviteWithModelName, status?.token]);
  const inviteToken = status?.token ?? "";
  const inviteClientCommand = useMemo(() => {
    const token = status?.token ?? "";
    return token ? `mesh-llm --client --join ${token}` : "";
  }, [status?.token]);
  const isLocalhost =
    typeof window !== "undefined" &&
    (window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1");
  const isFlyHosted =
    typeof window !== "undefined" &&
    FLY_DOMAINS.includes(window.location.hostname);
  const apiDirectUrl = useMemo(() => {
    if (!isLocalhost) return "";
    const port = status?.api_port ?? 9337;
    return `http://127.0.0.1:${port}/v1`;
  }, [status?.api_port, isLocalhost]);
  const modelCatalogKey = useMemo(() => {
    if (!status) return "";
    const local = [
      status.model_name,
      ...(status.models ?? []),
      ...(status.available_models ?? []),
      ...(status.requested_models ?? []),
      ...(status.serving_models ?? []),
      ...(status.hosted_models ?? []),
    ].join(",");
    const peers = [...(status.peers ?? [])]
      .map((peer) =>
        [
          peer.id,
          ...(peer.models ?? []),
          ...(peer.available_models ?? []),
          ...(peer.requested_models ?? []),
          ...(peer.serving_models ?? []),
          ...(peer.hosted_models ?? []),
        ].join(","),
      )
      .sort()
      .join("|");
    return `${status.node_id}::${local}::${peers}`;
  }, [status]);

  useEffect(() => {
    if (!warmModels.length) return;
    if (
      !selectedModel ||
      (selectedModel !== "auto" && !warmModels.includes(selectedModel))
    )
      setSelectedModel(warmModels.length > 1 ? "auto" : warmModels[0]);
  }, [warmModels, selectedModel]);

  useEffect(() => {
    applyThemeMode(themeMode);
    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode);
  }, [themeMode]);

  useEffect(() => {
    let cancelled = false;
    void loadPersistedChatState()
      .then((loaded) => {
        if (cancelled) return;
        setChatState(sanitizeChatState(loaded));
      })
      .catch((err) => {
        console.warn("Failed to hydrate chat state:", err);
      })
      .finally(() => {
        if (!cancelled) setChatStateHydrated(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!chatStateHydrated) return;
    const timeout = window.setTimeout(() => {
      void writeChatStateToDb(sanitizeChatState(chatState));
    }, CHAT_SAVE_DEBOUNCE_MS);
    return () => window.clearTimeout(timeout);
  }, [chatState, chatStateHydrated]);

  useEffect(() => {
    if (themeMode !== "auto") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => applyThemeMode("auto");
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [themeMode]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const current = sectionFromPathname(window.location.pathname);
    if (current == null) replaceRoute({ section: "dashboard", chatId: null });

    const onPopState = () => {
      const route = readRouteFromLocation();
      setSection(route.section);
      setRoutedChatId(route.chatId);
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    if (section !== "chat" || !chatStateHydrated) return;
    if (routedChatId) {
      if (
        conversations.some((conversation) => conversation.id === routedChatId)
      ) {
        setChatState((prev) =>
          prev.activeConversationId === routedChatId
            ? prev
            : { ...prev, activeConversationId: routedChatId },
        );
      } else {
        const fallbackId = conversations[0]?.id ?? null;
        setRoutedChatId(fallbackId);
        replaceRoute({ section: "chat", chatId: fallbackId });
        setChatState((prev) => ({
          ...prev,
          activeConversationId: fallbackId ?? "",
        }));
      }
      return;
    }
    if (activeConversation?.id) {
      setRoutedChatId(activeConversation.id);
      replaceRoute({ section: "chat", chatId: activeConversation.id });
    }
  }, [
    section,
    chatStateHydrated,
    routedChatId,
    conversations,
    activeConversation?.id,
  ]);

  useEffect(() => {
    let stop = false;
    let statusEvents: EventSource | null = null;
    let reconnectTimer: number | null = null;
    let retryMs = 1000;
    const MAX_RETRY_MS = 15000;
    const reconnectStatusMessage =
      "Trying to reconnect automatically. Live updates will resume shortly.";

    const clearReconnectTimer = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const closeStatusEvents = () => {
      if (!statusEvents) return;
      statusEvents.onopen = null;
      statusEvents.onmessage = null;
      statusEvents.onerror = null;
      statusEvents.close();
      statusEvents = null;
    };

    const loadStatus = () => {
      fetch("/api/status")
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json() as Promise<StatusPayload>;
        })
        .then((data) => {
          if (stop) return;
          setStatus(data);
          setStatusError(null);
        })
        .catch((err: Error) => {
          if (!stop) {
            setStatusError(reconnectStatusMessage);
            console.warn("Failed to fetch /api/status:", err.message);
          }
        });
    };

    const connectStatusEvents = () => {
      if (stop || statusEvents) return;

      let source: EventSource;
      try {
        source = new EventSource("/api/events");
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "failed to create EventSource";
        console.warn("Failed to connect status stream:", message);
        scheduleReconnect();
        return;
      }

      statusEvents = source;
      source.onopen = () => {
        if (stop) return;
        retryMs = 1000;
        setStatusError(null);
        loadStatus();
      };
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as StatusPayload;
          setStatus(payload);
          setStatusError(null);
        } catch {
          // ignore malformed status event
        }
      };
      source.onerror = () => {
        if (stop) return;
        scheduleReconnect();
      };
    };

    const scheduleReconnect = () => {
      if (stop || reconnectTimer !== null) return;
      setStatusError(reconnectStatusMessage);
      console.warn("Connection lost. Reconnecting...");
      closeStatusEvents();
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        connectStatusEvents();
        retryMs = Math.min(retryMs * 2, MAX_RETRY_MS);
      }, retryMs);
    };

    loadStatus();
    connectStatusEvents();

    return () => {
      stop = true;
      clearReconnectTimer();
      closeStatusEvents();
    };
  }, []);

  useEffect(() => {
    if (!modelCatalogKey) {
      setModelsPayload(null);
      setModelsLoading(false);
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    setModelsLoading(true);
    fetch("/api/models", { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<ModelsPayload>;
      })
      .then((data) => {
        if (cancelled) return;
        setModelsPayload(data);
      })
      .catch((err: Error) => {
        if (cancelled || err.name === "AbortError") return;
        console.warn("Failed to fetch /api/models:", err.message);
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [modelCatalogKey]);

  useEffect(() => {
    const el = chatScrollRef.current;
    if (!el) return;
    if (!activeConversationId && !lastMessageId && !isSending) return;
    // Only auto-scroll if the user is already near the bottom.
    // This lets them scroll up to read earlier content while streaming.
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceFromBottom < 80) {
      el.scrollTop = el.scrollHeight;
    }
  }, [activeConversationId, isSending, lastMessageId]);

  useEffect(() => () => currentAbortRef.current?.abort(), []);

  // Drain queued message once streaming finishes and state is fresh.
  useEffect(() => {
    if (isSending) return;
    const queued = queuedInputRef.current;
    if (queued == null) return;
    queuedInputRef.current = null;
    setQueuedText(null);
    void sendMessage(queued);
  }, [isSending]); // eslint-disable-line react-hooks/exhaustive-deps

  const canChat =
    !!status &&
    (status.llama_ready || (status.is_client && warmModels.length > 0));
  const canRegenerate =
    canChat &&
    !!activeConversation &&
    findLastUserMessageIndex(activeConversation.messages) >= 0;

  function updateChatState(updater: (prev: ChatState) => ChatState) {
    setChatState((prev) => updater(prev));
  }

  function markComposerAttachment(
    attachmentId: string,
    patch: Partial<Pick<ChatAttachment, "status" | "error">>,
  ) {
    setPendingAttachments((prev) =>
      prev.map((attachment) =>
        attachment.id === attachmentId ? { ...attachment, ...patch } : attachment,
      ),
    );
  }

  async function uploadRequestObject(params: {
    requestId: string;
    dataUrl: string;
    fileName?: string;
  }) {
    const parsed = parseDataUrl(params.dataUrl);
    if (!parsed) throw new Error("Attachment is not a valid base64 data URL");
    const response = await fetch("/api/objects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request_id: params.requestId,
        mime_type: parsed.mimeType,
        file_name: params.fileName,
        bytes_base64: parsed.base64,
      }),
    });
    if (!response.ok) {
      throw new Error(`Attachment upload failed (${response.status})`);
    }
    return (await response.json()) as { token: string };
  }

  async function buildAttachmentBlocks(
    attachments: ChatAttachment[],
    requestId: string,
    clientId: string,
    onStatusChange?: (
      attachmentId: string,
      patch: Partial<Pick<ChatAttachment, "status" | "error">>,
    ) => void,
  ) {
    const contentBlocks: Array<Record<string, unknown>> = [];
    for (const attachment of attachments) {
      onStatusChange?.(attachment.id, { status: "uploading", error: undefined });
      try {
        const upload = await uploadRequestObject({
          requestId,
          dataUrl: attachment.dataUrl,
          fileName: attachment.fileName,
        });
        const url = `mesh://blob/${clientId}/${upload.token}`;
        if (attachment.kind === "image") {
          contentBlocks.push({
            type: "input_image",
            image_url: url,
          });
        } else if (attachment.kind === "audio") {
          contentBlocks.push({
            type: "input_audio",
            audio_url: url,
          });
        } else {
          contentBlocks.push({
            type: "input_file",
            url,
            mime_type: attachment.mimeType,
            file_name: attachment.fileName,
          });
        }
        onStatusChange?.(attachment.id, { status: "pending", error: undefined });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        onStatusChange?.(attachment.id, { status: "failed", error: message });
        throw error;
      }
    }
    return contentBlocks;
  }

  async function buildResponsesInput(
    historyForRequest: ChatMessage[],
    requestId: string,
    clientId: string,
    prebuiltContentByMessageId?: Record<string, Array<Record<string, unknown>>>,
  ) {
    return Promise.all(
      historyForRequest.map(async (message) => {
        const contentBlocks: Array<Record<string, unknown>> =
          prebuiltContentByMessageId?.[message.id]?.slice() ?? [];
        const attachments = messageAttachments(message);
        if (message.content.trim()) {
          contentBlocks.push({ type: "input_text", text: message.content });
        }
        if (!prebuiltContentByMessageId?.[message.id] && attachments.length > 0) {
          contentBlocks.push(
            ...(await buildAttachmentBlocks(attachments, requestId, clientId)),
          );
        }
        return {
          role: message.role,
          content:
            contentBlocks.length === 1 &&
            attachments.length === 0 &&
            contentBlocks[0].type === "input_text"
              ? message.content
              : contentBlocks,
        };
      }),
    );
  }

  async function streamAssistantReply(params: {
    conversationId: string;
    assistantId: string;
    model: string;
    historyForRequest: ChatMessage[];
    requestId?: string;
    prebuiltContentByMessageId?: Record<string, Array<Record<string, unknown>>>;
  }) {
    const {
      conversationId,
      assistantId,
      model,
      historyForRequest,
      requestId: providedRequestId,
      prebuiltContentByMessageId,
    } = params;
    const reqStart = performance.now();
    const controller = new AbortController();
    currentAbortRef.current = controller;
    let batcher: ReturnType<typeof createRafBatcher> | null = null;

    try {
      const requestId = providedRequestId ?? randomId();
      const clientId = chatClientIdRef.current;
      const requestInput = await buildResponsesInput(
        historyForRequest,
        requestId,
        clientId,
        prebuiltContentByMessageId,
      );
      const MAX_RETRIES = 3;
      const RETRY_DELAYS = [1000, 2000, 4000];
      const RETRYABLE = new Set([500, 502, 503]);
      let response: Response | null = null;

      for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        response = await fetch("/api/responses", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            model,
            client_id: clientId,
            request_id: requestId,
            input: requestInput,
            stream: true,
            stream_options: { include_usage: true },
            chat_template_kwargs: { enable_thinking: false },
          }),
        });
        if (response.ok && response.body) break;
        if (!RETRYABLE.has(response.status) || attempt === MAX_RETRIES - 1)
          break;
        await new Promise((r) => setTimeout(r, RETRY_DELAYS[attempt]));
      }

      if (!response?.ok || !response?.body)
        throw new Error(`HTTP ${response?.status ?? "unknown"}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let full = "";
      let completionTokens: number | null = null;
      let firstTokenAt: number | null = null;

      // Batch token deltas so React re-renders at most once per frame.
      batcher = createRafBatcher((snapshot) => {
        updateChatState((prev) => ({
          ...prev,
          conversations: updateConversationList(
            prev.conversations,
            conversationId,
            (conversation) => ({
              ...conversation,
              messages: conversation.messages.map((m) =>
                m.id === assistantId ? { ...m, content: snapshot } : m,
              ),
              updatedAt: Date.now(),
            }),
          ),
        }));
      });

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        buf = buf.replace(/\r\n/g, "\n");
        let frameEnd = buf.indexOf("\n\n");
        while (frameEnd >= 0) {
          const frame = buf.slice(0, frameEnd);
          buf = buf.slice(frameEnd + 2);
          const lines = frame.split("\n");
          let eventName = "";
          const dataLines: string[] = [];
          for (const line of lines) {
            if (line.startsWith("event:")) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              dataLines.push(line.slice(5).trimStart());
            }
          }
          const data = dataLines.join("\n").trim();
          if (!data || data === "[DONE]") {
            frameEnd = buf.indexOf("\n\n");
            continue;
          }
          try {
            const payload = JSON.parse(data) as Record<string, unknown>;
            if (eventName === "response.output_text.delta") {
              const contentDelta =
                typeof payload.delta === "string" ? payload.delta : "";
              if (!contentDelta) {
                frameEnd = buf.indexOf("\n\n");
                continue;
              }
              if (firstTokenAt == null) firstTokenAt = performance.now();
              full += contentDelta;
              batcher.push(full);
            } else if (eventName === "response.completed") {
              const responsePayload =
                payload.response && typeof payload.response === "object"
                  ? (payload.response as {
                      output_text?: unknown;
                      usage?: { output_tokens?: unknown };
                    })
                  : null;
              if (
                responsePayload &&
                typeof responsePayload.output_text === "string" &&
                !full
              ) {
                full = responsePayload.output_text;
              }
              if (
                responsePayload &&
                responsePayload.usage &&
                Number.isFinite(responsePayload.usage.output_tokens)
              ) {
                completionTokens = Number(responsePayload.usage.output_tokens);
              }
            }
          } catch {
            // ignore malformed chunk
          }
          frameEnd = buf.indexOf("\n\n");
        }
      }

      batcher.flush();

      const endAt = performance.now();
      const genStart = firstTokenAt ?? reqStart;
      const genSecs = Math.max(0.001, (endAt - genStart) / 1000);
      const ttftMs = Math.max(
        0,
        Math.round((firstTokenAt ?? endAt) - reqStart),
      );
      const tokenCount = Number.isFinite(completionTokens)
        ? completionTokens!
        : Math.max(1, Math.round(Math.max(full.length, 1) / 4));
      const tps = tokenCount / genSecs;
      const stats = `${tokenCount} tok · ${tps.toFixed(1)} tok/s · TTFT ${ttftMs}ms`;

      updateChatState((prev) => ({
        ...prev,
        conversations: updateConversationList(
          prev.conversations,
          conversationId,
          (conversation) => ({
              ...conversation,
              messages: conversation.messages.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      content: m.content || "(empty response)",
                      stats,
                    }
                  : m,
              ),
            updatedAt: Date.now(),
          }),
        ),
      }));
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        updateChatState((prev) => ({
          ...prev,
          conversations: updateConversationList(
            prev.conversations,
            conversationId,
            (conversation) => ({
              ...conversation,
              messages: conversation.messages.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content || "(stopped)" }
                  : m,
              ),
              updatedAt: Date.now(),
            }),
          ),
        }));
      } else {
        const message = err instanceof Error ? err.message : String(err);
        updateChatState((prev) => ({
          ...prev,
          conversations: updateConversationList(
            prev.conversations,
            conversationId,
            (conversation) => ({
              ...conversation,
              messages: conversation.messages.map((m) =>
                m.id === assistantId
                  ? { ...m, content: `Error: ${message}`, error: true }
                  : m,
              ),
              updatedAt: Date.now(),
            }),
          ),
        }));
      }
    } finally {
      batcher?.cancel();
      if (currentAbortRef.current === controller)
        currentAbortRef.current = null;
      setIsSending(false);
    }
  }

  async function sendMessage(text: string) {
    const trimmed = text.trim();
    if ((!trimmed && pendingAttachments.length === 0) || !status)
      return;
    if (isSending) {
      queuedInputRef.current = trimmed;
      // For attachment-only sends trimmed is ""; show a placeholder so the
      // queued-message UI indicator is visible.
      setQueuedText(trimmed || "📎 Attachment");
      setInput("");
      return;
    }
    if (attachmentSendIssue) {
      setComposerError(attachmentSendIssue);
      return;
    }

    // Prefer an explicitly compatible model when sending media with model=auto.
    let model = selectedModel || status.model_name;
    if (pendingAttachments.length > 0 && (!model || model === "auto")) {
      const multimodalModel = warmModels.find(
        (m) =>
          (!pendingKinds.has("image") || visionModels.has(m)) &&
          (!pendingKinds.has("audio") || audioModels.has(m)) &&
          (!pendingKinds.has("file") || multimodalModels.has(m)),
      );
      if (multimodalModel) {
        model = multimodalModel;
      } else if (pendingKinds.has("image")) {
        const visionModel = warmModels.find((m) => visionModels.has(m));
        if (visionModel) model = visionModel;
      } else if (pendingKinds.has("audio")) {
        const audioModel = warmModels.find((m) => audioModels.has(m));
        if (audioModel) model = audioModel;
      } else if (pendingKinds.has("file")) {
        const fileModel = warmModels.find((m) => multimodalModels.has(m));
        if (fileModel) model = fileModel;
      }
    }
    setComposerError(null);
    const conversationId = activeConversation?.id ?? randomId();
    const userMessage: ChatMessage = {
      id: randomId(),
      role: "user",
      content: trimmed,
      model,
      attachments:
        pendingAttachments.length > 0
          ? pendingAttachments.map(({ status, error, ...attachment }) => attachment)
          : undefined,
    };
    const requestId = randomId();
    const clientId = chatClientIdRef.current;
    let prebuiltContentByMessageId: Record<string, Array<Record<string, unknown>>> | undefined;
    if (pendingAttachments.length > 0) {
      try {
        const blocks = await buildAttachmentBlocks(
          pendingAttachments,
          requestId,
          clientId,
          (attachmentId, patch) => markComposerAttachment(attachmentId, patch),
        );
        prebuiltContentByMessageId = { [userMessage.id]: blocks };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setComposerError(`Attachment upload failed: ${message}`);
        return;
      }
    }
    const assistantId = randomId();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      model,
    };
    const existingMessages = activeConversation?.messages ?? [];
    const historyForRequest = [...existingMessages, userMessage];
    const nextTitle =
      (activeConversation?.title === DEFAULT_CHAT_TITLE &&
        existingMessages.length === 0) ||
      !activeConversation
        ? deriveConversationTitle(trimmed)
        : activeConversation.title;

    updateChatState((prev) => ({
      ...(prev.conversations.some(
        (conversation) => conversation.id === conversationId,
      )
        ? {
            ...prev,
            conversations: updateConversationList(
              prev.conversations,
              conversationId,
              (conversation) => ({
                ...conversation,
                title: nextTitle,
                updatedAt: Date.now(),
                messages: [...historyForRequest, assistantMessage],
              }),
            ),
            activeConversationId: conversationId,
          }
        : {
            conversations: [
              {
                id: conversationId,
                title: nextTitle,
                createdAt: Date.now(),
                updatedAt: Date.now(),
                messages: [...historyForRequest, assistantMessage],
              },
              ...prev.conversations,
            ],
            activeConversationId: conversationId,
          }),
    }));
    setSection("chat");
    setRoutedChatId(conversationId);
    pushRoute({ section: "chat", chatId: conversationId });
    setInput("");
    setPendingAttachments([]);
    setIsSending(true);
    await streamAssistantReply({
      conversationId,
      assistantId,
      model,
      historyForRequest,
      requestId,
      prebuiltContentByMessageId,
    });
  }

  async function regenerateLastResponse() {
    if (!status || isSending || !activeConversation) return;
    const model = selectedModel || status.model_name;
    const conversationId = activeConversation.id;
    const lastUserIndex = findLastUserMessageIndex(activeConversation.messages);
    if (lastUserIndex < 0) return;
    const historyForRequest = activeConversation.messages.slice(
      0,
      lastUserIndex + 1,
    );
    const assistantId = randomId();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      model,
    };

    updateChatState((prev) => ({
      ...prev,
      conversations: updateConversationList(
        prev.conversations,
        conversationId,
        (conversation) => ({
          ...conversation,
          updatedAt: Date.now(),
          messages: [...historyForRequest, assistantMessage],
        }),
      ),
    }));
    setIsSending(true);
    await streamAssistantReply({
      conversationId,
      assistantId,
      model,
      historyForRequest,
    });
  }

  function stopStreaming() {
    currentAbortRef.current?.abort();
  }

  function createNewConversation() {
    queuedInputRef.current = null;
    setQueuedText(null);
    const conversation = createConversation();
    updateChatState((prev) => ({
      conversations: [conversation, ...prev.conversations],
      activeConversationId: conversation.id,
    }));
    setSection("chat");
    setRoutedChatId(conversation.id);
    pushRoute({ section: "chat", chatId: conversation.id });
    setReasoningOpen({});
    setInput("");
    setPendingAttachments([]);
  }

  function selectConversation(conversationId: string) {
    updateChatState((prev) => {
      if (prev.activeConversationId === conversationId) return prev;
      if (
        !prev.conversations.some(
          (conversation) => conversation.id === conversationId,
        )
      )
        return prev;
      return { ...prev, activeConversationId: conversationId };
    });
    setSection("chat");
    setRoutedChatId(conversationId);
    pushRoute({ section: "chat", chatId: conversationId });
    setReasoningOpen({});
    setInput("");
    setPendingAttachments([]);
  }

  function renameConversation(conversationId: string, nextTitle: string) {
    const title = clampText(nextTitle.trim(), 140) || DEFAULT_CHAT_TITLE;
    updateChatState((prev) => ({
      ...prev,
      conversations: updateConversationList(
        prev.conversations,
        conversationId,
        (conversation) => ({
          ...conversation,
          title,
          updatedAt: Date.now(),
        }),
      ),
    }));
  }

  function deleteConversation(conversationId: string) {
    const remaining = conversations.filter(
      (conversation) => conversation.id !== conversationId,
    );
    const nextActiveId =
      activeConversationId === conversationId
        ? (remaining[0]?.id ?? null)
        : activeConversationId || null;
    updateChatState((prev) => {
      return {
        conversations: prev.conversations.filter(
          (conversation) => conversation.id !== conversationId,
        ),
        activeConversationId:
          prev.activeConversationId === conversationId
            ? (nextActiveId ?? "")
            : prev.activeConversationId,
      };
    });
    setSection("chat");
    setRoutedChatId(nextActiveId);
    replaceRoute({ section: "chat", chatId: nextActiveId });
    setReasoningOpen({});
    setInput("");
    setPendingAttachments([]);
  }

  function clearAllConversations() {
    updateChatState(() => ({ conversations: [], activeConversationId: "" }));
    setSection("chat");
    setRoutedChatId(null);
    replaceRoute({ section: "chat", chatId: null });
    setReasoningOpen({});
    setInput("");
    setPendingAttachments([]);
  }

  const topologyNodes = useMemo<TopologyNode[]>(() => {
    if (!status) return [];
    const nodes: TopologyNode[] = [];
    if (status.node_id) {
      nodes.push({
        id: status.node_id,
        vram: overviewVramGb(status.is_client, status.my_vram_gb),
        self: true,
        host: status.is_host,
        client: status.is_client,
        serving: status.model_name || "",
        servingModels:
          status.hosted_models && status.hosted_models.length > 0
            ? status.hosted_models
            : status.serving_models && status.serving_models.length > 0
              ? status.serving_models
              : status.model_name
                ? [status.model_name]
                : [],
        statusLabel:
          status.node_status ||
          (status.is_client ? "Client" : status.is_host ? "Host" : "Idle"),
        latencyMs: null,
        hostname: status.my_hostname,
        isSoc: status.my_is_soc,
        gpus: status.gpus,
      });
    }
    for (const p of status.peers ?? []) {
      const pModels =
        peerRoutableModels(p).length > 0
          ? peerRoutableModels(p)
          : peerAssignedModels(p);
      nodes.push({
        id: p.id,
        vram: overviewVramGb(p.role === "Client", p.vram_gb),
        self: false,
        host: /^Host/.test(p.role),
        client: p.role === "Client",
        serving: peerPrimaryModel(p),
        servingModels: pModels,
        statusLabel: peerStatusLabel(p),
        latencyMs: p.rtt_ms ?? null,
        hostname: p.hostname,
        isSoc: p.is_soc,
        gpus: p.gpus,
      });
    }
    return nodes;
  }, [status]);

  const sections: Array<{ key: TopSection; label: string }> = [
    { key: "dashboard", label: "Network" },
    { key: "chat", label: "Chat" },
  ];

  function navigateToSection(next: TopSection) {
    if (next === section) return;
    const nextChatId =
      next === "chat" ? (activeConversation?.id ?? null) : null;
    pushRoute({ section: next, chatId: nextChatId });
    setSection(next);
    setRoutedChatId(nextChatId);
  }

  function handleSubmit() {
    if (!canChat) return;
    void sendMessage(input);
  }

  return (
    <TooltipProvider>
      <div className="h-screen overflow-hidden bg-background [height:100svh] [padding-top:env(safe-area-inset-top)] [padding-bottom:env(safe-area-inset-bottom)]">
        <div className="flex h-full min-h-0 flex-col">
          <AppHeader
            sections={sections}
            section={section}
            setSection={navigateToSection}
            themeMode={themeMode}
            setThemeMode={setThemeMode}
            statusError={statusError}
            inviteWithModelCommand={inviteWithModelCommand}
            inviteWithModelName={inviteWithModelName}
            inviteClientCommand={inviteClientCommand}
            inviteToken={inviteToken}
            apiDirectUrl={apiDirectUrl}
            isPublicMesh={status?.nostr_discovery ?? false}
          />

          <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
            {section === "chat" ? (
              <div className="mx-auto flex min-h-0 min-w-0 w-full max-w-7xl flex-1 flex-col overflow-hidden p-2 md:p-4">
                <ChatPage
                  status={status}
                  inviteToken={status?.token ?? ""}
                  isPublicMesh={status?.nostr_discovery ?? false}
                  isFlyHosted={isFlyHosted}
                  inflightRequests={status?.inflight_requests ?? 0}
                  warmModels={warmModels}
                  meshModelByName={meshModelByName}
                  modelStatsByName={modelStatsByName}
                  selectedModel={selectedModel}
                  setSelectedModel={setSelectedModel}
                  selectedModelNodeCount={selectedModelNodeCount}
                  selectedModelVramGb={selectedModelVramGb}
                  selectedModelVision={selectedModelVision}
                  selectedModelAudio={selectedModelAudio}
                  selectedModelMultimodal={selectedModelMultimodal}
                  composerError={composerError}
                  setComposerError={setComposerError}
                  attachmentSendIssue={attachmentSendIssue}
                  pendingAttachments={pendingAttachments}
                  setPendingAttachments={setPendingAttachments}
                  conversations={conversations}
                  activeConversationId={activeConversationId}
                  onConversationCreate={createNewConversation}
                  onConversationSelect={selectConversation}
                  onConversationRename={renameConversation}
                  onConversationDelete={deleteConversation}
                  onConversationsClear={clearAllConversations}
                  messages={messages}
                  reasoningOpen={reasoningOpen}
                  setReasoningOpen={setReasoningOpen}
                  chatScrollRef={chatScrollRef}
                  input={input}
                  setInput={setInput}
                  isSending={isSending}
                  queuedText={queuedText}
                  canChat={canChat}
                  canRegenerate={canRegenerate}
                  onStop={stopStreaming}
                  onRegenerate={regenerateLastResponse}
                  onSubmit={handleSubmit}
                />
              </div>

            ) : null}

            {section === "dashboard" ? (
              <div className="min-h-0 flex-1 overflow-y-auto">
                <div className="mx-auto w-full max-w-7xl p-4">
                  <DashboardPage
                    status={status}
                    meshModels={meshModels}
                    modelsLoading={modelsLoading}
                    topologyNodes={topologyNodes}
                    selectedModel={selectedModel || status?.model_name || ""}
                    meshModelByName={meshModelByName}
                    themeMode={themeMode}
                    isPublicMesh={status?.nostr_discovery ?? false}
                    inviteToken={inviteToken}
                    isLocalhost={isLocalhost}
                  />
                </div>
              </div>
            ) : null}
          </main>
          <footer
            className={cn(
              "shrink-0 bg-card/70",
              section === "chat" ? "hidden md:block" : "",
            )}
          >
            <div className="mx-auto flex h-8 w-full max-w-7xl items-center justify-center gap-2 px-4 text-xs text-muted-foreground">
              Mesh LLM{" "}
              {status?.version ? `v${status.version}` : "version loading..."}
              {status?.latest_version ? (
                <>
                  <span>·</span>
                  <a
                    href="https://github.com/michaelneale/mesh-llm/releases"
                    target="_blank"
                    rel="noreferrer"
                    className="underline-offset-2 hover:text-foreground hover:underline"
                    title="A newer mesh-llm version is available"
                  >
                    {status?.version
                      ? `Update available: v${status.version} -> v${status.latest_version}`
                      : `Update available: v${status.latest_version}`}
                  </a>
                </>
              ) : null}
              <span>·</span>
              <a
                href="https://github.com/michaelneale/mesh-llm"
                target="_blank"
                rel="noreferrer"
                className="inline-flex h-5 w-5 items-center justify-center hover:text-foreground"
                aria-label="GitHub repository"
                title="GitHub repository"
              >
                <span className="relative h-4 w-4">
                  <img
                    src={githubBlackLogo}
                    alt=""
                    aria-hidden="true"
                    className="h-4 w-4 dark:hidden"
                  />
                  <img
                    src={githubWhiteLogo}
                    alt=""
                    aria-hidden="true"
                    className="hidden h-4 w-4 dark:block"
                  />
                </span>
              </a>
            </div>
          </footer>
        </div>
      </div>
    </TooltipProvider>
  );
}

function AppHeader({
  sections,
  section,
  setSection,
  themeMode,
  setThemeMode,
  statusError,
  inviteWithModelCommand,
  inviteWithModelName,
  inviteClientCommand,
  inviteToken,
  apiDirectUrl,
  isPublicMesh,
}: {
  sections: Array<{ key: TopSection; label: string }>;
  section: TopSection;
  setSection: (section: TopSection) => void;
  themeMode: ThemeMode;
  setThemeMode: React.Dispatch<React.SetStateAction<ThemeMode>>;
  statusError: string | null;
  inviteWithModelCommand: string;
  inviteWithModelName: string;
  inviteClientCommand: string;
  inviteToken: string;
  apiDirectUrl: string;
  isPublicMesh: boolean;
}) {
  const [inviteWithModelCopied, setInviteWithModelCopied] = useState(false);
  const [inviteClientCopied, setInviteClientCopied] = useState(false);
  const [tokenCopied, setTokenCopied] = useState(false);
  const [apiDirectCopied, setApiDirectCopied] = useState(false);
  const [isThemePopoverOpen, setIsThemePopoverOpen] = useState(false);

  async function copyInviteWithModelCommand() {
    if (!inviteWithModelCommand) return;
    try {
      await navigator.clipboard.writeText(inviteWithModelCommand);
      setInviteWithModelCopied(true);
      window.setTimeout(() => setInviteWithModelCopied(false), 1500);
    } catch {
      setInviteWithModelCopied(false);
    }
  }

  async function copyInviteClientCommand() {
    if (!inviteClientCommand) return;
    try {
      await navigator.clipboard.writeText(inviteClientCommand);
      setInviteClientCopied(true);
      window.setTimeout(() => setInviteClientCopied(false), 1500);
    } catch {
      setInviteClientCopied(false);
    }
  }

  async function copyInviteToken() {
    if (!inviteToken) return;
    try {
      await navigator.clipboard.writeText(inviteToken);
      setTokenCopied(true);
      window.setTimeout(() => setTokenCopied(false), 1500);
    } catch {
      setTokenCopied(false);
    }
  }

  async function copyApiDirectUrl() {
    if (!apiDirectUrl) return;
    try {
      await navigator.clipboard.writeText(apiDirectUrl);
      setApiDirectCopied(true);
      window.setTimeout(() => setApiDirectCopied(false), 1500);
    } catch {
      setApiDirectCopied(false);
    }
  }

  function selectThemeMode(mode: ThemeMode) {
    setThemeMode(mode);
    setIsThemePopoverOpen(false);
  }

  return (
    <header className="shrink-0 border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/80">
      <div className="mx-auto flex h-14 w-full max-w-7xl items-center gap-2 px-3 md:h-16 md:gap-4 md:px-4">
        <div className="flex min-w-0 items-center gap-0">
          <div className="flex h-10 w-7 shrink-0 items-center justify-start">
            <BrandIcon className="h-6 w-6 text-primary" />
          </div>
          <div className="hidden min-w-0 sm:block">
            <div className="truncate text-base font-semibold">
              <MeshLlmWordmark />
            </div>
          </div>
        </div>
        <NavigationMenu>
          <NavigationMenuList>
            {sections.map(({ key, label }) => (
              <NavigationMenuItem key={key}>
                <NavigationMenuLink asChild>
                  <a
                    href={key === "chat" ? "/chat" : "/dashboard"}
                    onClick={(event) => {
                      const isPlainLeftClick =
                        event.button === 0 &&
                        !event.metaKey &&
                        !event.ctrlKey &&
                        !event.shiftKey &&
                        !event.altKey;
                      if (!isPlainLeftClick) return;
                      event.preventDefault();
                      setSection(key);
                    }}
                    className={navigationMenuTriggerStyle()}
                    data-active={section === key ? "" : undefined}
                    aria-current={section === key ? "page" : undefined}
                  >
                    {label}
                  </a>
                </NavigationMenuLink>
              </NavigationMenuItem>
            ))}
          </NavigationMenuList>
        </NavigationMenu>
        <div className="ml-auto flex items-center gap-2">
          <Popover>
            <Tooltip>
              <TooltipTrigger asChild>
                <PopoverTrigger asChild>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    aria-label="API access"
                  >
                    <Braces className="h-4 w-4" />
                  </Button>
                </PopoverTrigger>
              </TooltipTrigger>
              <TooltipContent>API</TooltipContent>
            </Tooltip>
            <PopoverContent
              className="w-[calc(100vw-2rem)] max-w-[420px] space-y-3"
              align="end"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Braces className="h-4 w-4 text-muted-foreground" />
                  <span>API Access</span>
                </div>
                <div className="text-xs text-muted-foreground">
                  OpenAI-compatible endpoint — works with any app that speaks
                  the OpenAI API.
                </div>
              </div>
              {apiDirectUrl ? (
                <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                  <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                    {apiDirectUrl}
                  </code>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 shrink-0"
                    aria-label="Copy endpoint URL"
                    onClick={() => void copyApiDirectUrl()}
                  >
                    {apiDirectCopied ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : (
                      <Copy className="h-3.5 w-3.5" />
                    )}
                  </Button>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="text-xs text-muted-foreground">
                    Run mesh-llm locally to get an OpenAI-compatible API on your
                    machine:
                  </div>
                  <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                    <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                      {isPublicMesh
                        ? "mesh-llm --auto"
                        : `mesh-llm --auto --join ${inviteToken || "(token)"}`}
                    </code>
                    <Button
                      type="button"
                      size="icon"
                      variant="ghost"
                      className="h-6 w-6 shrink-0"
                      aria-label="Copy command"
                      onClick={() =>
                        void navigator.clipboard.writeText(
                          isPublicMesh
                            ? "mesh-llm --auto"
                            : `mesh-llm --auto --join ${inviteToken || ""}`,
                        )
                      }
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    This gives you{" "}
                    <code className="text-[0.7rem]">
                      http://127.0.0.1:9337/v1
                    </code>{" "}
                    locally — point any OpenAI-compatible app at it.
                  </div>
                </div>
              )}
              <div className="space-y-2 pt-1">
                <div className="text-xs font-medium">Use with agents</div>
                <div className="space-y-1">
                  {["claude", "goose"].map((agent) => {
                    const cmd = isPublicMesh
                      ? `mesh-llm ${agent}`
                      : `mesh-llm ${agent} --join ${inviteToken || "(token)"}`;
                    return (
                      <div
                        key={agent}
                        className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5"
                      >
                        <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                          {cmd}
                        </code>
                        <Button
                          type="button"
                          size="icon"
                          variant="ghost"
                          className="h-6 w-6 shrink-0"
                          aria-label={`Copy ${agent} command`}
                          onClick={() =>
                            void navigator.clipboard.writeText(cmd)
                          }
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                      </div>
                    );
                  })}
                </div>
                <div className="text-xs text-muted-foreground">
                  Also works with pi and any OpenAI-compatible client.{" "}
                  <a
                    href={`${DOCS_URL}/#agents`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline hover:text-foreground"
                  >
                    Setup guide →
                  </a>
                </div>
              </div>
              <div className="text-xs text-muted-foreground pt-1">
                Don't have it yet?{" "}
                <a
                  href={`${DOCS_URL}/#install`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-foreground"
                >
                  Install mesh-llm →
                </a>
              </div>
              <div className="text-xs text-muted-foreground pt-1">
                Agents can gossip too —{" "}
                <code className="text-[0.9em]">
                  mesh-llm blackboard install-skill
                </code>{" "}
                <a
                  href={`${DOCS_URL}/#blackboard`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-foreground"
                >
                  →
                </a>
              </div>
            </PopoverContent>
          </Popover>
          <Popover>
            <Tooltip>
              <TooltipTrigger asChild>
                <PopoverTrigger asChild>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    aria-label="Invite"
                    disabled={!inviteToken}
                  >
                    <UserPlus className="h-4 w-4" />
                  </Button>
                </PopoverTrigger>
              </TooltipTrigger>
              <TooltipContent>Invite</TooltipContent>
            </Tooltip>
            <PopoverContent
              className="w-[calc(100vw-2rem)] max-w-[420px] space-y-3"
              align="end"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <UserPlus className="h-4 w-4 text-muted-foreground" />
                  <span>Invite to this mesh</span>
                </div>
                <div className="text-xs text-muted-foreground">
                  Invite with a model loaded to add compute, or invite as a
                  client for API-only access.
                </div>
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-xs font-medium">
                  <span>Contribute compute</span>
                  <Badge className="h-5 gap-1 border-emerald-500/40 bg-emerald-500/10 px-2 text-[10px] text-emerald-700 dark:text-emerald-300">
                    <Sparkles className="h-3 w-3" />
                    Recommended
                  </Badge>
                </div>
                <div className="text-xs text-muted-foreground">
                  Joins and serves the model{" "}
                  {inviteWithModelName || "selected model"}
                </div>
              </div>
              {inviteWithModelCommand ? (
                <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                  <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                    {inviteWithModelCommand}
                  </code>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 shrink-0"
                    aria-label="Copy model command"
                    onClick={() => void copyInviteWithModelCommand()}
                  >
                    {inviteWithModelCopied ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : (
                      <Copy className="h-3.5 w-3.5" />
                    )}
                  </Button>
                </div>
              ) : (
                <div className="text-xs text-muted-foreground">
                  No warm model selected yet.
                </div>
              )}
              <div className="space-y-1 pt-1">
                <div className="text-xs font-medium">Join as client</div>
                <div className="text-xs text-muted-foreground">
                  Connects for API access without loading a model.
                </div>
              </div>
              {inviteClientCommand ? (
                <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                  <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                    {inviteClientCommand}
                  </code>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 shrink-0"
                    aria-label="Copy client command"
                    onClick={() => void copyInviteClientCommand()}
                  >
                    {inviteClientCopied ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : (
                      <Copy className="h-3.5 w-3.5" />
                    )}
                  </Button>
                </div>
              ) : (
                <div className="text-xs text-muted-foreground">
                  No invite token available yet.
                </div>
              )}
              <div className="space-y-1 pt-1">
                <div className="text-xs font-medium">Invite token</div>
                {inviteToken ? (
                  <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                    <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                      {inviteToken}
                    </code>
                    <Button
                      type="button"
                      size="icon"
                      variant="ghost"
                      className="h-7 w-7 shrink-0"
                      aria-label="Copy invite token"
                      onClick={() => void copyInviteToken()}
                    >
                      {tokenCopied ? (
                        <Check className="h-3.5 w-3.5" />
                      ) : (
                        <Copy className="h-3.5 w-3.5" />
                      )}
                    </Button>
                  </div>
                ) : (
                  <div className="text-xs text-muted-foreground">
                    No invite token available yet.
                  </div>
                )}
              </div>
            </PopoverContent>
          </Popover>
          <Popover
            open={isThemePopoverOpen}
            onOpenChange={setIsThemePopoverOpen}
          >
            <Tooltip>
              <TooltipTrigger asChild>
                <PopoverTrigger asChild>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-9 gap-1 px-2"
                    aria-label={`Theme: ${themeMode}`}
                  >
                    {themeMode === "auto" ? (
                      <Laptop className="h-4 w-4" />
                    ) : null}
                    {themeMode === "light" ? <Sun className="h-4 w-4" /> : null}
                    {themeMode === "dark" ? <Moon className="h-4 w-4" /> : null}
                    <ChevronDown className="h-3 w-3 text-muted-foreground" />
                  </Button>
                </PopoverTrigger>
              </TooltipTrigger>
              <TooltipContent>Theme</TooltipContent>
            </Tooltip>
            <PopoverContent className="w-40 space-y-1 p-1" align="end">
              <button
                type="button"
                className={cn(
                  "flex w-full items-center justify-between rounded-md px-2 py-1.5 text-xs hover:bg-muted",
                  themeMode === "auto" ? "bg-muted" : "",
                )}
                onClick={() => selectThemeMode("auto")}
              >
                <span className="flex items-center gap-2">
                  <Laptop className="h-3.5 w-3.5" />
                  Auto
                </span>
                {themeMode === "auto" ? (
                  <Check className="h-3.5 w-3.5" />
                ) : null}
              </button>
              <button
                type="button"
                className={cn(
                  "flex w-full items-center justify-between rounded-md px-2 py-1.5 text-xs hover:bg-muted",
                  themeMode === "light" ? "bg-muted" : "",
                )}
                onClick={() => selectThemeMode("light")}
              >
                <span className="flex items-center gap-2">
                  <Sun className="h-3.5 w-3.5" />
                  Light
                </span>
                {themeMode === "light" ? (
                  <Check className="h-3.5 w-3.5" />
                ) : null}
              </button>
              <button
                type="button"
                className={cn(
                  "flex w-full items-center justify-between rounded-md px-2 py-1.5 text-xs hover:bg-muted",
                  themeMode === "dark" ? "bg-muted" : "",
                )}
                onClick={() => selectThemeMode("dark")}
              >
                <span className="flex items-center gap-2">
                  <Moon className="h-3.5 w-3.5" />
                  Dark
                </span>
                {themeMode === "dark" ? (
                  <Check className="h-3.5 w-3.5" />
                ) : null}
              </button>
            </PopoverContent>
          </Popover>
        </div>
      </div>
      {statusError ? (
        <div className="mx-auto w-full max-w-7xl px-4 pb-3">
          <Alert variant="destructive">
            <Loader2 className="h-4 w-4 animate-spin" />
            <AlertTitle>Connection Interrupted</AlertTitle>
            <AlertDescription>{statusError}</AlertDescription>
          </Alert>
        </div>
      ) : null}
    </header>
  );
}

export function ChatPage(props: {
  status: StatusPayload | null;
  inviteToken: string;
  isPublicMesh: boolean;
  isFlyHosted: boolean;
  inflightRequests: number;
  warmModels: string[];
  meshModelByName: Record<string, MeshModel>;
  modelStatsByName: Record<string, ModelServingStat>;
  selectedModel: string;
  setSelectedModel: (v: string) => void;
  selectedModelNodeCount: number | null;
  selectedModelVramGb: number | null;
  selectedModelVision: boolean;
  selectedModelAudio: boolean;
  selectedModelMultimodal: boolean;
  composerError: string | null;
  setComposerError: React.Dispatch<React.SetStateAction<string | null>>;
  attachmentSendIssue: string | null;
  pendingAttachments: ChatAttachment[];
  setPendingAttachments: React.Dispatch<React.SetStateAction<ChatAttachment[]>>;
  conversations: ChatConversation[];
  activeConversationId: string;
  onConversationCreate: () => void;
  onConversationSelect: (conversationId: string) => void;
  onConversationRename: (conversationId: string, title: string) => void;
  onConversationDelete: (conversationId: string) => void;
  onConversationsClear: () => void;
  messages: ChatMessage[];
  reasoningOpen: Record<string, boolean>;
  setReasoningOpen: React.Dispatch<
    React.SetStateAction<Record<string, boolean>>
  >;
  chatScrollRef: React.RefObject<HTMLDivElement>;
  input: string;
  setInput: (v: string) => void;
  isSending: boolean;
  queuedText: string | null;
  canChat: boolean;
  canRegenerate: boolean;
  onStop: () => void;
  onRegenerate: () => void;
  onSubmit: () => void;
}) {
  const {
    status,
    inviteToken,
    warmModels,
    meshModelByName,
    modelStatsByName,
    selectedModel,
    setSelectedModel,
    selectedModelNodeCount,
    selectedModelVramGb,
    selectedModelVision,
    selectedModelAudio,
    selectedModelMultimodal,
    composerError,
    setComposerError,
    attachmentSendIssue,
    pendingAttachments,
    setPendingAttachments,
    conversations,
    activeConversationId,
    onConversationCreate,
    onConversationSelect,
    onConversationRename,
    onConversationDelete,
    onConversationsClear,
    messages,
    reasoningOpen,
    setReasoningOpen,
    chatScrollRef,
    input,
    setInput,
    isSending,
    queuedText,
    canChat,
    canRegenerate,
    onStop,
    onRegenerate,
    onSubmit,
  } = props;

  const hasChats = conversations.length > 0;
  const selectedModelValue = selectedModel || warmModels[0] || "";
  const [editingConversationId, setEditingConversationId] = useState<
    string | null
  >(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const chatInputRef = useRef<HTMLTextAreaElement | null>(null);
  const editingTitleInputRef = useRef<HTMLInputElement | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const audioInputRef = useRef<HTMLInputElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  function markPendingAttachment(
    attachmentId: string,
    patch: Partial<Pick<ChatAttachment, "status" | "error">>,
  ) {
    setPendingAttachments((prev) =>
      prev.map((attachment) =>
        attachment.id === attachmentId ? { ...attachment, ...patch } : attachment,
      ),
    );
  }

  function addPendingAttachment(attachment: Omit<ChatAttachment, "id" | "status" | "error">) {
    setComposerError(null);
    setPendingAttachments((prev) => [
      ...prev,
      {
        id: randomId(),
        status: "pending",
        ...attachment,
      },
    ]);
  }

  function removePendingAttachment(attachmentId: string) {
    setComposerError(null);
    setPendingAttachments((prev) =>
      prev.filter((attachment) => attachment.id !== attachmentId),
    );
  }

  function resetAttachmentStatus(attachmentId: string) {
    markPendingAttachment(attachmentId, { status: "pending", error: undefined });
    setComposerError(null);
  }

  function handleImageSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const validationError = validateAttachmentFile(file, "image");
    if (validationError) {
      setComposerError(validationError);
      e.target.value = "";
      return;
    }
    // Read as data URL first (handles HEIC, JPEG, PNG — whatever the browser supports)
    // then resize via canvas to max 512px (vision encoders tile at ~448px internally)
    const reader = new FileReader();
    reader.onload = () => {
      const src = reader.result as string;
      const img = new Image();
      img.onload = () => {
        const MAX = 512;
        let { width, height } = img;
        if (width > MAX || height > MAX) {
          const scale = MAX / Math.max(width, height);
          width = Math.round(width * scale);
          height = Math.round(height * scale);
        }
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          addPendingAttachment({
            kind: "image",
            dataUrl: src,
            mimeType: parseDataUrl(src)?.mimeType || file.type || "image/jpeg",
            fileName: file.name,
          });
          return;
        }
        ctx.drawImage(img, 0, 0, width, height);
        addPendingAttachment({
          kind: "image",
          dataUrl: canvas.toDataURL("image/jpeg", 0.85),
          mimeType: "image/jpeg",
          fileName: file.name,
        });
      };
      img.onerror = () => {
        // Canvas resize failed — send original (may be large but better than nothing)
        addPendingAttachment({
          kind: "image",
          dataUrl: src,
          mimeType: parseDataUrl(src)?.mimeType || file.type || "image/jpeg",
          fileName: file.name,
        });
      };
      img.src = src;
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  }

  function handleAudioSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const validationError = validateAttachmentFile(file, "audio");
    if (validationError) {
      setComposerError(validationError);
      e.target.value = "";
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      addPendingAttachment({
        kind: "audio",
        dataUrl,
        mimeType: file.type || parseDataUrl(dataUrl)?.mimeType || "audio/wav",
        fileName: file.name,
      });
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const validationError = validateAttachmentFile(file, "file");
    if (validationError) {
      setComposerError(validationError);
      e.target.value = "";
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      addPendingAttachment({
        kind: "file",
        dataUrl,
        mimeType:
          file.type ||
          parseDataUrl(dataUrl)?.mimeType ||
          "application/octet-stream",
        fileName: file.name,
      });
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  }

  useEffect(() => {
    if (!activeConversationId || !canChat || isSending) return;
    chatInputRef.current?.focus();
  }, [activeConversationId, canChat, isSending]);

  useEffect(() => {
    if (!editingConversationId) return;
    const frame = window.requestAnimationFrame(() => {
      editingTitleInputRef.current?.focus();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [editingConversationId]);

  function startInlineRename(conversation: ChatConversation) {
    setEditingConversationId(conversation.id);
    setEditingTitle(conversation.title);
  }

  function cancelInlineRename() {
    setEditingConversationId(null);
    setEditingTitle("");
  }

  function saveInlineRename() {
    if (!editingConversationId) return;
    onConversationRename(editingConversationId, editingTitle);
    cancelInlineRename();
  }

  function handleDelete(conversation: ChatConversation) {
    if (!window.confirm(`Delete "${conversation.title}"?`)) return;
    onConversationDelete(conversation.id);
  }

  function handleClearAll() {
    if (!window.confirm("Clear all chats?")) return;
    onConversationsClear();
  }

  // Conversation list content — reused in desktop sidebar and mobile sheet
  const conversationListContent = (
    <div className="space-y-3 p-3">
      <Button
        type="button"
        size="sm"
        className="w-full"
        onClick={() => {
          onConversationCreate();
          setMobileSidebarOpen(false);
        }}
        disabled={isSending}
      >
        <MessageSquarePlus className="mr-1.5 h-4 w-4" />
        New chat
      </Button>
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs text-muted-foreground">Conversations</div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-7 px-2 text-xs"
          onClick={handleClearAll}
          disabled={!hasChats || isSending}
        >
          <Trash2 className="mr-1.5 h-3.5 w-3.5" />
          Clear
        </Button>
      </div>
      <ScrollArea className="h-[calc(100svh_-_14rem)] md:h-[calc(100svh_-_24rem)]">
        <div className="space-y-1">
          {conversations.map((conversation) => {
            const isActive = conversation.id === activeConversationId;
            const isEditing = editingConversationId === conversation.id;
            return (
              <div
                key={conversation.id}
                className={cn(
                  "group flex items-center gap-2 rounded-md border p-2",
                  isActive
                    ? "border-primary/40 bg-muted/40"
                    : "border-transparent",
                )}
              >
                {isEditing ? (
                  <div className="min-w-0 flex-1 space-y-1">
                    <input
                      ref={editingTitleInputRef}
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          saveInlineRename();
                        } else if (e.key === "Escape") {
                          e.preventDefault();
                          cancelInlineRename();
                        }
                      }}
                      className="h-7 w-full rounded-md border bg-background px-2 text-sm outline-none ring-offset-background focus:ring-2 focus:ring-ring"
                    />
                    <div className="text-xs text-muted-foreground">
                      {conversation.messages.length} message
                      {conversation.messages.length === 1 ? "" : "s"}
                    </div>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="min-w-0 flex-1 text-left"
                    onClick={() => {
                      onConversationSelect(conversation.id);
                      setMobileSidebarOpen(false);
                    }}
                    disabled={isSending}
                  >
                    <div className="text-sm font-medium leading-5 [display:-webkit-box] [overflow:hidden] [-webkit-box-orient:vertical] [-webkit-line-clamp:3] [overflow-wrap:anywhere]">
                      {conversation.title}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {conversation.messages.length} message
                      {conversation.messages.length === 1 ? "" : "s"}
                    </div>
                  </button>
                )}
                {isEditing ? (
                  <>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 shrink-0 opacity-70 hover:opacity-100"
                      onClick={saveInlineRename}
                      aria-label="Save conversation name"
                    >
                      <Check className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 shrink-0 opacity-70 hover:opacity-100"
                      onClick={cancelInlineRename}
                      aria-label="Cancel rename"
                    >
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </>
                ) : (
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 shrink-0 opacity-70 hover:opacity-100"
                    onClick={() => startInlineRename(conversation)}
                    disabled={isSending}
                    aria-label="Rename conversation"
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </Button>
                )}
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 shrink-0 opacity-70 hover:opacity-100"
                  onClick={() => handleDelete(conversation)}
                  disabled={isSending}
                  aria-label="Delete conversation"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );

  return (
    <Card className="flex h-full min-h-0 flex-1 flex-col overflow-hidden">
      <CardHeader className="px-3 py-2 md:px-6 md:py-4">
        <div className="flex items-center gap-2 md:gap-3">
          {/* Mobile: chats button (opens sheet) + new chat */}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 shrink-0 gap-1.5 md:hidden"
            onClick={() =>
              hasChats ? setMobileSidebarOpen(true) : onConversationCreate()
            }
            aria-label={hasChats ? "Chats" : "New chat"}
          >
            {hasChats ? (
              <>
                <Hash className="h-4 w-4" />
                <span className="text-xs tabular-nums">
                  {conversations.length}
                </span>
              </>
            ) : (
              <MessageSquarePlus className="h-4 w-4" />
            )}
          </Button>
          {hasChats ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0 md:hidden"
              onClick={onConversationCreate}
              disabled={isSending}
              aria-label="New chat"
            >
              <MessageSquarePlus className="h-4 w-4" />
            </Button>
          ) : null}
          <CardTitle className="hidden md:block text-base shrink-0">
            Chat
          </CardTitle>
          <div className="ml-auto flex items-center gap-2">
            {selectedModelNodeCount != null ? (
              <div className="hidden md:flex h-8 items-center gap-1.5 rounded-md border bg-muted/40 px-2">
                <Network className="h-3.5 w-3.5 text-muted-foreground" />
                <div className="text-xs leading-none">
                  <span className="font-medium">{selectedModelNodeCount}</span>
                  <span className="ml-1 text-muted-foreground">nodes</span>
                </div>
              </div>
            ) : null}
            {selectedModelVramGb != null ? (
              <div className="hidden md:flex h-8 items-center gap-1.5 rounded-md border bg-muted/40 px-2">
                <Cpu className="h-3.5 w-3.5 text-muted-foreground" />
                <div className="text-xs leading-none">
                  <span className="font-medium">
                    {selectedModelVramGb.toFixed(1)}
                  </span>
                  <span className="ml-1 text-muted-foreground">GB</span>
                </div>
              </div>
            ) : null}
            <span className="hidden md:inline text-xs text-muted-foreground">
              Model
            </span>
            <Select
              value={selectedModelValue}
              onValueChange={setSelectedModel}
              disabled={!warmModels.length}
            >
              <SelectTrigger className="h-8 w-full min-w-0 max-w-[180px] md:max-w-[320px] md:w-[320px]">
                <SelectValue placeholder="Select model">
                  {selectedModelValue === "auto"
                    ? "✨ Auto (router picks best)"
                    : selectedModelValue
                      ? shortName(
                          modelDisplayName(
                            meshModelByName[selectedModelValue],
                          ) || selectedModelValue,
                        )
                      : undefined}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {warmModels.length > 1 ? (
                  <SelectItem
                    key="auto"
                    value="auto"
                    className="group py-2 data-[state=checked]:bg-accent data-[state=checked]:text-accent-foreground"
                  >
                    <div className="flex min-w-0 flex-col gap-0.5">
                      <span className="leading-5">✨ Auto</span>
                      <span className="text-xs leading-4 text-muted-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground">
                        Router picks best model for each request
                      </span>
                    </div>
                  </SelectItem>
                ) : null}
                {warmModels.map((model) => {
                  const modelStats = modelStatsByName[model];
                  const selectedMeshModel = meshModelByName[model];
                  const displayName =
                    modelDisplayName(selectedMeshModel) || model;
                  const multimodalInfo = multimodalBadge(selectedMeshModel);
                  const visionInfo = visionBadge(selectedMeshModel);
                  const audioInfo = audioBadge(selectedMeshModel);
                  const reasoningInfo = reasoningBadge(selectedMeshModel);
                  return (
                    <SelectItem
                      key={model}
                      value={model}
                      className="group py-2 data-[state=checked]:bg-accent data-[state=checked]:text-accent-foreground"
                    >
                      <div className="flex min-w-0 flex-col gap-0.5">
                        <span className="truncate leading-5">
                          {shortName(displayName)}
                          {multimodalInfo && (
                            <span className="ml-1.5" title={multimodalInfo.title}>
                              {multimodalInfo.icon}
                            </span>
                          )}
                          {visionInfo && (
                            <span className="ml-1.5" title={visionInfo.title}>
                              {visionInfo.icon}
                            </span>
                          )}
                          {audioInfo && (
                            <span className="ml-1.5" title={audioInfo.title}>
                              {audioInfo.icon}
                            </span>
                          )}
                          {reasoningInfo && (
                            <span
                              className="ml-1.5"
                              title={reasoningInfo.title}
                            >
                              {reasoningInfo.icon}
                            </span>
                          )}
                        </span>
                        {displayName !== model ? (
                          <span className="truncate text-xs leading-4 text-muted-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground">
                            {model}
                          </span>
                        ) : null}
                        {modelStats ? (
                          <span className="grid grid-cols-[108px_132px] gap-x-3 text-xs leading-4 text-muted-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground">
                            <span className="inline-flex items-center gap-1">
                              <Network className="h-3 w-3 text-muted-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground" />
                              <span>Nodes</span>
                              <span className="font-medium tabular-nums text-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground">
                                {modelStats.nodes}
                              </span>
                            </span>
                            <span className="inline-flex items-center gap-1">
                              <Cpu className="h-3 w-3 text-muted-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground" />
                              <span>VRAM</span>
                              <span className="font-medium tabular-nums text-foreground group-data-[highlighted]:text-accent-foreground group-data-[state=checked]:text-accent-foreground">
                                {modelStats.vramGb.toFixed(1)} GB
                              </span>
                            </span>
                          </span>
                        ) : null}
                      </div>
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <Separator />
      {props.isFlyHosted ? (
        <div
          className={cn(
            "border-b px-4 py-2 text-xs",
            props.inflightRequests > 2
              ? "bg-orange-500/10 text-orange-700 dark:text-orange-400"
              : "bg-muted/40 text-muted-foreground",
          )}
        >
          {props.inflightRequests > 2 ? (
            <>
              <span className="font-medium">⏳ Busy</span> —{" "}
              {props.inflightRequests} requests in flight, responses may be
              slow. For direct access run{" "}
              <code className="rounded bg-muted px-1 py-0.5 font-mono">
                mesh-llm --auto
              </code>{" "}
              <a
                href={DOCS_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-foreground"
              >
                Learn more →
              </a>
            </>
          ) : (
            <>
              <span className="font-medium">Community demo</span> — best-effort
              public instance. For direct, faster access run{" "}
              <code className="rounded bg-muted px-1 py-0.5 font-mono">
                mesh-llm --auto
              </code>{" "}
              to join the mesh or start your own.{" "}
              <a
                href={DOCS_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-foreground"
              >
                Learn more →
              </a>
            </>
          )}
        </div>
      ) : null}
      {/* Mobile conversation sheet */}
      <Sheet open={mobileSidebarOpen} onOpenChange={setMobileSidebarOpen}>
        <SheetContent side="left" className="w-72 p-0">
          <SheetHeader className="sr-only">
            <SheetTitle>Chats</SheetTitle>
          </SheetHeader>
          {conversationListContent}
        </SheetContent>
      </Sheet>

      <CardContent className="min-h-0 flex-1 p-0">
        <div className="flex h-full min-h-0 min-w-0 md:flex-row">
          {/* Desktop sidebar — hidden on mobile */}
          {hasChats ? (
            <aside className="hidden md:block shrink-0 md:w-72 md:border-r">
              {conversationListContent}
            </aside>
          ) : null}

          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            <div
              ref={chatScrollRef}
              className={cn(
                "min-h-0 flex-1 overflow-x-hidden overflow-y-auto px-3 py-4 md:px-6",
                messages.length === 0 ? "" : "space-y-4",
              )}
            >
              {messages.length === 0 ? (
                <div className="flex min-h-full items-center justify-center">
                  <InviteFriendEmptyState
                    inviteToken={inviteToken}
                    selectedModel={selectedModel || warmModels[0] || ""}
                    isPublicMesh={props.isPublicMesh}
                  />
                </div>
              ) : (
                <>
                  {messages.map((message, i) => (
                    <ChatBubble
                      key={message.id}
                      message={message}
                      reasoningOpen={!!reasoningOpen[message.id]}
                      onReasoningToggle={(open) =>
                        setReasoningOpen((prev) => ({
                          ...prev,
                          [message.id]: open,
                        }))
                      }
                      streaming={isSending && i === messages.length - 1}
                    />
                  ))}

                  {isSending ? (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Loader2 className="h-3.5 w-3.5 animate-spin" /> Streaming
                      response...
                    </div>
                  ) : null}

                  {queuedText ? (
                    <div className="flex justify-end">
                      <div className="max-w-[92%] md:max-w-[82%] opacity-50">
                        <div className="mb-1 flex items-center gap-2 px-1 text-xs text-muted-foreground">
                          <User className="h-3.5 w-3.5" />
                          <span>Queued</span>
                        </div>
                        <div className="rounded-lg border border-dashed bg-muted px-4 py-3 text-sm whitespace-pre-wrap">
                          {queuedText}
                        </div>
                      </div>
                    </div>
                  ) : null}
                </>
              )}
            </div>
            <Separator />
            <div className="shrink-0 box-border w-full max-w-full space-y-2 overflow-hidden p-3 md:space-y-3 md:p-4">
              {pendingAttachments.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {pendingAttachments.map((attachment) =>
                    attachment.kind === "image" ? (
                      <div
                        key={attachment.id}
                        className="relative inline-block"
                        data-testid="pending-attachment"
                      >
                        <img
                          src={attachment.dataUrl}
                          alt={attachment.fileName || "Attached image"}
                          className="h-20 rounded-md border object-cover"
                        />
                        <button
                          onClick={() => removePendingAttachment(attachment.id)}
                          className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-destructive text-destructive-foreground text-xs hover:bg-destructive/80"
                          aria-label="Remove image"
                        >
                          <X className="h-3 w-3" />
                        </button>
                        {attachment.status === "uploading" ? (
                          <div className="absolute inset-0 flex items-center justify-center rounded-md bg-background/70 text-xs">
                            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                            Uploading
                          </div>
                        ) : attachment.status === "failed" ? (
                          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-md bg-background/80 text-xs">
                            <span>Upload failed</span>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="h-6 px-2 text-xs"
                              onClick={() => resetAttachmentStatus(attachment.id)}
                            >
                              Retry
                            </Button>
                          </div>
                        ) : null}
                      </div>
                    ) : (
                      <div
                        key={attachment.id}
                        className="relative inline-flex max-w-full items-center gap-2 rounded-md border bg-muted/40 px-3 py-2 text-sm"
                        data-testid="pending-attachment"
                      >
                        {attachment.kind === "audio" ? (
                          <FileAudio className="h-4 w-4 shrink-0 text-muted-foreground" />
                        ) : (
                          <File className="h-4 w-4 shrink-0 text-muted-foreground" />
                        )}
                        <span className="truncate">
                          {attachment.fileName ||
                            (attachment.kind === "audio"
                              ? "Audio attachment"
                              : "File attachment")}
                        </span>
                        {attachment.status === "uploading" ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                        ) : null}
                        {attachment.status === "failed" ? (
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2 text-xs"
                            onClick={() => resetAttachmentStatus(attachment.id)}
                          >
                            Retry
                          </Button>
                        ) : null}
                        <button
                          onClick={() => removePendingAttachment(attachment.id)}
                          className="flex h-5 w-5 items-center justify-center rounded-full text-muted-foreground hover:bg-muted"
                          aria-label={`Remove ${attachment.kind}`}
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </div>
                    ),
                  )}
                </div>
              ) : null}
              {composerError || attachmentSendIssue ? (
                <Alert variant="destructive" data-testid="composer-error">
                  <AlertTitle>Attachment Issue</AlertTitle>
                  <AlertDescription>
                    {composerError || attachmentSendIssue}
                  </AlertDescription>
                </Alert>
              ) : null}
              <Textarea
                ref={chatInputRef}
                value={input}
                onChange={(e) => {
                  setComposerError(null);
                  setInput(e.target.value);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    onSubmit();
                  }
                }}
                data-testid="chat-input"
                rows={2}
                placeholder={
                  props.canChat
                    ? "Ask me anything..."
                    : "Waiting for a warm model..."
                }
                disabled={!props.canChat}
                className="min-h-[56px] md:min-h-[80px] resize-none text-base md:text-sm"
              />
              <div className="flex items-center justify-between gap-2">
                <div className="hidden md:block text-xs text-muted-foreground">
                  Enter to send. Shift+Enter for newline.
                </div>
                <div className="flex items-center gap-2">
                  {selectedModelVision && (
                    <>
                      <input
                        ref={imageInputRef}
                        type="file"
                        accept="image/*"
                        className="hidden"
                        data-testid="chat-image-input"
                        onChange={handleImageSelect}
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        onClick={() => imageInputRef.current?.click()}
                        disabled={!props.canChat || isSending}
                        title="Attach image"
                        aria-label="Attach image"
                      >
                        <ImagePlus className="h-4 w-4" />
                      </Button>
                    </>
                  )}
                  {selectedModelAudio && (
                    <>
                      <input
                        ref={audioInputRef}
                        type="file"
                        accept="audio/*"
                        className="hidden"
                        data-testid="chat-audio-input"
                        onChange={handleAudioSelect}
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        onClick={() => audioInputRef.current?.click()}
                        disabled={!props.canChat || isSending}
                        title="Attach audio"
                        aria-label="Attach audio"
                      >
                        <FileAudio className="h-4 w-4" />
                      </Button>
                    </>
                  )}
                  {selectedModelMultimodal && (
                    <>
                      <input
                        ref={fileInputRef}
                        type="file"
                        className="hidden"
                        data-testid="chat-file-input"
                        onChange={handleFileSelect}
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={!props.canChat || isSending}
                        title="Attach file"
                        aria-label="Attach file"
                      >
                        <Paperclip className="h-4 w-4" />
                      </Button>
                    </>
                  )}
                  {isSending ? (
                    <Button
                      type="button"
                      variant="outline"
                      size="icon"
                      onClick={onStop}
                      aria-label="Stop"
                    >
                      <Square className="h-3.5 w-3.5" />
                    </Button>
                  ) : (
                    <Button
                      type="button"
                      variant="outline"
                      size="icon"
                      onClick={onRegenerate}
                      disabled={!canRegenerate}
                      aria-label="Regenerate"
                    >
                      <RotateCcw className="h-4 w-4" />
                    </Button>
                  )}
                  <Button
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={onSubmit}
                    data-testid="chat-send"
                    disabled={
                      !props.canChat ||
                      (!input.trim() && pendingAttachments.length === 0)
                    }
                  >
                    {isSending ? (
                      <Send className="mr-2 h-4 w-4 text-muted-foreground" />
                    ) : (
                      <Send className="mr-2 h-4 w-4" />
                    )}
                    {isSending ? "Queue" : "Send"}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function InviteFriendEmptyState({
  inviteToken,
  selectedModel,
  isPublicMesh,
}: {
  inviteToken: string;
  selectedModel: string;
  isPublicMesh: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [inviteWithModelCopied, setInviteWithModelCopied] = useState(false);
  const inviteWithModelCommand =
    inviteToken && selectedModel
      ? `mesh-llm --join ${inviteToken} --model ${selectedModel}`
      : "";

  async function copyInviteWithModelCommand() {
    if (!inviteWithModelCommand) return;
    try {
      await navigator.clipboard.writeText(inviteWithModelCommand);
      setInviteWithModelCopied(true);
      window.setTimeout(() => setInviteWithModelCopied(false), 1500);
    } catch {
      setInviteWithModelCopied(false);
    }
  }

  if (isPublicMesh) {
    return (
      <div className="mx-auto w-full max-w-md space-y-4 px-2 text-center">
        <div className="flex justify-center">
          <BrandIcon className="h-12 w-12 text-primary/50 animate-wiggle" />
        </div>
        <p className="text-sm text-muted-foreground">
          Mesh LLM is a project to let people contribute spare compute, build
          private personal AI, using open models.
        </p>
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="mx-auto flex items-center gap-1.5 text-xs text-muted-foreground/70 hover:text-foreground transition-colors"
        >
          <ChevronDown
            className={cn(
              "h-3 w-3 transition-transform",
              open ? "" : "-rotate-90",
            )}
          />
          <span>Learn more…</span>
        </button>
        {open ? (
          <div className="space-y-4 rounded-md border border-dashed p-3 text-left">
            <div className="text-xs text-muted-foreground">
              <a
                href={DOCS_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-foreground"
              >
                Learn about this project →
              </a>
            </div>
            <Separator />
            <div className="space-y-2">
              <div className="text-xs font-medium">Contribute to the pool</div>
              <div className="text-xs text-muted-foreground">
                Have a spare machine? Add it to this mesh and share compute with
                others.
              </div>
              <code className="block rounded-md border bg-muted/40 px-2 py-1.5 text-xs">
                mesh-llm --auto
              </code>
            </div>
            <Separator />
            <div className="space-y-2">
              <div className="text-xs font-medium">
                Run your own private mesh
              </div>
              <div className="text-xs text-muted-foreground">
                Pool machines across your home, office, or friends — fully
                private, no cloud needed.{" "}
                <a
                  href={DOCS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-foreground"
                >
                  Getting started →
                </a>
              </div>
            </div>
            <Separator />
            <div className="space-y-2">
              <div className="text-xs font-medium">Use with coding agents</div>
              <div className="text-xs text-muted-foreground">
                Works with Claude Code, Goose, pi, and any OpenAI-compatible
                client.{" "}
                <a
                  href={`${DOCS_URL}/#agents`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-foreground"
                >
                  Agent setup →
                </a>
              </div>
            </div>
            <Separator />
            <div className="space-y-2">
              <div className="text-xs font-medium">Agent gossip</div>
              <div className="text-xs text-muted-foreground">
                Let your agents coordinate across machines — share status,
                findings, and questions. Works with any LLM setup.{" "}
                <a
                  href={`${DOCS_URL}/#blackboard`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-foreground"
                >
                  Blackboard docs →
                </a>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    );
  }

  // Private mesh — invite a friend to join
  return (
    <div className="mx-auto w-full max-w-md space-y-3 px-2 text-center">
      <div className="flex justify-center">
        <BrandIcon className="h-12 w-12 text-primary/50 animate-wiggle" />
      </div>
      <p className="text-sm text-muted-foreground">
        Mesh LLM lets you build private personal AI, using open models.{" "}
        <a
          href={DOCS_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-foreground"
        >
          Learn more →
        </a>
      </p>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="mx-auto flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 transition-transform",
            open ? "" : "-rotate-90",
          )}
        />
        <Network className="h-3.5 w-3.5" />
        <span>Invite someone to the mesh</span>
      </button>
      {open && inviteWithModelCommand ? (
        <div className="space-y-2 rounded-md border border-dashed p-3 text-left">
          <div className="text-xs text-muted-foreground">
            Share this command — they'll join and contribute compute:
          </div>
          <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
            <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
              {inviteWithModelCommand}
            </code>
            <Button
              type="button"
              size="icon"
              variant="ghost"
              className="h-7 w-7 shrink-0"
              aria-label="Copy command"
              onClick={() => void copyInviteWithModelCommand()}
            >
              {inviteWithModelCopied ? (
                <Check className="h-3.5 w-3.5" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
            </Button>
          </div>
          <div className="text-xs text-muted-foreground pt-1">
            Don't have it yet?{" "}
            <a
              href={`${DOCS_URL}/#install`}
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-foreground"
            >
              Install mesh-llm →
            </a>
          </div>
          <div className="text-xs text-muted-foreground pt-1">
            Agents can gossip too —{" "}
            <a
              href={`${DOCS_URL}/#blackboard`}
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-foreground"
            >
              Blackboard docs →
            </a>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function DashboardPage({
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
  const [modelFilter, setModelFilter] = useState<"all" | "warm" | "cold">(
    "all",
  );
  const [isMeshOverviewFullscreen, setIsMeshOverviewFullscreen] =
    useState(false);
  const [selectedTopologyNodeId, setSelectedTopologyNodeId] =
    useState<string>("");
  const [detailPanelStack, setDetailPanelStack] = useState<DetailPanelEntry[]>(
    [],
  );
  const [meshTopologyLayoutMode, setMeshTopologyLayoutMode] =
    useState<TopologyLayoutMode>("elk");
  const topologyDiagramNodes = useMemo(
    () => topologyNodes,
    [topologyNodes],
  );
  const filteredModels = useMemo(() => {
    const models = meshModels;
    return [...models]
      .filter((m) => (modelFilter === "all" ? true : m.status === modelFilter))
      .sort(
        (a, b) => b.node_count - a.node_count || a.name.localeCompare(b.name),
      );
  }, [meshModels, modelFilter]);
  const totalMeshVramGb = useMemo(() => meshGpuVram(status), [status]);
  const sortedPeers = useMemo(() => {
    return [...(status?.peers ?? [])].sort((a, b) => {
      const bOverviewVramGb = overviewVramGb(b.role === "Client", b.vram_gb);
      const aOverviewVramGb = overviewVramGb(a.role === "Client", a.vram_gb);
      return bOverviewVramGb - aOverviewVramGb || a.id.localeCompare(b.id);
    });
  }, [status?.peers]);
  const peerRows = useMemo(() => {
    return sortedPeers.map((peer) => {
      const statusLabel =
        peer.role === "Client" ? "Client" : peerStatusLabel(peer);
      const primaryModel = peerPrimaryModel(peer);
      const modelLabel =
        primaryModel && primaryModel !== "(idle)"
          ? shortName(primaryModel)
          : "idle";
      const latencyLabel = formatLatency(peer.rtt_ms);
      const displayVramGb = overviewVramGb(
        peer.role === "Client",
        peer.vram_gb,
      );
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
    const currentDetail = detailPanelStack[detailPanelStack.length - 1];
    const activeModelName =
      currentDetail?.kind === "model" ? currentDetail.modelName : null;
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
  }, [detailPanelStack, meshModelByName, peerRows, status]);

  const activeDetail = detailPanelStack[detailPanelStack.length - 1] ?? null;
  const activeModel =
    activeDetail?.kind === "model"
      ? meshModelByName[activeDetail.modelName] ?? null
      : null;
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
      vramSharePct:
        topologyNode.client
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
      version: topologyNode.self ? status.version : undefined,
      latestVersion: topologyNode.self ? status.latest_version : undefined,
      llamaReady: topologyNode.self ? status.llama_ready : undefined,
      apiPort: topologyNode.self ? status.api_port : undefined,
      inflightRequests: topologyNode.self ? status.inflight_requests : undefined,
      privacyLimited:
        !topologyNode.self &&
        !topologyNode.hostname &&
        !(topologyNode.gpus?.length ?? 0),
    } satisfies NodeSidebarRecord;
  }, [activeDetail, status, topologyNodes, totalMeshVramGb]);

  function pushDetail(entry: DetailPanelEntry) {
    if (isMeshOverviewFullscreen) return;
    setDetailPanelStack((prev) => {
      const current = prev[prev.length - 1];
      const isSameEntry =
        !!current &&
        ((current.kind === "node" &&
          entry.kind === "node" &&
          current.nodeId === entry.nodeId) ||
          (current.kind === "model" &&
            entry.kind === "model" &&
            current.modelName === entry.modelName));
      if (isSameEntry) {
        return prev;
      }
      return [...prev, entry];
    });
  }

  function openNodeDetail(nodeId: string) {
    pushDetail({ kind: "node", nodeId });
  }

  function openModelDetail(modelName: string) {
    if (!meshModelByName[modelName]) return;
    pushDetail({ kind: "model", modelName });
  }

  function closeDetailPanel() {
    setDetailPanelStack([]);
  }

  function goBackDetailPanel() {
    setDetailPanelStack((prev) => prev.slice(0, -1));
  }

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
      <SelectContent
        align="end"
        className={isMeshOverviewFullscreen ? "z-[130]" : undefined}
      >
        {TOPOLOGY_LAYOUT_OPTIONS.map((option) => (
          <SelectItem key={option.value} value={option.value} className="text-xs">
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );

  const gpuNodeCount = topologyDiagramNodes.filter((n) => !n.client).length;
  const clientCount = topologyDiagramNodes.filter((n) => n.client).length;

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
            href="https://github.com/michaelneale/mesh-llm"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 underline hover:text-foreground"
          >
            GitHub
          </a>
        </AlertDescription>
      </Alert>
      {modelsLoading && meshModels.length === 0 ? (
        <Alert className="border-border/60 bg-card/80">
          <Loader2 className="h-4 w-4 animate-spin" />
          <AlertTitle className="text-sm font-medium">
            Loading model catalog
          </AlertTitle>
          <AlertDescription className="space-y-2">
            <div className="text-xs text-muted-foreground">
              Scanning local models and assembling mesh metadata.
            </div>
          </AlertDescription>
        </Alert>
      ) : null}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <StatCard
          title="Node ID"
          value={status?.node_id ?? "n/a"}
          valueSuffix={
            <StatusPill
              label={status?.node_status ?? "n/a"}
              tone={topologyStatusTone(status?.node_status ?? "n/a")}
              tooltip={topologyStatusTooltip(status?.node_status ?? "n/a")}
            />
          }
          icon={<Hash className="h-4 w-4" />}
          tooltip="Current node identifier in this mesh."
        />
        <StatCard
          title="Active Models"
          value={`${meshModels.filter((m) => m.status === "warm").length}`}
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
                <div className="flex h-[360px] md:h-[420px] lg:h-[460px] xl:h-[520px] items-center justify-center rounded-lg border border-dashed bg-muted/20 text-sm text-muted-foreground">
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
                  highlightedNodeId={selectedTopologyNodeId}
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
                  onValueChange={(v) =>
                    setModelFilter(v as "all" | "warm" | "cold")
                  }
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
                                <span
                                  role="img"
                                  aria-label="Mixture of Experts"
                                >
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
                          {model.node_count} node
                          {model.node_count === 1 ? "" : "s"}
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
                  title={
                    meshModels.length > 0
                      ? `No ${modelFilter} models`
                      : "No model catalog data"
                  }
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
                      <TableHead>Status</TableHead>
                      <TableHead>Model</TableHead>
                      <TableHead className="text-right">Latency</TableHead>
                      <TableHead className="text-right">VRAM</TableHead>
                      <TableHead className="text-right whitespace-nowrap">
                        Share
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                      {peerRows.map((peer) => (
                       <TableRow
                         key={peer.id}
                         data-id={peer.id}
                         tabIndex={0}
                         className={cn(
                            "cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:bg-muted/40",
                            peer.id === selectedTopologyNodeId &&
                              "bg-muted/50 hover:bg-muted/60",
                          )}
                          onClick={() => setSelectedTopologyNodeId(peer.id)}
                          onKeyDown={(e) => {
                            if (e.target !== e.currentTarget) return;
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              setSelectedTopologyNodeId(peer.id);
                            }
                          }}
                       >
                        <TableCell className="font-mono text-xs">
                          <button
                            type="button"
                            className="text-left underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedTopologyNodeId(peer.id);
                              openNodeDetail(peer.id);
                            }}
                          >
                            {peer.id}
                          </button>
                        </TableCell>
                        <TableCell>{peer.role}</TableCell>
                        <TableCell>{peer.statusLabel}</TableCell>
                        <TableCell className="max-w-[180px] truncate">
                          {peer.modelLabel}
                        </TableCell>
                        <TableCell className="text-right">
                          {peer.latencyLabel}
                        </TableCell>
                        <TableCell className="text-right">
                          {peer.role === "Client"
                            ? "n/a"
                            : `${peer.displayVramGb.toFixed(1)} GB`}
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
                      highlightedNodeId={selectedTopologyNodeId}
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

      <Sheet open={detailPanelStack.length > 0 && !isMeshOverviewFullscreen} onOpenChange={(open) => !open && closeDetailPanel()}>
        <SheetContent side="right" className="w-full overflow-y-auto border-l bg-background/95 p-0 backdrop-blur sm:max-w-2xl" onOpenAutoFocus={(e) => { e.preventDefault(); (e.currentTarget as HTMLElement).focus(); }}>
          {activeDetail?.kind === "node" && activeNode ? (
            <NodeSidebar
              node={activeNode}
              meshModelByName={meshModelByName}
              onOpenModel={openModelDetail}
              onBack={detailPanelStack.length > 1 ? goBackDetailPanel : undefined}
            />
          ) : activeDetail?.kind === "node" && !activeNode ? (
            <div className="flex h-full flex-col items-center justify-center gap-4 p-8 text-center">
              <p className="text-sm text-muted-foreground">This node is no longer available.</p>
              <button
                type="button"
                className="text-xs underline hover:text-foreground"
                onClick={detailPanelStack.length > 1 ? goBackDetailPanel : closeDetailPanel}
              >
                {detailPanelStack.length > 1 ? "Go back" : "Close"}
              </button>
            </div>
          ) : null}
          {activeDetail?.kind === "model" && activeModel ? (
            <ModelSidebar
              model={activeModel}
              activePeers={activePeerRows}
              onOpenNode={openNodeDetail}
              onBack={detailPanelStack.length > 1 ? goBackDetailPanel : undefined}
            />
          ) : activeDetail?.kind === "model" && !activeModel ? (
            <div className="flex h-full flex-col items-center justify-center gap-4 p-8 text-center">
              <p className="text-sm text-muted-foreground">This model is no longer available.</p>
              <button
                type="button"
                className="text-xs underline hover:text-foreground"
                onClick={detailPanelStack.length > 1 ? goBackDetailPanel : closeDetailPanel}
              >
                {detailPanelStack.length > 1 ? "Go back" : "Close"}
              </button>
            </div>
          ) : null}
        </SheetContent>
      </Sheet>

      {/* Connect panel */}
      <Card className="mt-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Connect</CardTitle>
          <div className="text-xs text-muted-foreground">
            Run mesh-llm on your machine to get a local OpenAI-compatible API
            and contribute compute to the mesh.
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
              This auto-selects a model for your hardware, joins the mesh, and
              serves a local API at{" "}
              <code className="text-[0.7rem]">http://127.0.0.1:9337/v1</code>
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

type TopologyFlowNodeData = {
  node: PositionedTopologyNode;
  info: TopologyNodeInfo;
  selected: boolean;
  sameModelAsCurrent: boolean;
  layoutDirection: "horizontal" | "vertical";
};

type TopologyFlowDiagramNode = Node<TopologyFlowNodeData, "topologyNode" | "clientDot">;

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
                {data.info.loadedModels.map((m) => (
                  <div
                    key={m}
                    className="inline-flex items-center gap-1 text-[11px] font-medium leading-4"
                  >
                    <Sparkles className="h-3 w-3 shrink-0 text-muted-foreground" />
                    <span className="min-w-0 truncate" title={m}>
                      {shortName(m)}
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
              <span className="font-medium">
                {formatLatency(data.info.latencyMs)}
              </span>
            </div>
          ) : null}
          <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
            <MemoryStick className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">VRAM</span>
            <span className="font-medium">
              {data.info.vramGb.toFixed(1)} GB
            </span>
          </div>
          <div className="inline-flex items-center gap-1 rounded-full border bg-muted/30 px-2 py-1">
            <Gauge className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">Share</span>
            <span className="whitespace-nowrap font-medium">
              {data.info.vramSharePct}%
            </span>
          </div>
          {data.info.gpus?.map((gpu, i) => {
            const duplicateCount = data.info.gpus
              ?.slice(0, i)
              .filter(
                (candidate) =>
                  candidate.name === gpu.name &&
                  candidate.vram_bytes === gpu.vram_bytes,
              ).length ?? 0;
            const lower = gpu.name.toLowerCase();
            const isNvidia =
              lower.includes("nvidia") || lower.includes("jetson");
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
  const handleStyle = { opacity: 0, width: 1, height: 1, border: 0, pointerEvents: "none" as const };
  return (
    <div className={cn("flex flex-col items-center gap-0.5", data.selected ? "opacity-100" : "opacity-40")}>
      <div className={cn("relative h-3.5 w-3.5 rounded-full border bg-muted", data.selected ? "border-ring ring-1 ring-ring/50" : "border-border")}>
        <Handle type="target" position={isHorizontal ? Position.Left : Position.Top} style={handleStyle} />
        <Handle type="source" position={isHorizontal ? Position.Right : Position.Bottom} style={handleStyle} />
      </div>
      <span className={cn("max-w-[60px] truncate text-[8px] leading-none", data.selected ? "text-foreground" : "text-muted-foreground")}>
        {data.node.hostname || data.node.id.slice(0, 8)}
      </span>
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

function MeshTopologyDiagram({
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
  status: StatusPayload | null;
  nodes: TopologyNode[];
  selectedModel: string;
  layoutMode: TopologyLayoutMode;
  themeMode: ThemeMode;
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
  status: StatusPayload;
  nodes: TopologyNode[];
  selectedModel: string;
  layoutMode: TopologyLayoutMode;
  themeMode: ThemeMode;
  onOpenNode?: (nodeId: string) => void;
  highlightedNodeId?: string;
  fullscreen: boolean;
  heightClass?: string;
  containerStyle?: CSSProperties;
}) {
  const {
    center,
    serving,
    workers,
    clients,
    nodeRadius,
    clientEdgeStride,
    meshVramGb,
  } = useMemo(() => {
    const center =
      nodes.find((n) => n.host) || nodes.find((n) => n.self) || nodes[0];
    const others = nodes
      .filter((n) => n.id !== center.id)
      .sort((a, b) => b.vram - a.vram || a.id.localeCompare(b.id));
    const focusModel = selectedModel || status.model_name || "";
    const serving = others.filter(
      (n) =>
        !n.client && !!n.serving && (!focusModel || n.serving === focusModel),
    );
    const servingIds = new Set(serving.map((n) => n.id));
    const clients = others.filter((n) => n.client);
    const workers = others.filter(
      (n) => !n.client && !servingIds.has(n.id),
    );

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
    const clientEdgeStride =
      total > 320 ? 6 : total > 220 ? 4 : total > 120 ? 2 : 1;
    const meshVramGb = nodes
      .filter((n) => !n.client)
      .reduce((sum, n) => sum + Math.max(0, n.vram), 0);

    return {
      center,
      serving,
      workers,
      clients,
      nodeRadius,
      clientEdgeStride,
      meshVramGb,
    };
  }, [nodes, selectedModel, status.model_name]);
  const currentNodeServingModel = useMemo(() => {
    const current = nodes.find((n) => n.self);
    if (
      !current ||
      current.client ||
      !current.serving ||
      current.serving === "(idle)"
    )
      return "";
    return current.serving;
  }, [nodes]);

  const [selectedNodeId, setSelectedNodeId] = useState(center.id);

  useEffect(() => {
    setSelectedNodeId((prev) =>
      nodes.some((n) => n.id === prev) ? prev : center.id,
    );
  }, [nodes, center.id]);

  const nodeIdsRef = useRef(new Set(nodes.map((n) => n.id)));
  useEffect(() => {
    nodeIdsRef.current = new Set(nodes.map((n) => n.id));
  }, [nodes]);
  useEffect(() => {
    if (highlightedNodeId && nodeIdsRef.current.has(highlightedNodeId)) {
      setSelectedNodeId(highlightedNodeId);
    }
  }, [highlightedNodeId]);

  const nodeInfoById = useMemo(() => {
    const out = new Map<string, TopologyNodeInfo>();
    for (const node of nodes) {
      const servingModel =
        !node.client && node.serving && node.serving !== "(idle)"
          ? node.serving
          : "";
      const servingModels = !node.client
        ? node.servingModels.filter((m) => m && m !== "(idle)")
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
      out.set(node.id, {
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
    return out;
  }, [nodes, meshVramGb]);
  const activeLayout = TOPOLOGY_LAYOUTS[layoutMode];
  const topologyLayoutNodes = useMemo<BucketedTopologyNode[]>(() => {
    const toBucketedNode = (
      node: TopologyNode,
      bucket: BucketedTopologyNode["bucket"],
    ): BucketedTopologyNode => ({
      ...node,
      bucket,
      width: node.client ? 64 : TOPOLOGY_NODE_WIDTH,
      height: node.client ? 28 : estimateTopologyNodeHeight(node, nodeInfoById.get(node.id)),
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
      positionedTopologyLayoutsEqual(previous, immediateLayout)
        ? previous
        : immediateLayout,
    );

    void activeLayout
      .getLayout(nextLayoutContext)
      .then((next) => {
        if (
          !cancelled &&
          layoutInputSignatureRef.current === runSignature
        ) {
          setPositioned((previous) =>
            positionedTopologyLayoutsEqual(previous, next) ? previous : next,
          );
        }
      })
      .catch(() => {
        if (
          !cancelled &&
          layoutInputSignatureRef.current === runSignature
        ) {
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
  const layoutReady =
    positioned.length > 0 && positionedIdsKey === expectedPositionIdsKey;
  const flowColorMode =
    themeMode === "auto"
      ? typeof document !== "undefined" &&
        document.documentElement.classList.contains("dark")
        ? "dark"
        : "light"
      : themeMode;
  const layoutFitSignature = useMemo(
    () =>
      `${fullscreen ? "fs" : "std"}:${layoutMode}:${positionedTopologySignature(positioned)}`,
    [fullscreen, layoutMode, positioned],
  );
  const flowContainerRef = useRef<HTMLDivElement | null>(null);
  const flowInstanceRef = useRef<
    ReactFlowInstance<TopologyFlowDiagramNode, Edge> | null
  >(null);
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
      const size = ready
        ? `${Math.round(rect.width)}x${Math.round(rect.height)}`
        : "";

      setContainerReady((previous) => (previous === ready ? previous : ready));
      setContainerSizeSignature((previous) =>
        previous === size ? previous : size,
      );
    };

    update();
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
    return positioned.map((p) => ({
      id: p.id,
      type: p.client ? "clientDot" : "topologyNode",
      position: { x: p.x, y: p.y },
      origin: [0.5, 0],
      sourcePosition: isHorizontal ? Position.Right : Position.Bottom,
      targetPosition: isHorizontal ? Position.Left : Position.Top,
      data: {
        node: p,
        layoutDirection: activeLayout.direction,
        info: nodeInfoById.get(p.id) ?? {
          role: "Node",
          statusLabel: "n/a",
          latencyMs: null,
          loadedModel: "idle",
          loadedModels: [],
          vramGb: 0,
          vramSharePct: 0,
        },
        selected: p.id === selectedNodeId,
        sameModelAsCurrent:
          !!currentNodeServingModel &&
          !p.client &&
          p.serving === currentNodeServingModel,
      },
      draggable: false,
      selectable: false,
      connectable: false,
      zIndex: p.id === center.id ? 10 : 1,
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
      .filter((p) => p.id !== center.id)
      .map((p) => {
        const isClient = p.bucket === "client";
        const stroke = isClient
          ? "rgba(160,160,160,0.15)"
          : p.bucket === "serving"
            ? "rgba(34,197,94,0.35)"
            : "rgba(56,189,248,0.3)";
        return {
          id: `edge-${center.id}-${p.id}`,
          source: center.id,
          target: p.id,
          type: activeLayout.edgeType,
          className: `mesh-edge mesh-edge--${p.bucket}`,
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
          colorMode={flowColorMode}
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


// KaTeX math renderer — loads from CDN on first use
let katexCssLoaded = false;
const katexPromise = import(
  "https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.mjs" as string
)
  .then((m) => {
    if (!katexCssLoaded) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = "https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.css";
      document.head.appendChild(link);
      katexCssLoaded = true;
    }
    return m.default;
  })
  .catch(() => null);

function KaTeXBlock({ math, display }: { math: string; display: boolean }) {
  const [rendered, setRendered] = useState(false);
  const blockRef = useRef<HTMLDivElement | null>(null);
  const inlineRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    setRendered(false);
    katexPromise.then((katex) => {
      const container = display ? blockRef.current : inlineRef.current;
      if (cancelled || !katex || !container) return;
      container.innerHTML = "";
      try {
        katex.render(math, container, {
          displayMode: display,
          throwOnError: false,
        });
        if (!cancelled) setRendered(true);
      } catch {}
    });
    return () => {
      cancelled = true;
    };
  }, [math, display]);

  return display ? (
    <>
      <div ref={blockRef} className={rendered ? "my-2 overflow-x-auto" : "hidden"} />
      {!rendered && (
        <div className="my-2 overflow-x-auto text-sm">
          <code>{math}</code>
        </div>
      )}
    </>
  ) : (
    <>
      <span ref={inlineRef} className={rendered ? undefined : "hidden"} />
      {!rendered && <code>{math}</code>}
    </>
  );
}

// Mermaid diagram renderer — loads mermaid from CDN on first use
const mermaidPromise = import(
  "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs" as string
)
  .then((m) => {
    m.default.initialize({
      startOnLoad: false,
      theme: "dark",
      securityLevel: "antiscript",
    });
    return m.default;
  })
  .catch(() => null);

function MermaidBlock({ code }: { code: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!svg || !containerRef.current) return;
    containerRef.current.innerHTML = svg;
    return () => {
      if (containerRef.current) containerRef.current.innerHTML = "";
    };
  }, [svg]);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setSvg(null);
    mermaidPromise.then(async (mermaid) => {
      if (cancelled || !mermaid) {
        if (!cancelled) setError("Mermaid failed to load");
        return;
      }
      try {
        const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        const { svg: rendered } = await mermaid.render(id, code);
        if (!cancelled) setSvg(rendered);
      } catch (e: unknown) {
        if (!cancelled)
          setError(e instanceof Error ? e.message : "Render failed");
      }
    });
    return () => {
      cancelled = true;
    };
  }, [code]);

  if (error)
    return (
      <pre className="my-2 rounded-lg border border-border/70 bg-background/80 p-3 text-xs text-muted-foreground">
        <code>{code}</code>
      </pre>
    );
  if (!svg)
    return (
      <div className="my-2 flex items-center gap-2 rounded-lg border border-border/70 bg-background/80 p-3 text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        Rendering diagram…
      </div>
    );

  return (
    <div
      ref={containerRef}
      className="my-2 overflow-x-auto rounded-lg border border-border/70 bg-background/80 p-3 [&_svg]:max-w-full"
    />
  );
}

function MarkdownMessage({
  content,
  streaming,
}: {
  content: string;
  streaming?: boolean;
}) {
  return (
    <div
      className={cn(
        "break-words text-sm leading-6",
        "[&_a]:underline [&_a]:underline-offset-2",
        "[&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:italic",
        "[&_code]:rounded [&_code]:bg-background/70 [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.9em]",
        "[&_h1]:mb-2 [&_h1]:mt-3 [&_h1]:text-base [&_h1]:font-semibold [&_h1:first-child]:mt-0",
        "[&_h2]:mb-2 [&_h2]:mt-3 [&_h2]:text-sm [&_h2]:font-semibold [&_h2:first-child]:mt-0",
        "[&_hr]:my-3 [&_hr]:border-border",
        "[&_li]:my-0.5",
        "[&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5",
        "[&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0",
        "[&_pre]:my-2 [&_pre]:max-w-full [&_pre]:overflow-x-auto [&_pre]:whitespace-pre [&_pre]:rounded-lg [&_pre]:border [&_pre]:border-border/70 [&_pre]:bg-background/80 [&_pre]:p-3",
        "[&_pre_code]:bg-transparent [&_pre_code]:p-0",
        "[&_table]:my-2 [&_table]:w-full [&_table]:border-collapse [&_table]:text-xs [&_table]:block [&_table]:overflow-x-auto",
        "[&_td]:border [&_td]:border-border/60 [&_td]:px-2 [&_td]:py-1",
        "[&_th]:border [&_th]:border-border/60 [&_th]:bg-muted/40 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left",
        "[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5",
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          code({ className, children, ...props }) {
            const text = String(children).replace(/\n$/, "");
            if (!streaming) {
              if (/language-mermaid/.test(className || ""))
                return <MermaidBlock code={text} />;
              if (/language-math/.test(className || ""))
                return (
                  <KaTeXBlock
                    math={text}
                    display={/math-display/.test(className || "")}
                  />
                );
            }
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function ChatBubble({
  message,
  reasoningOpen,
  onReasoningToggle,
  streaming,
}: {
  message: ChatMessage;
  reasoningOpen: boolean;
  onReasoningToggle: (open: boolean) => void;
  streaming?: boolean;
}) {
  const isUser = message.role === "user";
  const isThinking = !isUser && message.reasoning && !message.content;
  const hasFinishedThinking = !isUser && message.reasoning && !!message.content;
  const attachments = messageAttachments(message);

  return (
    <div
      className={cn("flex", isUser ? "justify-end" : "justify-start")}
      data-testid={isUser ? "chat-bubble-user" : "chat-bubble-assistant"}
    >
      <div className="w-full min-w-0 max-w-[92%] md:max-w-[82%]">
        <div className="mb-1 flex items-center gap-2 px-1 text-xs text-muted-foreground">
          {isUser ? (
            <User className="h-3.5 w-3.5" />
          ) : (
            <Bot className="h-3.5 w-3.5" />
          )}
          <span>{isUser ? "You" : "Assistant"}</span>
          {message.model ? <span>· {shortName(message.model)}</span> : null}
        </div>

        {/* Thinking indicator — click to expand and watch reasoning stream live */}
        {isThinking ? (
          <div className="mb-2 rounded-lg border border-dashed">
            <button
              type="button"
              className="flex w-full items-center gap-2 px-3 py-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => onReasoningToggle(!reasoningOpen)}
            >
              <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
              <span>Thinking…</span>
              <ChevronDown
                className={cn(
                  "ml-auto h-3 w-3 transition-transform",
                  reasoningOpen ? "" : "-rotate-90",
                )}
              />
            </button>
            {reasoningOpen && message.reasoning ? (
              <div className="border-t border-dashed px-3 pb-2 pt-1">
                <ScrollArea className="max-h-60">
                  <div className="whitespace-pre-wrap text-xs leading-5 text-muted-foreground">
                    {message.reasoning}
                  </div>
                </ScrollArea>
              </div>
            ) : null}
          </div>
        ) : null}

        {/* Collapsed reasoning accordion — shown after thinking is done */}
        {hasFinishedThinking ? (
          <Accordion
            type="single"
            collapsible
            value={reasoningOpen ? "reasoning" : ""}
            onValueChange={(v) => onReasoningToggle(v === "reasoning")}
            className="mb-2"
          >
            <AccordionItem
              value="reasoning"
              className="rounded-lg border border-dashed px-3"
            >
              <AccordionTrigger className="py-2 text-xs text-muted-foreground hover:no-underline">
                <span className="flex items-center gap-1.5">
                  <Sparkles className="h-3 w-3" />
                  Thought for a moment
                </span>
              </AccordionTrigger>
              <AccordionContent>
                <div className="whitespace-pre-wrap text-xs leading-5 text-muted-foreground">
                  {message.reasoning}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        ) : null}

        {isUser && attachments.length > 0 ? (
          <div className="mb-2 space-y-2">
            {attachments.map((attachment) =>
              attachment.kind === "image" ? (
                <div key={attachment.id}>
                  <img
                    src={attachment.dataUrl}
                    alt={attachment.fileName || "Attached image"}
                    className="max-h-48 rounded-lg border object-contain"
                  />
                </div>
              ) : attachment.kind === "audio" ? (
                <div
                  key={attachment.id}
                  className="rounded-lg border bg-muted/40 p-3"
                >
                  <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
                    <FileAudio className="h-3.5 w-3.5" />
                    <span>{attachment.fileName || "Audio attachment"}</span>
                  </div>
                  <audio controls className="w-full" src={attachment.dataUrl} />
                </div>
              ) : (
                <div
                  key={attachment.id}
                  className="rounded-lg border bg-muted/40 p-3 text-sm"
                >
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <File className="h-3.5 w-3.5" />
                    <span>{attachment.fileName || "File attachment"}</span>
                  </div>
                </div>
              ),
            )}
          </div>
        ) : null}

        {/* Main content */}
        {isUser || message.content ? (
          <div
            className={cn(
              "rounded-lg border px-4 py-3 text-sm leading-6 break-words",
              isUser
                ? "bg-muted whitespace-pre-wrap"
                : message.error
                  ? "border-destructive/50 text-destructive"
                  : "bg-background",
            )}
          >
            {message.content ? (
              <MarkdownMessage
                content={message.content}
                streaming={streaming}
              />
            ) : !isUser ? (
              "..."
            ) : (
              ""
            )}
          </div>
        ) : null}

        {message.stats ? (
          <div className="mt-1 px-1 text-xs text-muted-foreground">
            {message.stats}
          </div>
        ) : null}
      </div>
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

function EmptyPanel({ text }: { text: string }) {
  return (
    <Card>
      <CardContent className="p-4 text-sm text-muted-foreground">
        {text}
      </CardContent>
    </Card>
  );
}

function DashboardPanelEmpty({
  icon,
  title,
  description,
}: {
  icon: ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="flex h-full min-h-[18rem] flex-col items-center justify-center rounded-md border border-dashed bg-muted/20 px-4 text-center md:min-h-[20rem]">
      <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-full border bg-background text-muted-foreground">
        {icon}
      </div>
      <div className="text-sm font-medium">{title}</div>
      <div className="mt-1 max-w-md text-xs text-muted-foreground">
        {description}
      </div>
    </div>
  );
}

function NodeSidebar({
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
    node.hotModels.forEach((name, index) => order.set(name, index));
    node.servingModels.forEach((name, index) => {
      if (!order.has(name)) order.set(name, node.hotModels.length + index);
    });
    node.requestedModels.forEach((name, index) => {
      if (!order.has(name)) order.set(name, node.hotModels.length + node.servingModels.length + index);
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
      .sort((a, b) => (order.get(a.name) ?? Number.MAX_SAFE_INTEGER) - (order.get(b.name) ?? Number.MAX_SAFE_INTEGER));
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
            {node.gpus.length > 0 ? (
              <div className="grid gap-3">
                {node.gpus.map((gpu, index) => (
                  <ModelMetaItem
                    key={`${gpu.name}-${index}`}
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

function ModelSidebar({
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
            value={
              model.size_gb > 0 ? `${model.size_gb.toFixed(1)} GB` : "Unknown"
            }
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
                <p className="leading-7 text-muted-foreground">
                  {model.description}
                </p>
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
                        <span
                          aria-hidden="true"
                          className="text-sm leading-none"
                        >
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
                The same model file shown as the mesh shorthand, repository
                path, and pinned revision.
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
                          className="text-left underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-sm"
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
              ? ` · active ${formatAge(model.last_active_secs_ago)} ago`
              : ""}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function CapabilityBadge({
  label,
  icon,
  tooltip,
}: {
  label: string;
  icon: ReactNode;
  tooltip?: string;
}) {
  const badge = (
    <Badge className="gap-1.5 rounded-full border-sky-300 bg-sky-50 px-3 py-1.5 text-xs font-medium text-foreground dark:border-sky-900 dark:bg-sky-950/30">
      {icon}
      {label}
    </Badge>
  );
  if (!tooltip) return badge;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {/* eslint-disable-next-line jsx-a11y/prefer-tag-over-role -- span avoids nested <button> when used inside StatCard */}
        <span
          role="button"
          tabIndex={0}
          className="inline-flex rounded-full bg-transparent p-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {badge}
        </span>
      </TooltipTrigger>
      <TooltipContent side="bottom" align="center" sideOffset={8}>
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

function ModelFactCard({
  title,
  value,
  icon,
  tooltip,
}: {
  title: string;
  value: string;
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

function ModelMetaItem({
  label,
  value,
  icon,
  copyValue,
}: {
  label: string;
  value: string;
  icon?: ReactNode;
  copyValue?: string;
}) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async () => {
    if (!copyValue) return;
    try {
      await navigator.clipboard.writeText(copyValue);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }, [copyValue]);

  return (
    <div className="rounded-lg border bg-muted/25 px-3 py-2">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          {icon ? <span className="shrink-0">{icon}</span> : null}
          <span>{label}</span>
        </div>
        {copyValue ? (
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="h-6 w-6 rounded-full p-0 text-muted-foreground hover:text-foreground"
            onClick={() => void copy()}
            aria-label={copied ? `${label} copied` : `Copy ${label}`}
            title={copied ? "Copied" : `Copy ${label}`}
          >
            {copied ? (
              <Check className="h-3 w-3" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </Button>
        ) : null}
      </div>
      <div className="mt-1 text-sm font-medium [overflow-wrap:anywhere]">
        {value}
      </div>
    </div>
  );
}

function ModelMetaLinkItem({
  label,
  href,
  text,
  icon,
}: {
  label: string;
  href: string;
  text: string;
  icon?: ReactNode;
}) {
  return (
    <div className="rounded-lg border bg-muted/25 px-3 py-2">
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
        {icon ? <span className="shrink-0">{icon}</span> : null}
        <span>{label}</span>
      </div>
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        className="mt-1 inline-flex items-center gap-1.5 text-sm font-medium underline-offset-4 hover:text-foreground hover:underline [overflow-wrap:anywhere]"
      >
        <span>{text}</span>
        <ExternalLink className="h-3 w-3 shrink-0" />
      </a>
    </div>
  );
}

function StatusPill({
  label,
  tone,
  dot = false,
  icon,
  className,
  tooltip,
}: {
  label: string;
  tone: "warm" | "cold" | "good" | "info" | "warn" | "bad" | "neutral";
  dot?: boolean;
  icon?: ReactNode;
  className?: string;
  tooltip?: string;
}) {
  const badge = (
    <Badge
      className={cn(
        "h-5 shrink-0 rounded-full px-2 text-[10px] font-medium",
        statusPillToneClass(tone),
        dot || icon ? "gap-1" : "",
        className,
      )}
    >
      {dot ? <span className="h-1.5 w-1.5 rounded-full bg-current" /> : null}
      {icon}
      {label}
    </Badge>
  );
  if (!tooltip) return badge;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {/* eslint-disable-next-line jsx-a11y/prefer-tag-over-role -- span avoids nested <button> when used inside StatCard */}
        <span
          role="button"
          tabIndex={0}
          className="inline-flex rounded-full bg-transparent p-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {badge}
        </span>
      </TooltipTrigger>
      <TooltipContent side="bottom" align="center" sideOffset={8}>
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

function meshGpuVram(status: StatusPayload | null) {
  if (!status) return 0;
  return (
    overviewVramGb(status.is_client, status.my_vram_gb) +
    (status.peers || []).reduce(
      (sum, peer) => sum + overviewVramGb(peer.role === "Client", peer.vram_gb),
      0,
    )
  );
}

function overviewVramGb(isClient: boolean, vramGb?: number | null) {
  if (isClient) return 0;
  return Math.max(0, vramGb || 0);
}

function shortName(name: string) {
  return (name || "").replace(/-Q\w+$/, "").replace(/-Instruct/, "");
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
  // Prefer canonical Hugging Face style: repo@rev/file
  if (model.source_ref && model.source_file) {
    return `${model.source_ref}@${model.source_revision}/${model.source_file}`;
  }
  // Fallback: append revision to whatever name we have
  const fullName = modelFullFileName(model);
  if (!fullName) return null;
  return `${fullName}@${model.source_revision}`;
}

function formatAge(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

function formatContextLength(value?: number) {
  if (!value || !Number.isFinite(value)) return "Unknown";
  if (value >= 1000) {
    const rounded = value / 1000;
    return Number.isInteger(rounded) ? `${rounded}K` : `${rounded.toFixed(1)}K`;
  }
  return `${value}`;
}

function formatLatency(value?: number | null) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const ms = Math.round(Number(value));
  if (ms <= 0) return "<1 ms";
  return `${ms} ms`;
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

function modelStatusTooltip(status?: string) {
  if (status === "warm") {
    return "Loaded and serving in the mesh.";
  }
  if (status === "cold") {
    return "Downloaded locally, but not currently serving.";
  }
  return "Current model availability in the mesh.";
}

function nodeRoleTooltip(role: string) {
  if (role === 'Host') {
    return 'Coordinates requests and mesh routing for this node.';
  }
  if (role === 'Worker') {
    return 'Contributes VRAM and compute capacity to the mesh.';
  }
  if (role === 'Client') {
    return 'Sends requests, but does not contribute VRAM.';
  }
  return 'Connected to the mesh, but not actively serving a model.';
}

function nodeLatencyTooltip(self: boolean) {
  if (self) {
    return 'This is the local node.';
  }
  return 'Observed round-trip latency from your node to this peer.';
}

function nodeVramTooltip(role: string) {
  if (role === 'Client') {
    return 'Clients do not contribute serving VRAM to the mesh.';
  }
  return 'Serving VRAM contributed by this node to the mesh.';
}

function nodeMeshShareTooltip(role: string) {
  if (role === 'Client') {
    return 'Clients do not contribute serving VRAM, so they have no mesh share.';
  }
  return 'Approximate share of total serving VRAM contributed by this node.';
}

function nodeHotModelsTooltip() {
  return 'Models this node is hosting, serving, or requesting right now.';
}

function nodeModelFlagTooltip(flag: string) {
  if (flag === 'Serving') {
    return 'This node is actively serving requests for this model.';
  }
  if (flag === 'Hosted') {
    return 'This node has the model available locally for routing.';
  }
  if (flag === 'Requested') {
    return 'This model has been requested on this node, but is not active yet.';
  }
  return 'Current model role on this node.';
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

function topologyStatusTone(
  status: string,
): "good" | "info" | "warn" | "bad" | "neutral" {
  if (status === "Serving" || status === "Serving (split)") return "good";
  if (status === "Client") return "info";
  if (status === "Host") return "info";
  if (status === "Idle" || status === "Standby") return "neutral";
  if (status === "Worker (split)") return "warn";
  return "neutral";
}

function topologyStatusTooltip(status: string) {
  if (status === "Serving") {
    return "Actively serving a model.";
  }
  if (status === "Serving (split)") {
    return "Serving a split model with the mesh.";
  }
  if (status === "Worker (split)") {
    return "Contributing compute to a split model.";
  }
  if (status === "Host") {
    return "Coordinating requests for the mesh.";
  }
  if (status === "Client") {
    return "Sends requests, but does not contribute VRAM.";
  }
  if (status === "Idle" || status === "Standby") {
    return "Connected, but not serving a model.";
  }
  return "Current serving role.";
}

function statusPillToneClass(
  tone: "warm" | "cold" | "good" | "info" | "warn" | "bad" | "neutral",
) {
  if (tone === "warm" || tone === "good") {
    return "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
  }
  if (tone === "cold" || tone === "info") {
    return "border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300";
  }
  if (tone === "warn") {
    return "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300";
  }
  if (tone === "bad") {
    return "border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300";
  }
  return "border-border bg-muted text-foreground";
}
