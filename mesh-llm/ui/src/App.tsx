import { Brain, Boxes, Cpu, ImagePlus, Network, Sparkles, Volume2 } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";

import { Badge } from "./components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "./components/ui/toggle-group";
import { TooltipProvider } from "./components/ui/tooltip";
import { AppHeader } from "./features/app-shell/components/AppHeader";
import { CommandBarModal } from "./features/app-shell/components/command-bar/CommandBarModal";
import { CommandBarProvider } from "./features/app-shell/components/command-bar/CommandBarProvider";
import type {
  CommandBarMode,
  CommandBarResultContainerProps,
  CommandBarResultItemProps,
} from "./features/app-shell/components/command-bar/command-bar-types";
import { type TopSection } from "./features/app-shell/lib/routes";
import {
  applyThemeMode,
  formatLiveNodeState,
  localRoutableModels,
  modelDisplayName,
  overviewVramGb,
  peerAssignedModels,
  peerPrimaryModel,
  peerRoutableModels,
  readThemeMode,
  shortName,
} from "./features/app-shell/lib/status-helpers";
import { useAppRouting } from "./features/app-shell/hooks/useAppRouting";
import {
  useStatusStream,
  type MeshModel,
} from "./features/app-shell/hooks/useStatusStream";
import type {
  ModelServingStat,
  ThemeMode,
} from "./features/app-shell/lib/status-types";
import { DashboardPage } from "./features/dashboard/components/DashboardPage";
import { useChatSession } from "./features/chat/hooks/useChatSession";
import { attachmentForMessage } from "./features/chat/lib/message-content";
import {
  describeImageAttachmentForPrompt,
  describeRenderedPagesAsText,
} from "./features/chat/lib/vision-describe";
import { ChatPage } from "./features/chat/components/ChatPage";
import { cn } from "./lib/utils";
import type { TopologyNode } from "./features/app-shell/lib/topology-types";
import githubBlackLogo from "./assets/icons/github-invertocat-black.svg";
import githubWhiteLogo from "./assets/icons/github-invertocat-white.svg";

// Playground is dev-only. We conditionally create the lazy component based on import.meta.env.DEV
// so Vite can completely exclude it from production builds (true tree-shaking).
// In dev: creates lazy component for code-splitting + HMR
// In prod: returns null immediately, zero bundle impact
// CRITICAL: Must memoize at module scope to avoid creating new component type on every render
const LazyPlaygroundPage = import.meta.env.DEV
  ? lazy(() => import("./features/app-shell/playground/components/PlaygroundPage"))
  : null;

const DevOnlyPlayground = () => {
  if (!LazyPlaygroundPage) {
    return null; // Production: should never render
  }
  return <LazyPlaygroundPage />;
};

export {
  attachmentForMessage,
  ChatPage,
  describeImageAttachmentForPrompt,
  describeRenderedPagesAsText,
};

function modelCommandBarStatusLabel(status: MeshModel["status"]) {
  if (status === "warm") return "Warm";
  if (status === "cold") return "Cold";
  return status || "Unknown";
}

function modelCommandBarStatusClass(status: MeshModel["status"]) {
  if (status === "warm") {
    return "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
  }
  if (status === "cold") {
    return "border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300";
  }
  return "border-border bg-muted text-foreground";
}

function modelCommandBarCapabilityLabels(model: MeshModel) {
  return [
    model.multimodal ? "multimodal" : null,
    model.vision ? "vision" : null,
    model.audio ? "audio" : null,
    model.reasoning ? "reasoning" : null,
    model.moe ? "moe" : null,
    model.tool_use ? "tool use" : null,
  ].filter((label): label is string => Boolean(label));
}

function modelCommandBarSearchText(model: MeshModel) {
  return [
    model.name,
    modelDisplayName(model),
    ...modelCommandBarCapabilityLabels(model),
  ]
    .filter(Boolean)
    .join(" ");
}

function ModelCommandBarResultContainer({
  children,
}: CommandBarResultContainerProps<MeshModel>) {
  return (
    <div className="max-h-[420px] overflow-y-auto p-2">
      <div className="space-y-2">{children}</div>
    </div>
  );
}

function ModelCommandBarResultItem({
  item,
  selected,
}: CommandBarResultItemProps<MeshModel>) {
  const displayName = modelDisplayName(item) || item.name;
  const capabilityLabels = modelCommandBarCapabilityLabels(item);

  return (
    <div className="flex min-w-0 items-start gap-3">
      <div
        className={cn(
          "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md border text-muted-foreground",
          selected
            ? "border-primary/30 bg-primary/10 text-primary"
            : "border-border bg-muted/40",
        )}
      >
        <Sparkles className="h-3.5 w-3.5" />
      </div>
      <div className="grid min-w-0 flex-1 grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3">
        <div className="min-w-0 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <div className="text-sm font-medium leading-5 [overflow-wrap:anywhere]">
              {shortName(displayName)}
            </div>
            <Badge
              className={cn(
                "h-5 shrink-0 rounded-full px-2 text-[10px] font-medium",
                modelCommandBarStatusClass(item.status),
              )}
            >
              {modelCommandBarStatusLabel(item.status)}
            </Badge>
          </div>
          <div className="text-xs leading-4 text-muted-foreground [overflow-wrap:anywhere]">
            {item.name}
          </div>
          {capabilityLabels.length > 0 ? (
            <div className="flex flex-wrap items-center gap-1.5 pt-0.5 text-xs text-muted-foreground">
              {capabilityLabels.map((label) => (
                <Badge
                  key={`${item.name}-${label}`}
                  className="min-w-0 max-w-full items-start justify-start gap-1 rounded-md border-border/60 bg-muted/40 px-2 py-1 text-[10px] uppercase tracking-[0.08em] text-muted-foreground"
                >
                  {label === "vision" ? (
                    <ImagePlus className="mt-0.5 h-3 w-3 shrink-0" />
                  ) : null}
                  {label === "audio" ? (
                    <Volume2 className="mt-0.5 h-3 w-3 shrink-0" />
                  ) : null}
                  {label === "reasoning" ? (
                    <Brain className="mt-0.5 h-3 w-3 shrink-0" />
                  ) : null}
                  {label === "moe" ? (
                    <Boxes className="mt-0.5 h-3 w-3 shrink-0" />
                  ) : null}
                  {label === "multimodal" ? (
                    <Sparkles className="mt-0.5 h-3 w-3 shrink-0" />
                  ) : null}
                  <span className="min-w-0 break-words leading-tight [overflow-wrap:anywhere]">
                    {label}
                  </span>
                </Badge>
              ))}
            </div>
          ) : null}
        </div>
        <div className="flex shrink-0 flex-col items-end gap-0.5 pt-0.5 text-xs leading-4 text-muted-foreground">
          <span className="inline-flex items-center gap-1 whitespace-nowrap">
            <Network className="h-3.5 w-3.5" />
            <span>
              {item.node_count} node{item.node_count === 1 ? "" : "s"}
            </span>
          </span>
          <span className="inline-flex items-center gap-1 whitespace-nowrap">
            <Cpu className="h-3.5 w-3.5" />
            <span>{item.size_gb.toFixed(1)} GB</span>
          </span>
        </div>
      </div>
    </div>
  );
}

const FLY_DOMAINS = [
  "mesh-llm-console.fly.dev",
  "www.mesh-llm.com",
  "www.anarchai.org",
];

const THEME_STORAGE_KEY = "mesh-llm-theme";

type ModelCapabilityFilterId =
  | "multimodal"
  | "vision"
  | "audio"
  | "reasoning"
  | "moe"
  | "tool_use";

type ModelStatusFilter = "all" | "warm" | "cold";

const MODEL_CAPABILITY_FILTERS: ReadonlyArray<{
  id: ModelCapabilityFilterId;
  label: string;
}> = [
  { id: "multimodal", label: "Multimodal" },
  { id: "vision", label: "Vision" },
  { id: "audio", label: "Audio" },
  { id: "reasoning", label: "Reasoning" },
  { id: "moe", label: "MoE" },
  { id: "tool_use", label: "Tool use" },
];

function modelMatchesCapabilityFilter(
  model: MeshModel,
  capability: ModelCapabilityFilterId,
) {
  if (capability === "multimodal") return Boolean(model.multimodal);
  if (capability === "vision") return Boolean(model.vision);
  if (capability === "audio") return Boolean(model.audio);
  if (capability === "reasoning") return Boolean(model.reasoning);
  if (capability === "moe") return Boolean(model.moe);
  return Boolean(model.tool_use);
}

function modelMatchesStructuredFilters(
  model: MeshModel,
  capabilityFilters: readonly ModelCapabilityFilterId[],
  statusFilter: ModelStatusFilter,
) {
  const matchesStatus = statusFilter === "all" || model.status === statusFilter;
  const matchesCapabilities = capabilityFilters.every((capability) =>
    modelMatchesCapabilityFilter(model, capability),
  );

  return matchesStatus && matchesCapabilities;
}

function ModelCommandBarFilterBar({
  capabilityFilters,
  onCapabilityFiltersChange,
  statusFilter,
  onStatusFilterChange,
}: {
  capabilityFilters: readonly ModelCapabilityFilterId[];
  onCapabilityFiltersChange: (nextValue: ModelCapabilityFilterId[]) => void;
  statusFilter: ModelStatusFilter;
  onStatusFilterChange: (nextValue: ModelStatusFilter) => void;
}) {
  return (
    <div className="flex items-center gap-3 pr-1">
      <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
        <span className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">
          Capabilities
        </span>
        <ToggleGroup
          type="multiple"
          value={[...capabilityFilters]}
          onValueChange={(value: string[]) =>
            onCapabilityFiltersChange(value as ModelCapabilityFilterId[])
          }
          aria-label="Model capability filters"
          className="flex min-w-0 flex-nowrap gap-1.5 overflow-x-auto pb-1"
        >
          {MODEL_CAPABILITY_FILTERS.map((filter) => (
            <ToggleGroupItem
              key={filter.id}
              value={filter.id}
              className="h-7 shrink-0 rounded-md px-2 text-[11px]"
              aria-label={filter.label}
            >
              {filter.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      <div className="flex shrink-0 items-center gap-2">
        <span className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">
          Status
        </span>
        <Select
          value={statusFilter}
          onValueChange={(value) =>
            onStatusFilterChange(value as ModelStatusFilter)
          }
        >
          <SelectTrigger
            aria-label="Model status filter"
            className="h-7 w-[88px] rounded-md text-[11px]"
          >
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
  );
}

export function App() {
  const [themeMode, setThemeMode] = useState<ThemeMode>(() =>
    readThemeMode(THEME_STORAGE_KEY),
  );
  const topologyFirstSeenAtRef = useRef<Map<string, number>>(new Map());
  const [modelCapabilityFilters, setModelCapabilityFilters] = useState<
    ModelCapabilityFilterId[]
  >([]);
  const [modelStatusFilter, setModelStatusFilter] =
    useState<ModelStatusFilter>("all");
  const { status, statusError, meshModels, modelsLoading } = useStatusStream();
  const {
    section,
    routedChatId,
    navigateToSection,
    pushChatRoute,
    replaceChatRoute,
  } = useAppRouting();
  const chatSession = useChatSession({
    status,
    meshModels,
    section,
    routedChatId,
    pushChatRoute,
    replaceChatRoute,
  });
  const {
    selectedModel,
    setSelectedModel,
    warmModels,
    selectedModelAudio,
    selectedModelMultimodal,
    composerError,
    setComposerError,
    attachmentSendIssue,
    attachmentPreparationMessage,
    pendingAttachments,
    setPendingAttachments,
    conversations,
    activeConversationId,
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
    createNewConversation,
    selectConversation,
    renameConversation,
    deleteConversation,
    clearAllConversations,
    stopStreaming,
    regenerateLastResponse,
    handleSubmit,
  } = chatSession;
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
      if (peer.state === "client") continue;
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
  const meshModelByName = useMemo(() => {
    const entries = meshModels.map((model) => [model.name, model] as const);
    return Object.fromEntries(entries) as Record<string, MeshModel>;
  }, [meshModels]);
  const filteredCommandBarModels = useMemo(() => {
    return meshModels.filter((model) =>
      modelMatchesStructuredFilters(
        model,
        modelCapabilityFilters,
        modelStatusFilter,
      ),
    );
  }, [meshModels, modelCapabilityFilters, modelStatusFilter]);
  const selectedChatModel =
    selectedModel || warmModels[0] || status?.model_name || "";
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
  const isPublicMesh = (() => {
    if (!status) return false;
    if (status.publication_state === "public") return true;
    if (status.publication_state != null) return false;
    return !!status.nostr_discovery;
  })();

  useEffect(() => {
    applyThemeMode(themeMode);
    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode);
  }, [themeMode]);

  useEffect(() => {
    if (themeMode !== "auto") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => applyThemeMode("auto");
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [themeMode]);
  const topologyNodes = useMemo<TopologyNode[]>(() => {
    if (!status) return [];
    const now = Date.now();
    const topologyFirstSeenAt = topologyFirstSeenAtRef.current;
    const ensureFirstSeenAt = (nodeId: string) => {
      const existing = topologyFirstSeenAt.get(nodeId);
      if (existing != null) return existing;
      topologyFirstSeenAt.set(nodeId, now);
      return now;
    };
    const elapsedSecondsFrom = (startedAtUnix: number | null | undefined) => {
      if (
        startedAtUnix == null ||
        !Number.isFinite(startedAtUnix) ||
        startedAtUnix <= 0
      ) {
        return null;
      }
      const startedAtMs =
        startedAtUnix > 1_000_000_000_000
          ? startedAtUnix
          : startedAtUnix * 1000;
      return Math.max(0, Math.floor((now - startedAtMs) / 1000));
    };
    const nodes: TopologyNode[] = [];
    if (status.node_id) {
      const selfFirstSeenAt = ensureFirstSeenAt(status.node_id);
      const selfStartedAtUnix = status.local_instances?.find(
        (instance) => instance.is_self,
      )?.started_at_unix;
      nodes.push({
        id: status.node_id,
        vram: overviewVramGb(status.node_state === "client", status.my_vram_gb),
        state: status.node_state,
        self: true,
        host: status.is_host,
        client: status.node_state === "client",
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
        ageSeconds:
          elapsedSecondsFrom(status.first_joined_mesh_ts)
          ?? elapsedSecondsFrom(selfStartedAtUnix)
          ?? elapsedSecondsFrom(selfFirstSeenAt),
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
        vram: overviewVramGb(p.state === "client", p.vram_gb),
        state: p.state,
        self: false,
        host: /^Host/.test(p.role),
        client: p.state === "client",
        serving: peerPrimaryModel(p),
        servingModels: pModels,
        statusLabel: formatLiveNodeState(p.state),
        ageSeconds: elapsedSecondsFrom(p.first_joined_mesh_ts) ?? elapsedSecondsFrom(ensureFirstSeenAt(p.id)),
        latencyMs: p.rtt_ms ?? null,
        hostname: p.hostname,
        isSoc: p.is_soc,
        gpus: p.gpus,
      });
    }
    const activeIds = new Set(nodes.map((n) => n.id));
    for (const key of topologyFirstSeenAt.keys()) {
      if (!activeIds.has(key)) topologyFirstSeenAt.delete(key);
    }
    return nodes;
  }, [status]);

  const sections: Array<{ key: TopSection; label: string }> = [
    { key: "dashboard", label: "Network" },
    { key: "chat", label: "Chat" },
  ];
  const commandBarModes = useMemo<readonly CommandBarMode<MeshModel>[]>(() => {
    return [
      {
        id: "models",
        label: "Models",
        leadingIcon: Sparkles,
        source: filteredCommandBarModels,
        getItemKey: (model) => model.name,
        getSearchText: (model) => modelCommandBarSearchText(model),
        getKeywords: (model) =>
          [
            model.status,
            model.architecture ?? "",
            model.fit_label ?? "",
            model.description ?? "",
          ].filter(Boolean),
        ResultContainer: ModelCommandBarResultContainer,
        ResultItem: ModelCommandBarResultItem,
        onSelect: (model) => {
          setSelectedModel(model.name);
        },
      },
    ];
  }, [filteredCommandBarModels, setSelectedModel]);
  const commandBarEmptyMessage =
    modelsLoading && meshModels.length === 0
      ? "Loading model catalog…"
      : meshModels.length === 0
        ? "No model catalog data yet."
        : filteredCommandBarModels.length === 0
          ? "No models match the current filters."
          : "No models match this search.";
  const showPlayground = import.meta.env.DEV && section === "playground";
  const showDashboard =
    section === "dashboard" || (!import.meta.env.DEV && section === "playground");
  const modelCommandBarFilterBar = (
    <ModelCommandBarFilterBar
      capabilityFilters={modelCapabilityFilters}
      onCapabilityFiltersChange={setModelCapabilityFilters}
      statusFilter={modelStatusFilter}
      onStatusFilterChange={setModelStatusFilter}
    />
  );

  return (
    <CommandBarProvider>
      <TooltipProvider>
        <div className="h-screen overflow-hidden bg-background [height:100svh] [padding-top:env(safe-area-inset-top)] [padding-bottom:env(safe-area-inset-bottom)]">
          <div className="flex h-full min-h-0 flex-col">
            <AppHeader
              sections={sections}
              section={section}
              setSection={(next) =>
                navigateToSection(next, activeConversationId || null)
              }
              themeMode={themeMode}
              setThemeMode={setThemeMode}
              statusError={statusError}
              inviteWithModelCommand={inviteWithModelCommand}
              inviteWithModelName={inviteWithModelName}
              inviteClientCommand={inviteClientCommand}
              inviteToken={inviteToken}
              apiDirectUrl={apiDirectUrl}
              isPublicMesh={isPublicMesh}
            />
            <CommandBarModal
              modes={commandBarModes}
              behavior="distinct"
              defaultModeId="models"
              title="Switch models"
              description="Search the mesh model catalog and select a model without leaving the current view."
              placeholder="Search models"
              emptyMessage={commandBarEmptyMessage}
              interstitial={modelCommandBarFilterBar}
            />

            <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
              {section === "chat" ? (
                <div className="mx-auto flex min-h-0 min-w-0 w-full max-w-7xl flex-1 flex-col overflow-hidden p-2 md:p-4">
                  <ChatPage
                    status={status}
                    inviteToken={status?.token ?? ""}
                    isPublicMesh={isPublicMesh}
                    isFlyHosted={isFlyHosted}
                    inflightRequests={status?.inflight_requests ?? 0}
                    warmModels={warmModels}
                    meshModelByName={meshModelByName}
                    modelStatsByName={modelStatsByName}
                    selectedModel={selectedModel}
                    setSelectedModel={setSelectedModel}
                    selectedModelNodeCount={selectedModelNodeCount}
                    selectedModelVramGb={selectedModelVramGb}
                    selectedModelAudio={selectedModelAudio}
                    selectedModelMultimodal={selectedModelMultimodal}
                    composerError={composerError}
                    setComposerError={setComposerError}
                    attachmentSendIssue={attachmentSendIssue}
                    attachmentPreparationMessage={attachmentPreparationMessage}
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

              {showDashboard ? (
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
                      isPublicMesh={isPublicMesh}
                      inviteToken={inviteToken}
                      isLocalhost={isLocalhost}
                    />
                  </div>
                </div>
              ) : null}

              {showPlayground ? (
                <Suspense
                  fallback={
                    <div className="flex min-h-0 items-center justify-center p-8 text-muted-foreground">
                      Loading playground...
                    </div>
                  }
                >
                  <DevOnlyPlayground />
                </Suspense>
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
                      href="https://github.com/Mesh-LLM/mesh-llm/releases"
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
                  href="https://github.com/Mesh-LLM/mesh-llm"
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
    </CommandBarProvider>
  );
}
