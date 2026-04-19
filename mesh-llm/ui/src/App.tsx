import { useEffect, useMemo, useRef, useState } from "react";

import {
  TooltipProvider,
} from "./components/ui/tooltip";
import { AppHeader } from "./features/app-shell/components/AppHeader";
import {
  type TopSection,
} from "./features/app-shell/lib/routes";
import {
  applyThemeMode,
  localRoutableModels,
  overviewVramGb,
  peerAssignedModels,
  peerPrimaryModel,
  peerRoutableModels,
  peerStatusLabel,
  readThemeMode,
} from "./features/app-shell/lib/status-helpers";
import { useAppRouting } from "./features/app-shell/hooks/useAppRouting";
import {
  useStatusStream,
  type MeshModel,
} from "./features/app-shell/hooks/useStatusStream";
import type { ModelServingStat, ThemeMode } from "./features/app-shell/lib/status-types";
import { DashboardPage } from "./features/dashboard/components/DashboardPage";
import { useChatSession } from "./features/chat/hooks/useChatSession";
import {
  attachmentForMessage,
} from "./features/chat/lib/message-content";
import {
  describeImageAttachmentForPrompt,
  describeRenderedPagesAsText,
} from "./features/chat/lib/vision-describe";
import { ChatPage } from "./features/chat/components/ChatPage";
import { cn } from "./lib/utils";
import type {
  TopologyNode,
} from "./features/app-shell/lib/topology-types";
import githubBlackLogo from "./assets/icons/github-invertocat-black.svg";
import githubWhiteLogo from "./assets/icons/github-invertocat-white.svg";

export {
  attachmentForMessage,
  ChatPage,
  describeImageAttachmentForPrompt,
  describeRenderedPagesAsText,
};

const FLY_DOMAINS = [
  "mesh-llm-console.fly.dev",
  "www.mesh-llm.com",
  "www.anarchai.org",
];

const THEME_STORAGE_KEY = "mesh-llm-theme";

export function App() {
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => readThemeMode(THEME_STORAGE_KEY));
  const topologyFirstSeenAtRef = useRef<Map<string, number>>(new Map());
  const { status, statusError, meshModels, modelsLoading } = useStatusStream();
  const { section, routedChatId, navigateToSection, pushChatRoute, replaceChatRoute } =
    useAppRouting();
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
  const meshModelByName = useMemo(() => {
    const entries = meshModels.map((model) => [model.name, model] as const);
    return Object.fromEntries(entries) as Record<string, MeshModel>;
  }, [meshModels]);
  const selectedChatModel = selectedModel || warmModels[0] || status?.model_name || "";
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
      if (startedAtUnix == null || !Number.isFinite(startedAtUnix) || startedAtUnix <= 0) {
        return null;
      }
      const startedAtMs = startedAtUnix > 1_000_000_000_000 ? startedAtUnix : startedAtUnix * 1000;
      return Math.max(0, Math.floor((now - startedAtMs) / 1000));
    };
    const nodes: TopologyNode[] = [];
    if (status.node_id) {
      const selfFirstSeenAt = ensureFirstSeenAt(status.node_id);
      const selfStartedAtUnix = status.local_instances?.find((instance) => instance.is_self)
        ?.started_at_unix;
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
        ageSeconds:
          elapsedSecondsFrom(selfStartedAtUnix) ?? elapsedSecondsFrom(selfFirstSeenAt),
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
        ageSeconds: elapsedSecondsFrom(ensureFirstSeenAt(p.id)),
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

  return (
    <TooltipProvider>
      <div className="h-screen overflow-hidden bg-background [height:100svh] [padding-top:env(safe-area-inset-top)] [padding-bottom:env(safe-area-inset-bottom)]">
        <div className="flex h-full min-h-0 flex-col">
          <AppHeader
            sections={sections}
            section={section}
            setSection={(next) => navigateToSection(next, activeConversationId || null)}
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
  );
}
