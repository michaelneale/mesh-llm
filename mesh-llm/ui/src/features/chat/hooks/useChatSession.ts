import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getAttachmentSendIssue } from "../../../lib/attachments";
import { isPdfMimeType } from "../../../lib/pdf";
import {
  createRafBatcher,
  hasBlobContent,
  parseApiErrorBody,
} from "../../../lib/streaming";
import type { TopSection } from "../../app-shell/lib/routes";
import type { MeshModel, StatusPayload } from "../../app-shell/hooks/useStatusStream";
import { parseDataUrl } from "../lib/chat-attachments";
import { createChatId } from "../lib/chat-id";
import {
  attachmentForMessage,
  buildAttachmentBlocks,
  buildResponsesInput,
} from "../lib/message-content";
import {
  createConversation,
  createInitialChatState,
  findLastUserMessageIndex,
  loadPersistedChatState as loadPersistedChatStateFromStorage,
} from "../lib/chat-storage";
import type {
  AttachmentStatePatch,
  ChatAttachment,
  ChatConversation,
  ChatMessage,
  ChatState,
} from "../lib/chat-types";

const CHAT_CLIENT_ID_STORAGE_KEY = "mesh-llm-chat-client-id";
const DEFAULT_CHAT_TITLE = "New chat";
const CHAT_DB_NAME = "mesh-llm-chat-db";
const CHAT_DB_STORE = "state";
const CHAT_DB_KEY = "chat-state";
const CHAT_SAVE_DEBOUNCE_MS = 500;
const CHAT_MAX_CONVERSATIONS = 80;
const CHAT_MAX_MESSAGES_PER_CONVERSATION = 240;
const CHAT_MAX_TEXT_CHARS = 12000;

function readOrCreateChatClientId(): string {
  if (typeof window === "undefined") return createChatId();
  const stored = window.localStorage.getItem(CHAT_CLIENT_ID_STORAGE_KEY);
  if (stored && stored.trim()) return stored;
  const created = createChatId();
  window.localStorage.setItem(CHAT_CLIENT_ID_STORAGE_KEY, created);
  return created;
}

function deriveConversationTitle(input: string): string {
  const compact = input.replace(/\s+/g, " ").trim();
  if (!compact) return DEFAULT_CHAT_TITLE;
  return compact.length > 52 ? `${compact.slice(0, 52).trimEnd()}...` : compact;
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
    id: typeof item.id === "string" && item.id ? item.id : createChatId(),
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
    extractedText:
      typeof item.extractedText === "string" ? item.extractedText : undefined,
    extractionSummary:
      typeof item.extractionSummary === "string"
        ? item.extractionSummary
        : undefined,
    renderedPageImages:
      Array.isArray(item.renderedPageImages) &&
      item.renderedPageImages.every((value) => typeof value === "string")
        ? (item.renderedPageImages as string[])
        : undefined,
    imageDescription:
      typeof item.imageDescription === "string"
        ? item.imageDescription
        : undefined,
  };
}

function sanitizeMessages(raw: unknown): ChatMessage[] {
  if (!Array.isArray(raw)) return [];
  const sanitized = raw.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const role = (item as { role?: unknown }).role;
    const content = (item as { content?: unknown }).content;
    if ((role !== "user" && role !== "assistant") || typeof content !== "string") {
      return [];
    }
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
      typeof ((item as { audio: { dataUrl?: unknown } }).audio.dataUrl) === "string" &&
      typeof ((item as { audio: { mimeType?: unknown } }).audio.mimeType) === "string"
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
          id: createChatId(),
          kind: "image",
          dataUrl: legacyImage,
          mimeType: parseDataUrl(legacyImage)?.mimeType || "image/jpeg",
          fileName: "image.jpg",
        });
      }
      if (legacyAudio) {
        attachments.push({
          id: createChatId(),
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
            : createChatId(),
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
      const id = typeof obj.id === "string" && obj.id ? obj.id : createChatId();
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
    activeConversationId: conversations[0]?.id ?? "",
  };
}

function openChatDb(): Promise<IDBDatabase | null> {
  if (typeof window === "undefined" || !("indexedDB" in window)) {
    return Promise.resolve(null);
  }
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(CHAT_DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(CHAT_DB_STORE)) {
        db.createObjectStore(CHAT_DB_STORE, { keyPath: "id" });
      }
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
      store.get(CHAT_DB_KEY) as IDBRequest<{ id: string; state?: unknown } | undefined>,
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
  return loadPersistedChatStateFromStorage(readChatStateFromDb);
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

export function useChatSession({
  status,
  meshModels,
  section,
  routedChatId,
  pushChatRoute,
  replaceChatRoute,
}: {
  status: StatusPayload | null;
  meshModels: MeshModel[];
  section: TopSection;
  routedChatId: string | null;
  pushChatRoute: (chatId: string | null) => void;
  replaceChatRoute: (chatId: string | null) => void;
}) {
  const [chatState, setChatState] = useState<ChatState>(() => createInitialChatState());
  const [chatStateHydrated, setChatStateHydrated] = useState(false);
  const [input, setInput] = useState("");
  const [pendingAttachments, setPendingAttachments] = useState<ChatAttachment[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [composerError, setComposerError] = useState<string | null>(null);
  const [reasoningOpen, setReasoningOpen] = useState<Record<string, boolean>>({});
  const [queuedText, setQueuedText] = useState<string | null>(null);
  const chatClientIdRef = useRef<string>(readOrCreateChatClientId());
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const currentAbortRef = useRef<AbortController | null>(null);
  const queuedInputRef = useRef<string | null>(null);

  const activeConversationId = chatState.activeConversationId;
  const conversations = chatState.conversations;
  const activeConversation = useMemo(
    () =>
      conversations.find(
        (conversation) => conversation.id === activeConversationId,
      ) ?? conversations[0],
    [activeConversationId, conversations],
  );
  const messages = activeConversation?.messages ?? [];
  const lastMessageId = messages[messages.length - 1]?.id ?? "";

  const warmModels = useMemo(
    () => meshModels.filter((model) => model.status === "warm").map((model) => model.name),
    [meshModels],
  );
  const audioModels = useMemo(() => {
    const set = new Set<string>();
    for (const model of meshModels) {
      if (model.audio) set.add(model.name);
    }
    return set;
  }, [meshModels]);
  const multimodalModels = useMemo(() => {
    const set = new Set<string>();
    for (const model of meshModels) {
      if (model.multimodal) set.add(model.name);
    }
    return set;
  }, [meshModels]);
  const selectedModelAudio = useMemo(() => {
    if (selectedModel) return audioModels.has(selectedModel);
    return meshModels.some((model) => model.status === "warm" && model.audio);
  }, [audioModels, meshModels, selectedModel]);
  const selectedModelMultimodal = useMemo(() => {
    if (selectedModel) return multimodalModels.has(selectedModel);
    return meshModels.some((model) => model.status === "warm" && model.multimodal);
  }, [meshModels, multimodalModels, selectedModel]);
  const pendingKinds = useMemo(() => {
    const kinds = new Set<"image" | "audio" | "file">();
    for (const attachment of pendingAttachments) {
      if (attachment.extractedText) continue;
      if (attachment.kind === "image") continue;
      if (attachment.renderedPageImages?.length) continue;
      kinds.add(attachment.kind);
    }
    return kinds;
  }, [pendingAttachments]);
  const attachmentPreparationMessage = useMemo(() => {
    const preparingAttachment = pendingAttachments.find(
      (attachment) => attachment.status === "uploading",
    );
    if (!preparingAttachment) return null;
    if (preparingAttachment.kind === "image") {
      return "Describing image in browser… (first time downloads ~230 MB model)";
    }
    if (
      isPdfMimeType(preparingAttachment.mimeType) ||
      preparingAttachment.fileName?.toLowerCase().endsWith(".pdf")
    ) {
      return "Preparing PDF in browser…";
    }
    return "Preparing attachment…";
  }, [pendingAttachments]);
  const attachmentSendIssue = useMemo(() => {
    if (!pendingAttachments.length || !status) return null;
    if (attachmentPreparationMessage) return attachmentPreparationMessage;
    return getAttachmentSendIssue({
      pendingKinds,
      selectedModel,
      warmModels,
      audioModels,
      multimodalModels,
    });
  }, [
    attachmentPreparationMessage,
    audioModels,
    multimodalModels,
    pendingAttachments.length,
    pendingKinds,
    selectedModel,
    status,
    warmModels,
  ]);

  useEffect(() => {
    if (!warmModels.length) return;
    if (!selectedModel || (selectedModel !== "auto" && !warmModels.includes(selectedModel))) {
      setSelectedModel(warmModels.length > 1 ? "auto" : warmModels[0]);
    }
  }, [selectedModel, warmModels]);

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
    if (section !== "chat" || !chatStateHydrated) return;
    if (routedChatId) {
      if (conversations.some((conversation) => conversation.id === routedChatId)) {
        setChatState((prev) =>
          prev.activeConversationId === routedChatId
            ? prev
            : { ...prev, activeConversationId: routedChatId },
        );
      } else {
        const fallbackId = conversations[0]?.id ?? null;
        replaceChatRoute(fallbackId);
        setChatState((prev) => ({
          ...prev,
          activeConversationId: fallbackId ?? "",
        }));
      }
      return;
    }
    if (activeConversation?.id) {
      replaceChatRoute(activeConversation.id);
    }
  }, [
    activeConversation?.id,
    chatStateHydrated,
    conversations,
    replaceChatRoute,
    routedChatId,
    section,
  ]);

  useEffect(() => {
    const element = chatScrollRef.current;
    if (!element) return;
    if (!activeConversationId && !lastMessageId && !isSending) return;
    const distanceFromBottom =
      element.scrollHeight - element.scrollTop - element.clientHeight;
    if (distanceFromBottom < 80) {
      element.scrollTop = element.scrollHeight;
    }
  }, [activeConversationId, isSending, lastMessageId]);

  useEffect(() => () => currentAbortRef.current?.abort(), []);

  const canChat =
    !!status && (status.llama_ready || (status.is_client && warmModels.length > 0));
  const canRegenerate =
    canChat &&
    !!activeConversation &&
    findLastUserMessageIndex(activeConversation.messages) >= 0;

  const updateChatState = useCallback((updater: (prev: ChatState) => ChatState) => {
    setChatState((prev) => updater(prev));
  }, []);

  const markComposerAttachment = useCallback(
    (attachmentId: string, patch: AttachmentStatePatch) => {
      setPendingAttachments((prev) =>
        prev.map((attachment) =>
          attachment.id === attachmentId ? { ...attachment, ...patch } : attachment,
        ),
      );
    },
    [],
  );

  const streamAssistantReply = useCallback(
    async (params: {
      conversationId: string;
      assistantId: string;
      model: string;
      historyForRequest: ChatMessage[];
      requestId?: string;
      prebuiltContentByMessageId?: Record<string, Array<Record<string, unknown>>>;
    }) => {
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
        const requestId = providedRequestId ?? createChatId();
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
        const effectiveMaxRetries = hasBlobContent(requestInput) ? 1 : MAX_RETRIES;
        let response: Response | null = null;

        for (let attempt = 0; attempt < effectiveMaxRetries; attempt++) {
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
          if (!RETRYABLE.has(response.status) || attempt === effectiveMaxRetries - 1) {
            break;
          }
          await new Promise((resolve) => setTimeout(resolve, RETRY_DELAYS[attempt]));
        }

        if (!response?.ok || !response?.body) {
          const errorMessage = response
            ? await parseApiErrorBody(response)
            : "HTTP unknown";
          throw new Error(errorMessage);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let full = "";
        let completionTokens: number | null = null;
        let firstTokenAt: number | null = null;

        batcher = createRafBatcher((snapshot) => {
          updateChatState((prev) => ({
            ...prev,
            conversations: updateConversationList(
              prev.conversations,
              conversationId,
              (conversation) => ({
                ...conversation,
                messages: conversation.messages.map((message) =>
                  message.id === assistantId ? { ...message, content: snapshot } : message,
                ),
                updatedAt: Date.now(),
              }),
            ),
          }));
        });

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          buffer = buffer.replace(/\r\n/g, "\n");
          let frameEnd = buffer.indexOf("\n\n");
          while (frameEnd >= 0) {
            const frame = buffer.slice(0, frameEnd);
            buffer = buffer.slice(frameEnd + 2);
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
              frameEnd = buffer.indexOf("\n\n");
              continue;
            }
            try {
              const payload = JSON.parse(data) as Record<string, unknown>;
              if (eventName === "response.output_text.delta") {
                const contentDelta =
                  typeof payload.delta === "string" ? payload.delta : "";
                if (!contentDelta) {
                  frameEnd = buffer.indexOf("\n\n");
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
            frameEnd = buffer.indexOf("\n\n");
          }
        }

        batcher.flush();

        const endAt = performance.now();
        const genStart = firstTokenAt ?? reqStart;
        const genSecs = Math.max(0.001, (endAt - genStart) / 1000);
        const ttftMs = Math.max(0, Math.round((firstTokenAt ?? endAt) - reqStart));
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
              messages: conversation.messages.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: message.content || "(empty response)",
                      stats,
                    }
                  : message,
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
                messages: conversation.messages.map((message) =>
                  message.id === assistantId
                    ? { ...message, content: message.content || "(stopped)" }
                    : message,
                ),
                updatedAt: Date.now(),
              }),
            ),
          }));
        } else {
          const errorMessage = err instanceof Error ? err.message : String(err);
          updateChatState((prev) => ({
            ...prev,
            conversations: updateConversationList(
              prev.conversations,
              conversationId,
              (conversation) => ({
                ...conversation,
                messages: conversation.messages.map((message) =>
                  message.id === assistantId
                    ? { ...message, content: `Error: ${errorMessage}`, error: true }
                    : message,
                ),
                updatedAt: Date.now(),
              }),
            ),
          }));
        }
      } finally {
        batcher?.cancel();
        if (currentAbortRef.current === controller) currentAbortRef.current = null;
        setIsSending(false);
      }
    },
    [updateChatState],
  );

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if ((!trimmed && pendingAttachments.length === 0) || !status) return;
      if (isSending) {
        queuedInputRef.current = trimmed;
        setQueuedText(trimmed || "📎 Attachment");
        setInput("");
        currentAbortRef.current?.abort();
        return;
      }
      if (attachmentSendIssue) {
        if (attachmentPreparationMessage) return;
        setComposerError(attachmentSendIssue);
        return;
      }

      let model = selectedModel || status.model_name;
      if (pendingAttachments.length > 0 && (!model || model === "auto")) {
        const compatibleModel = warmModels.find(
          (candidate) =>
            (!pendingKinds.has("audio") || audioModels.has(candidate)) &&
            (!pendingKinds.has("file") || multimodalModels.has(candidate)),
        );
        if (compatibleModel) {
          model = compatibleModel;
        } else if (pendingKinds.has("audio")) {
          const audioModel = warmModels.find((candidate) => audioModels.has(candidate));
          if (audioModel) model = audioModel;
        } else if (pendingKinds.has("file")) {
          const fileModel = warmModels.find((candidate) => multimodalModels.has(candidate));
          if (fileModel) model = fileModel;
        }
      }

      setComposerError(null);
      const conversationId = activeConversation?.id ?? createChatId();
      const normalizedPendingAttachments = pendingAttachments.map((attachment) => ({
        ...attachment,
      }));
      const userMessageId = createChatId();
      const requestId = createChatId();
      const clientId = chatClientIdRef.current;
      let prebuiltContentByMessageId:
        | Record<string, Array<Record<string, unknown>>>
        | undefined;

      if (normalizedPendingAttachments.length > 0) {
        try {
          const blocks = await buildAttachmentBlocks(
            normalizedPendingAttachments,
            requestId,
            clientId,
            (attachmentId, patch) => markComposerAttachment(attachmentId, patch),
          );
          prebuiltContentByMessageId = { [userMessageId]: blocks };
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          setComposerError(message);
          return;
        }
      }

      const userMessage: ChatMessage = {
        id: userMessageId,
        role: "user",
        content: trimmed,
        model,
        attachments:
          normalizedPendingAttachments.length > 0
            ? normalizedPendingAttachments.map(attachmentForMessage)
            : undefined,
      };
      const assistantId = createChatId();
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        model,
      };
      const existingMessages = activeConversation?.messages ?? [];
      const historyForRequest = [...existingMessages, userMessage];
      const nextTitle =
        ((activeConversation?.title === DEFAULT_CHAT_TITLE &&
          existingMessages.length === 0) ||
          !activeConversation)
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
      pushChatRoute(conversationId);
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
    },
    [
      activeConversation,
      attachmentPreparationMessage,
      attachmentSendIssue,
      audioModels,
      isSending,
      markComposerAttachment,
      multimodalModels,
      pendingAttachments,
      pendingKinds,
      pushChatRoute,
      selectedModel,
      status,
      streamAssistantReply,
      updateChatState,
      warmModels,
    ],
  );

  useEffect(() => {
    if (isSending) return;
    const queued = queuedInputRef.current;
    if (queued == null) return;
    queuedInputRef.current = null;
    setQueuedText(null);
    void sendMessage(queued);
  }, [isSending, sendMessage]);

  const regenerateLastResponse = useCallback(async () => {
    if (!status || isSending || !activeConversation) return;
    const model = selectedModel || status.model_name;
    const conversationId = activeConversation.id;
    const lastUserIndex = findLastUserMessageIndex(activeConversation.messages);
    if (lastUserIndex < 0) return;

    const historyForRequest = activeConversation.messages.slice(0, lastUserIndex + 1);
    const assistantId = createChatId();
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
  }, [activeConversation, isSending, selectedModel, status, streamAssistantReply, updateChatState]);

  const stopStreaming = useCallback(() => {
    currentAbortRef.current?.abort();
  }, []);

  const createNewConversation = useCallback(() => {
    queuedInputRef.current = null;
    setQueuedText(null);
    const conversation = createConversation();
    updateChatState((prev) => ({
      conversations: [conversation, ...prev.conversations],
      activeConversationId: conversation.id,
    }));
    pushChatRoute(conversation.id);
    setReasoningOpen({});
    setInput("");
    setPendingAttachments([]);
  }, [pushChatRoute, updateChatState]);

  const selectConversation = useCallback(
    (conversationId: string) => {
      updateChatState((prev) => {
        if (prev.activeConversationId === conversationId) return prev;
        if (!prev.conversations.some((conversation) => conversation.id === conversationId)) {
          return prev;
        }
        return { ...prev, activeConversationId: conversationId };
      });
      pushChatRoute(conversationId);
      setReasoningOpen({});
      setInput("");
      setPendingAttachments([]);
    },
    [pushChatRoute, updateChatState],
  );

  const renameConversation = useCallback(
    (conversationId: string, nextTitle: string) => {
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
    },
    [updateChatState],
  );

  const deleteConversation = useCallback(
    (conversationId: string) => {
      const remaining = conversations.filter(
        (conversation) => conversation.id !== conversationId,
      );
      const nextActiveId =
        activeConversationId === conversationId
          ? (remaining[0]?.id ?? null)
          : activeConversationId || null;
      updateChatState((prev) => ({
        conversations: prev.conversations.filter(
          (conversation) => conversation.id !== conversationId,
        ),
        activeConversationId:
          prev.activeConversationId === conversationId
            ? (nextActiveId ?? "")
            : prev.activeConversationId,
      }));
      replaceChatRoute(nextActiveId);
      setReasoningOpen({});
      setInput("");
      setPendingAttachments([]);
    },
    [activeConversationId, conversations, replaceChatRoute, updateChatState],
  );

  const clearAllConversations = useCallback(() => {
    updateChatState(() => ({ conversations: [], activeConversationId: "" }));
    replaceChatRoute(null);
    setReasoningOpen({});
    setInput("");
    setPendingAttachments([]);
  }, [replaceChatRoute, updateChatState]);

  const handleSubmit = useCallback(() => {
    if (!canChat || attachmentPreparationMessage) return;
    void sendMessage(input);
  }, [attachmentPreparationMessage, canChat, input, sendMessage]);

  return {
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
  };
}
