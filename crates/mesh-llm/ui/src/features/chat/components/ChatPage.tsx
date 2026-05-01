import {
  type ChangeEvent,
  type Dispatch,
  type MutableRefObject,
  type Ref,
  type SetStateAction,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  ChevronDown,
  Cpu,
  Hash,
  Loader2,
  MessageSquarePlus,
  Network,
  Square,
  User,
} from "lucide-react";

import { BrandIcon } from "../../../components/brand-icon";
import { Button } from "../../../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../../../components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../../components/ui/select";
import { Separator } from "../../../components/ui/separator";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "../../../components/ui/sheet";
import { cn } from "../../../lib/utils";
import {
  dataUrlToArrayBuffer,
  extractPdfText,
  isPdfMimeType,
  renderPdfPagesToImages,
} from "../../../lib/pdf";
import { validateAttachmentFile } from "../../../lib/attachments";
import {
  modelDisplayName,
  shortName,
} from "../../app-shell/lib/status-helpers";
import type {
  MeshModel,
  ModelServingStat,
  StatusPayload,
} from "../../app-shell/lib/status-types";
import { parseDataUrl } from "../lib/chat-attachments";
import { createChatId } from "../lib/chat-id";
import type {
  AttachmentStatePatch,
  ChatAttachment,
  ChatConversation,
  ChatMessage,
} from "../lib/chat-types";
import { describeImageAttachmentForPrompt, describeRenderedPagesAsText } from "../lib/vision-describe";
import { ChatComposer } from "./composer/ChatComposer";
import { ChatBubble } from "./messages/ChatBubble";
import { ConversationList } from "./sidebar/ConversationList";

const DOCS_URL = "https://docs.anarchai.org";

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
  if (model.multimodal) {
    return { icon: "🎛️", title: "Multimodal — supports media inputs" };
  }
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
      {inviteWithModelCommand ? (
        <div className="space-y-2 rounded-md border border-dashed p-3 text-left">
          <div className="text-xs text-muted-foreground">
            Invite another machine preconfigured for this model.
          </div>
          <div className="rounded-md border bg-muted/40 px-2 py-1.5">
            <code className="block overflow-x-auto whitespace-nowrap text-xs">
              {inviteWithModelCommand}
            </code>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => void copyInviteWithModelCommand()}
          >
            {inviteWithModelCopied ? "Copied" : "Copy invite command"}
          </Button>
        </div>
      ) : null}
    </div>
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
  selectedModelAudio: boolean;
  selectedModelMultimodal: boolean;
  composerError: string | null;
  setComposerError: Dispatch<SetStateAction<string | null>>;
  attachmentSendIssue: string | null;
  attachmentPreparationMessage: string | null;
  pendingAttachments: ChatAttachment[];
  setPendingAttachments: Dispatch<SetStateAction<ChatAttachment[]>>;
  conversations: ChatConversation[];
  activeConversationId: string;
  onConversationCreate: () => void;
  onConversationSelect: (conversationId: string) => void;
  onConversationRename: (conversationId: string, title: string) => void;
  onConversationDelete: (conversationId: string) => void;
  onConversationsClear: () => void;
  messages: ChatMessage[];
  reasoningOpen: Record<string, boolean>;
  setReasoningOpen: Dispatch<SetStateAction<Record<string, boolean>>>;
  chatScrollRef: MutableRefObject<HTMLDivElement | null>;
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
    inviteToken,
    warmModels,
    meshModelByName,
    modelStatsByName,
    selectedModel,
    setSelectedModel,
    selectedModelNodeCount,
    selectedModelVramGb,
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
  const [editingConversationId, setEditingConversationId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const chatInputRef = useRef<HTMLTextAreaElement | null>(null);
  const editingTitleInputRef = useRef<HTMLInputElement | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const audioInputRef = useRef<HTMLInputElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pdfInputRef = useRef<HTMLInputElement | null>(null);

  function markPendingAttachment(
    attachmentId: string,
    patch: AttachmentStatePatch,
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
        id: createChatId(),
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
    const attachment = pendingAttachments.find((item) => item.id === attachmentId);
    if (!attachment) return;
    if (attachment.kind === "image" && !attachment.imageDescription) {
      markPendingAttachment(attachmentId, {
        status: "uploading",
        error: undefined,
        extractionSummary: "Describing image...",
      });
      void describeImageAttachment(attachmentId, attachment.dataUrl);
      setComposerError(null);
      return;
    }
    markPendingAttachment(attachmentId, {
      status: "pending",
      error: undefined,
      extractionSummary: undefined,
    });
    setComposerError(null);
  }

  function addImageAttachment(attachment: Omit<ChatAttachment, "id" | "status" | "error">) {
    const attachmentId = createChatId();
    setPendingAttachments((prev) => [
      ...prev,
      {
        id: attachmentId,
        status: "uploading",
        extractionSummary: "Describing image...",
        ...attachment,
      },
    ]);
    void describeImageAttachment(attachmentId, attachment.dataUrl);
  }

  async function describeImageAttachment(attachmentId: string, dataUrl: string) {
    const result = await describeImageAttachmentForPrompt(dataUrl, {
      onProgress: (message) => {
        setPendingAttachments((prev) =>
          prev.map((a) =>
            a.id === attachmentId
              ? { ...a, extractionSummary: message }
              : a,
          ),
        );
      },
    });
    setPendingAttachments((prev) =>
      prev.map((a) =>
        a.id === attachmentId
          ? {
              ...a,
              status: result.error ? "failed" : "pending",
              imageDescription: result.imageDescription,
              extractionSummary: result.extractionSummary,
              error: result.error,
            }
          : a,
      ),
    );
    if (result.error) {
      console.warn(result.error);
    }
  }

  function handleImageSelect(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    if (!files.length) return;
    const rejected: string[] = [];
    for (const file of files) {
      const validationError = validateAttachmentFile(file, "image");
      if (validationError) {
        rejected.push(`${file.name}: ${validationError}`);
        continue;
      }
      processImageFile(file);
    }
    if (rejected.length) {
      setComposerError(rejected.join(" "));
    }
  }

  function processImageFile(file: File) {
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
          addImageAttachment({
            kind: "image",
            dataUrl: src,
            mimeType: parseDataUrl(src)?.mimeType || file.type || "image/jpeg",
            fileName: file.name,
          });
          return;
        }
        ctx.drawImage(img, 0, 0, width, height);
        addImageAttachment({
          kind: "image",
          dataUrl: canvas.toDataURL("image/jpeg", 0.85),
          mimeType: "image/jpeg",
          fileName: file.name,
        });
      };
      img.onerror = () => {
        addImageAttachment({
          kind: "image",
          dataUrl: src,
          mimeType: parseDataUrl(src)?.mimeType || file.type || "image/jpeg",
          fileName: file.name,
        });
      };
      img.src = src;
    };
    reader.readAsDataURL(file);
  }

  function handleAudioSelect(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    if (!files.length) return;
    const rejected: string[] = [];
    for (const file of files) {
      const validationError = validateAttachmentFile(file, "audio");
      if (validationError) {
        rejected.push(`${file.name}: ${validationError}`);
        continue;
      }
      processAudioFile(file);
    }
    if (rejected.length) {
      setComposerError(rejected.join(" "));
    }
  }

  function processAudioFile(file: File) {
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
  }

  function handleFileSelect(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    if (!files.length) return;
    const rejected: string[] = [];
    for (const file of files) {
      const validationError = validateAttachmentFile(file, "file");
      if (validationError) {
        rejected.push(`${file.name}: ${validationError}`);
        continue;
      }
      processGenericFile(file);
    }
    if (rejected.length) {
      setComposerError(rejected.join(" "));
    }
  }

  function processGenericFile(file: File) {
    const mimeType = file.type || "application/octet-stream";
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      const detectedMime = parseDataUrl(dataUrl)?.mimeType ?? mimeType;
      const isPdf = isPdfMimeType(detectedMime) || file.name.toLowerCase().endsWith(".pdf");
      if (isPdf) {
        void handlePdfAttachment(dataUrl, file.name);
      } else {
        addPendingAttachment({
          kind: "file",
          dataUrl,
          mimeType: parseDataUrl(dataUrl)?.mimeType || mimeType,
          fileName: file.name,
        });
      }
    };
    reader.readAsDataURL(file);
  }

  async function handlePdfAttachment(dataUrl: string, fileName: string) {
    const attachmentId = createChatId();
    setPendingAttachments((prev) => [
      ...prev,
      {
        id: attachmentId,
        kind: "file",
        dataUrl,
        mimeType: "application/pdf",
        fileName,
        status: "uploading",
        extractionSummary: "Extracting text...",
      },
    ]);

    try {
      const buffer = dataUrlToArrayBuffer(dataUrl);
      const result = await extractPdfText(buffer);

      if (result.pagesWithText > 0 && result.wordCount > 20) {
        const summary = `${result.pageCount} page${result.pageCount !== 1 ? "s" : ""}, ~${result.wordCount.toLocaleString()} words extracted`;
        setPendingAttachments((prev) =>
          prev.map((a) =>
            a.id === attachmentId
              ? {
                  ...a,
                  status: "pending",
                  extractedText: result.text,
                  renderedPageImages: undefined,
                  extractionSummary: summary,
                }
              : a,
          ),
        );
      } else {
        const images = await renderPdfPagesToImages(buffer, {
          maxPages: 8,
        });
        if (images.length > 0) {
          setPendingAttachments((prev) =>
            prev.map((a) =>
              a.id === attachmentId
                ? {
                    ...a,
                    renderedPageImages: images,
                    extractionSummary: `Describing ${images.length} page${images.length !== 1 ? "s" : ""}...`,
                  }
                : a,
            ),
          );
          const combinedText = await describeRenderedPagesAsText(images, {
            onProgress: (message) => {
              setPendingAttachments((prev) =>
                prev.map((a) =>
                  a.id === attachmentId
                    ? { ...a, extractionSummary: message }
                    : a,
                ),
              );
            },
          });
          const summary = `${result.pageCount} page${result.pageCount !== 1 ? "s" : ""}, ${images.length} page${images.length !== 1 ? "s" : ""} described (scanned PDF)`;
          setPendingAttachments((prev) =>
            prev.map((a) =>
              a.id === attachmentId
                ? {
                    ...a,
                    status: "pending",
                    extractedText: combinedText,
                    renderedPageImages: undefined,
                    extractionSummary: summary,
                  }
                : a,
            ),
          );
        } else {
          throw new Error("Could not render PDF pages");
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setPendingAttachments((prev) =>
        prev.map((a) =>
          a.id === attachmentId
            ? { ...a, status: "failed", error: `PDF extraction failed: ${message}` }
            : a,
        ),
      );
    }
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

  const conversationListContent = (
    <ConversationList
      conversations={conversations}
      activeConversationId={activeConversationId}
      editingConversationId={editingConversationId}
      editingTitle={editingTitle}
      editingTitleInputRef={editingTitleInputRef}
      setEditingTitle={setEditingTitle}
      hasChats={hasChats}
      isSending={isSending}
      onConversationCreate={onConversationCreate}
      onConversationSelect={onConversationSelect}
      onConversationStartRename={startInlineRename}
      onConversationSaveRename={saveInlineRename}
      onConversationCancelRename={cancelInlineRename}
      onConversationDelete={handleDelete}
      onConversationsClear={handleClearAll}
      onConversationAction={() => setMobileSidebarOpen(false)}
    />
  );

  return (
    <Card className="flex h-full min-h-0 flex-1 flex-col overflow-hidden">
      <CardHeader className="px-3 py-2 md:px-6 md:py-4">
        <div className="flex items-center gap-2 md:gap-3">
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
                  const displayName = modelDisplayName(selectedMeshModel) || model;
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
                          {multimodalInfo ? (
                            <span className="ml-1.5" title={multimodalInfo.title}>
                              {multimodalInfo.icon}
                            </span>
                          ) : null}
                          {visionInfo ? (
                            <span className="ml-1.5" title={visionInfo.title}>
                              {visionInfo.icon}
                            </span>
                          ) : null}
                          {audioInfo ? (
                            <span className="ml-1.5" title={audioInfo.title}>
                              {audioInfo.icon}
                            </span>
                          ) : null}
                          {reasoningInfo ? (
                            <span className="ml-1.5" title={reasoningInfo.title}>
                              {reasoningInfo.icon}
                            </span>
                          ) : null}
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
              <span className="font-medium">⏳ Busy</span> — {props.inflightRequests} requests in flight, responses may be
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
              <span className="font-medium">Community demo</span> — best-effort public instance. For direct, faster access run{" "}
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
          {hasChats ? (
            <aside className="hidden md:block shrink-0 md:w-72 md:border-r">
              {conversationListContent}
            </aside>
          ) : null}

          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            <div
              ref={chatScrollRef as Ref<HTMLDivElement>}
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
                    <button
                      type="button"
                      onClick={onStop}
                      className="group flex items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                      title="Click to stop (Esc)"
                    >
                      <Loader2 className="h-3.5 w-3.5 animate-spin group-hover:hidden" />
                      <Square className="hidden h-3.5 w-3.5 group-hover:block" />
                      <span>
                        <span className="group-hover:hidden">Streaming response...</span>
                        <span className="hidden group-hover:inline">Stop generating</span>
                      </span>
                    </button>
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
            <ChatComposer
              pendingAttachments={pendingAttachments}
              composerError={composerError}
              setComposerError={setComposerError}
              attachmentSendIssue={attachmentSendIssue}
              attachmentPreparationMessage={attachmentPreparationMessage}
              input={input}
              setInput={setInput}
              canChat={canChat}
              isSending={isSending}
              canRegenerate={canRegenerate}
              selectedModelAudio={selectedModelAudio}
              selectedModelMultimodal={selectedModelMultimodal}
              chatInputRef={chatInputRef}
              pdfInputRef={pdfInputRef}
              imageInputRef={imageInputRef}
              audioInputRef={audioInputRef}
              fileInputRef={fileInputRef}
              onFileSelect={handleFileSelect}
              onImageSelect={handleImageSelect}
              onAudioSelect={handleAudioSelect}
              onStop={onStop}
              onRegenerate={onRegenerate}
              onSubmit={onSubmit}
              onRemovePendingAttachment={removePendingAttachment}
              onRetryPendingAttachment={resetAttachmentStatus}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
