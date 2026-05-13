import * as DialogPrimitive from '@radix-ui/react-dialog'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Check,
  Cpu,
  Download,
  FileIcon,
  FileImage,
  FileText,
  HardDrive,
  Loader2,
  MessageSquareMore,
  Music,
  Play,
  ScanText,
  X
} from 'lucide-react'
import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { DestructiveActionDialog } from '@/components/ui/DestructiveActionDialog'
import { EmptyState } from '@/components/ui/EmptyState'
import { TextInputDialog } from '@/components/ui/TextInputDialog'
import { ChatLiveLoadingGhost } from '@/features/chat/components/ChatLiveLoadingGhost'
import { ChatSidebar } from '@/features/chat/components/ChatSidebar'
import { Composer } from '@/features/chat/components/Composer'
import { MessageRow, type MessageAttachmentAction } from '@/features/chat/components/MessageRow'
import { ModelSelect } from '@/features/chat/components/ModelSelect'
import { TransparencyPane } from '@/features/chat/components/transparency/TransparencyPane'
import { ChatLayout } from '@/features/chat/layouts/ChatLayout'
import { uiMessagesToThreadMessages } from '@/features/chat/api/use-chat-messages'
import { ChatSessionProvider } from '@/features/chat/api/chat-session'
import { createChatDraftConversationId } from '@/features/chat/api/chat-session-ids'
import { useOptionalChatSession, useChatSession } from '@/features/chat/api/chat-session-hooks'
import { buildComposerMessageContent } from '@/features/chat/api/legacy-attachments'
import type { AttachmentProcessingStage } from '@/features/chat/api/legacy-attachments'
import {
  describeImageForPrompt,
  describeScannedPdf,
  extractPdfTextFromFile,
  isBrowserVisionModelLoaded
} from '@/features/chat/api/attachment-preprocessing'
import { useModelsQuery } from '@/features/network/api/use-models-query'
import { useStatusQuery } from '@/features/network/api/use-status-query'
import { adaptModelsToSummary } from '@/features/network/api/models-adapter'
import { useDataMode } from '@/lib/data-mode'
import { useBooleanFeatureFlag } from '@/lib/feature-flags'
import { CHAT_HARNESS } from '@/features/app-tabs/data'
import type {
  ChatActionMetric,
  ChatHarnessData,
  Conversation,
  ModelSelectOption,
  ModelSummary,
  TransparencyMessage
} from '@/features/app-tabs/types'

type ChatPageProps = { data?: ChatHarnessData }
type ComposerSubmission = { prompt: string; attachments: File[] }
type ConversationComposerDraft = ComposerSubmission
type QueuedSubmission = ComposerSubmission & { id: string; timestamp: string; conversationId: string }
type AttachmentProcessingStatus = {
  conversationId: string
  stage: AttachmentProcessingStage
  attachmentCount: number
  prompt: string
  usesBrowserAnalyzer: boolean
  browserAnalyzerReady: boolean
}
type SubmittedAttachmentKind = MessageAttachmentAction['kind']
type SubmittedAttachmentPreview = {
  id: string
  conversationId: string
  messageId: string
  label: string
  kind: SubmittedAttachmentKind
  fileName: string
  mimeType: string
  objectUrl: string
}
type FailedSubmission = ComposerSubmission & {
  id: string
  timestamp: string
  conversationId: string
  errorMessage: string
  model: string
  includeUserRow: boolean
}
type DeleteConversationOptions = { returnFocusElement?: HTMLElement | null }

const AUTO_MODEL_VALUE = 'auto'
const AUTO_MODEL_OPTION: ModelSelectOption = {
  value: AUTO_MODEL_VALUE,
  label: 'Auto',
  meta: 'Router picks best model for each request',
  status: { label: 'Auto', tone: 'accent' }
}

function hasLastUserTurn(messages: Array<{ role: string }>): boolean {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') return true
  }

  return false
}

function getMessageTextContent(message: { parts?: Array<{ type: string; content?: string }> }): string {
  const textPart = message.parts?.find((part) => part.type === 'text' && typeof part.content === 'string')
  return typeof textPart?.content === 'string' ? textPart.content.trim() : ''
}

function createQueuedSubmissionId(): string {
  return `queued-${createChatDraftConversationId()}`
}

function getQueuedSubmissionBody(submission: QueuedSubmission): string {
  const trimmedPrompt = submission.prompt.trim()
  if (trimmedPrompt) return trimmedPrompt

  return `${submission.attachments.length} attachment${submission.attachments.length === 1 ? '' : 's'} queued`
}

function getSubmissionBody(submission: ComposerSubmission): string {
  const trimmedPrompt = submission.prompt.trim()
  if (trimmedPrompt) return trimmedPrompt

  return `${submission.attachments.length} attachment${submission.attachments.length === 1 ? '' : 's'}`
}

function createStoppedAssistantThreadMessage(model: string) {
  return {
    id: `stopped-${createChatDraftConversationId()}`,
    messageRole: 'assistant' as const,
    timestamp: new Date().toISOString(),
    body: '',
    model
  }
}

const ATTACHMENT_PROCESSING_STEPS: Array<{
  stage: AttachmentProcessingStage
  title: string
  description: string
}> = [
  {
    stage: 'downloading',
    title: 'Downloading',
    description: 'Fetching the browser analyzer and attachment assets.'
  },
  {
    stage: 'starting',
    title: 'Starting',
    description: 'Warming the local vision and document pipeline.'
  },
  {
    stage: 'processing',
    title: 'Processing',
    description: 'Reading the attachment before the prompt is sent.'
  }
]

const ATTACHMENT_PROCESSING_ORDER: Record<AttachmentProcessingStage, number> = {
  downloading: 0,
  starting: 1,
  processing: 2
}

function usesBrowserAnalyzerForAttachment(file: File): boolean {
  const mimeType = file.type.toLowerCase()
  const fileName = file.name.toLowerCase()
  return mimeType.startsWith('image/') || mimeType === 'application/pdf' || fileName.endsWith('.pdf')
}

function getAttachmentProcessingStepCopy(
  step: (typeof ATTACHMENT_PROCESSING_STEPS)[number],
  status: AttachmentProcessingStatus
) {
  if (!status.usesBrowserAnalyzer) {
    if (step.stage === 'downloading') {
      return { title: 'Reading', description: 'Loading the attachment bytes in this browser.' }
    }
    if (step.stage === 'starting') {
      return { title: 'Preparing', description: 'Packaging the attachment for the request.' }
    }
  }

  if (status.browserAnalyzerReady) {
    if (step.stage === 'downloading') {
      return { title: 'Cached', description: 'Reusing the browser analyzer already loaded in this tab.' }
    }
    if (step.stage === 'starting') {
      return { title: 'Ready', description: 'The local vision and document pipeline is already warm.' }
    }
  }

  return { title: step.title, description: step.description }
}

function getSubmittedAttachmentKind(file: File): SubmittedAttachmentKind {
  const mimeType = file.type.toLowerCase()
  const fileName = file.name.toLowerCase()
  if (mimeType.startsWith('image/')) return 'image'
  if (mimeType === 'application/pdf' || fileName.endsWith('.pdf')) return 'pdf'
  if (mimeType.startsWith('audio/')) return 'audio'
  return 'file'
}

function getSubmittedAttachmentLabel(kind: SubmittedAttachmentKind, ordinal: number): string {
  if (kind === 'image') return `Image ${ordinal}`
  if (kind === 'pdf') return `PDF ${ordinal}`
  if (kind === 'audio') return `Audio ${ordinal}`
  return `File ${ordinal}`
}

function createObjectUrl(file: File): string {
  if (typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function') return ''
  return URL.createObjectURL(file)
}

function revokeObjectUrl(objectUrl: string) {
  if (!objectUrl || typeof URL === 'undefined' || typeof URL.revokeObjectURL !== 'function') return
  URL.revokeObjectURL(objectUrl)
}

function getAttachmentProcessingHeadline(status: AttachmentProcessingStatus): string {
  if (status.stage === 'starting') return 'Starting local analyzer'
  if (status.stage === 'processing') return 'Processing attachment content'
  return 'Downloading browser model'
}

function AttachmentProcessingPanel({ status }: { status: AttachmentProcessingStatus }) {
  const activeIndex = ATTACHMENT_PROCESSING_ORDER[status.stage]
  const prompt = status.prompt.trim()

  return (
    <section
      aria-live="polite"
      aria-label="Attachment preparation status"
      className="mx-auto my-10 flex w-full max-w-[34rem] flex-col items-center text-center"
    >
      <div className="relative w-full overflow-hidden rounded-[calc(var(--radius-lg)+6px)] border border-[color:color-mix(in_oklab,var(--color-accent)_35%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-panel)_88%,var(--color-accent)_12%)] p-5 shadow-[0_22px_70px_color-mix(in_oklab,var(--color-accent)_10%,transparent)]">
        <div className="absolute left-1/2 top-0 h-px w-2/3 -translate-x-1/2 bg-[color:color-mix(in_oklab,var(--color-accent)_48%,transparent)]" />
        <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-full border border-[color:color-mix(in_oklab,var(--color-accent)_40%,var(--color-border))] bg-panel-strong text-accent">
          <Loader2 className="size-5 animate-spin" aria-hidden={true} strokeWidth={1.7} />
        </div>
        <div className="space-y-1">
          <p className="text-[length:var(--density-type-label)] font-semibold uppercase tracking-[0.18em] text-accent">
            Preparing attachments
          </p>
          <h2 className="text-[length:var(--density-type-title)] font-semibold text-foreground">
            {getAttachmentProcessingHeadline(status)}
          </h2>
          <p className="mx-auto max-w-[26rem] text-[length:var(--density-type-body)] leading-6 text-fg-muted">
            {status.attachmentCount} file{status.attachmentCount === 1 ? '' : 's'} will be converted locally, then your
            prompt will be sent to the model.
          </p>
        </div>
        {prompt ? (
          <div className="mx-auto mt-4 max-w-[28rem] rounded-[var(--radius)] border border-border-soft bg-panel px-3 py-2 text-left text-[length:var(--density-type-caption)] text-fg-muted">
            <span className="font-medium text-fg">Prompt waiting:</span> {prompt}
          </div>
        ) : null}
        <ol className="mt-5 grid gap-2 text-left sm:grid-cols-3">
          {ATTACHMENT_PROCESSING_STEPS.map((step, index) => {
            const complete = index < activeIndex
            const active = index === activeIndex
            const Icon = step.stage === 'downloading' ? Download : step.stage === 'starting' ? Play : ScanText
            const copy = getAttachmentProcessingStepCopy(step, status)
            return (
              <li
                key={step.stage}
                className="rounded-[var(--radius)] border border-border-soft bg-panel px-3 py-3"
                aria-current={active ? 'step' : undefined}
              >
                <div className="mb-2 flex items-center gap-2">
                  <span
                    className="inline-flex size-6 items-center justify-center rounded-full border text-[length:var(--density-type-label)]"
                    style={{
                      borderColor: active || complete ? 'var(--color-accent)' : 'var(--color-border)',
                      background: complete
                        ? 'var(--color-accent)'
                        : active
                          ? 'color-mix(in oklab, var(--color-accent) 18%, transparent)'
                          : 'transparent',
                      color: complete ? 'var(--color-panel)' : active ? 'var(--color-accent)' : 'var(--color-fg-faint)'
                    }}
                  >
                    {complete ? (
                      <Check className="size-3.5" aria-hidden={true} />
                    ) : active ? (
                      <Loader2 className="size-3.5 animate-spin" aria-hidden={true} />
                    ) : (
                      <Icon className="size-3.5" aria-hidden={true} />
                    )}
                  </span>
                  <span className={active ? 'font-semibold text-foreground' : 'font-medium text-fg-muted'}>
                    {copy.title}
                  </span>
                </div>
                <p className="text-[length:var(--density-type-caption)] leading-5 text-fg-faint">{copy.description}</p>
              </li>
            )
          })}
        </ol>
      </div>
    </section>
  )
}

function AttachmentPreviewIcon({ kind }: { kind: SubmittedAttachmentKind }) {
  if (kind === 'image') return <FileImage className="size-4" aria-hidden={true} />
  if (kind === 'pdf') return <FileText className="size-4" aria-hidden={true} />
  if (kind === 'audio') return <Music className="size-4" aria-hidden={true} />
  return <FileIcon className="size-4" aria-hidden={true} />
}

function AttachmentPreviewBody({ attachment }: { attachment: SubmittedAttachmentPreview }) {
  if (!attachment.objectUrl) {
    return (
      <div className="grid min-h-[18rem] place-items-center rounded-[var(--radius)] border border-border-soft bg-panel-strong px-6 text-center">
        <div className="max-w-[26rem] space-y-2">
          <AttachmentPreviewIcon kind={attachment.kind} />
          <p className="text-[length:var(--density-type-body)] font-medium text-fg">Preview unavailable</p>
          <p className="text-[length:var(--density-type-control)] leading-6 text-fg-muted">
            The file was submitted with this prompt, but this browser cannot create a local preview URL for it.
          </p>
        </div>
      </div>
    )
  }

  if (attachment.kind === 'image') {
    return (
      <div className="grid max-h-[min(72vh,760px)] min-h-[18rem] place-items-center overflow-auto rounded-[var(--radius)] border border-border-soft bg-panel-strong p-3">
        <img
          src={attachment.objectUrl}
          alt={attachment.fileName}
          className="max-h-[68vh] max-w-full rounded-[calc(var(--radius)-2px)] object-contain"
        />
      </div>
    )
  }

  if (attachment.kind === 'pdf') {
    return (
      <iframe
        title={`Preview ${attachment.fileName}`}
        src={attachment.objectUrl}
        className="h-[min(72vh,760px)] w-full rounded-[var(--radius)] border border-border-soft bg-panel-strong"
      />
    )
  }

  if (attachment.kind === 'audio') {
    return (
      <div className="grid min-h-[16rem] place-items-center rounded-[var(--radius)] border border-border-soft bg-panel-strong px-6">
        <audio controls src={attachment.objectUrl} className="w-full max-w-[32rem]">
          <track kind="captions" />
        </audio>
      </div>
    )
  }

  return (
    <div className="grid min-h-[18rem] place-items-center rounded-[var(--radius)] border border-border-soft bg-panel-strong px-6 text-center">
      <div className="max-w-[26rem] space-y-2">
        <AttachmentPreviewIcon kind={attachment.kind} />
        <p className="text-[length:var(--density-type-body)] font-medium text-fg">{attachment.fileName}</p>
        <p className="text-[length:var(--density-type-control)] leading-6 text-fg-muted">
          This attachment was sent with the prompt. Inline preview is available for images, PDFs, and audio files.
        </p>
      </div>
    </div>
  )
}

function AttachmentPreviewDialog({
  attachment,
  onOpenChange
}: {
  attachment: SubmittedAttachmentPreview | null
  onOpenChange: (open: boolean) => void
}) {
  const open = attachment !== null

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="surface-scrim fixed inset-0 z-50 data-[state=closed]:animate-out data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" />
        <DialogPrimitive.Content className="shadow-surface-modal fixed left-1/2 top-1/2 z-50 flex max-h-[calc(100vh-2rem)] w-[min(920px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel text-foreground outline-none data-[state=closed]:animate-out data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95">
          {attachment ? (
            <>
              <div className="flex items-start justify-between gap-4 border-b border-border-soft px-5 py-4">
                <div className="min-w-0">
                  <DialogPrimitive.Title className="flex min-w-0 items-center gap-2 text-[length:var(--density-type-headline)] font-semibold leading-5 tracking-[-0.02em] text-fg">
                    <span className="grid size-8 shrink-0 place-items-center rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-accent)_34%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_12%,var(--color-panel))] text-accent">
                      <AttachmentPreviewIcon kind={attachment.kind} />
                    </span>
                    <span className="truncate">{attachment.fileName}</span>
                  </DialogPrimitive.Title>
                  <DialogPrimitive.Description className="mt-1.5 text-[length:var(--density-type-caption)] text-fg-faint">
                    {attachment.label} {attachment.mimeType ? `· ${attachment.mimeType}` : ''}
                  </DialogPrimitive.Description>
                </div>
                <DialogPrimitive.Close asChild>
                  <button
                    type="button"
                    className="ui-control inline-flex size-8 shrink-0 items-center justify-center rounded-[var(--radius)] border text-fg-muted outline-none transition-[background,color,box-shadow,transform] hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                    aria-label="Close attachment preview"
                  >
                    <X className="size-4" aria-hidden={true} />
                  </button>
                </DialogPrimitive.Close>
              </div>
              <div className="min-h-0 overflow-auto p-4">
                <AttachmentPreviewBody attachment={attachment} />
              </div>
            </>
          ) : null}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}

function ChatMetricBadge({ metric }: { metric: ChatActionMetric }) {
  const Icon = metric.icon === 'cpu' ? Cpu : HardDrive

  return (
    <span className="hidden shrink-0 items-center gap-1 whitespace-nowrap rounded-full border border-border px-2 py-px text-[length:var(--density-type-label)] font-medium text-fg-faint md:inline-flex">
      <Icon className="size-[10px]" /> {metric.label}
    </span>
  )
}

function modelStatusBadge(model: ModelSummary): ModelSelectOption['status'] {
  if (model.status === 'offline') return { label: 'Offline', tone: 'bad' }
  if (model.status === 'warming') return { label: 'Warming', tone: 'warn' }
  if (model.status === 'ready') return { label: 'Ready', tone: 'good' }
  return { label: 'Warm', tone: 'good' }
}

function isChatSelectableModel(model: ModelSummary): boolean {
  return model.status === 'ready' || model.status === 'warm'
}

export function ChatPageContent({ data = CHAT_HARNESS }: ChatPageProps) {
  const { mode, setMode } = useDataMode()
  const liveMode = mode === 'live'
  const modelsQuery = useModelsQuery({ enabled: mode === 'live' })
  const statusQuery = useStatusQuery({ enabled: liveMode })
  const liveStatus = statusQuery.data
  const liveModels = modelsQuery.data ? adaptModelsToSummary(modelsQuery.data.mesh_models) : undefined
  const resolvedModels = liveMode ? liveModels : data.models
  const displayModels = resolvedModels ?? data.models
  const selectableModels = useMemo(() => displayModels.filter(isChatSelectableModel), [displayModels])
  const showLiveError = liveMode && !liveModels && !modelsQuery.isFetching && modelsQuery.isError
  const showLiveLoading = liveMode && !liveModels && !showLiveError
  const warmModelCount = liveModels?.filter(isChatSelectableModel).length ?? 0
  const canChat =
    !liveMode ||
    (liveStatus?.llama_ready ?? false) ||
    warmModelCount > 0 ||
    (liveStatus?.serving_models?.length ?? 0) > 0
  const transparencyTabEnabled = useBooleanFeatureFlag('chat/transparencyTab')
  const systemPromptButtonEnabled = useBooleanFeatureFlag('chat/systemPromptButton')
  const [sidebarTab, setSidebarTab] = useState<'conversations' | 'transparency'>('conversations')
  const [inspectedMessage, setInspectedMessage] = useState<TransparencyMessage | undefined>()
  const [systemPromptDialogOpen, setSystemPromptDialogOpen] = useState(false)
  const [systemPromptDraft, setSystemPromptDraft] = useState('')
  const [composerDrafts, setComposerDrafts] = useState<Record<string, ConversationComposerDraft>>({})
  const [model, setModel] = useState('')
  const modelExists = selectableModels.some((item) => item.name === model)
  const activeModelName = model === AUTO_MODEL_VALUE ? AUTO_MODEL_VALUE : modelExists ? model : AUTO_MODEL_VALUE
  const [queuedSubmissions, setQueuedSubmissions] = useState<QueuedSubmission[]>([])
  const [attachmentProcessingStatus, setAttachmentProcessingStatus] = useState<AttachmentProcessingStatus | null>(null)
  const [submittedAttachmentsByMessageId, setSubmittedAttachmentsByMessageId] = useState<
    Record<string, SubmittedAttachmentPreview[]>
  >({})
  const [selectedAttachmentPreview, setSelectedAttachmentPreview] = useState<SubmittedAttachmentPreview | null>(null)
  const [failedSubmission, setFailedSubmission] = useState<FailedSubmission | null>(null)
  const queuedSubmissionsRef = useRef<QueuedSubmission[]>([])
  const queueDrainInFlightRef = useRef(false)
  const submittedAttachmentUrlsRef = useRef<Set<string>>(new Set())
  const composerTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  const systemPromptButtonRef = useRef<HTMLButtonElement | null>(null)
  const [stoppedConversationIds, setStoppedConversationIds] = useState<Set<string>>(() => new Set())
  const [conversationPendingDelete, setConversationPendingDelete] = useState<Conversation | null>(null)
  const deleteDialogReturnFocusRef = useRef<HTMLElement | null>(null)

  const {
    activeConversation,
    activeConversationKey,
    activeMessages,
    chat,
    chatConversationId,
    conversations,
    createConversation,
    deleteConversation,
    draftConversationId,
    isStreaming,
    liveMessagesWithModels,
    messageCounts,
    renameConversation,
    selectConversation: persistConversationSelection,
    setDraftConversationId,
    setMessageModels,
    setSessionModel,
    setSystemPrompt,
    streamingConversationIds,
    systemPrompt,
    updateThread
  } = useChatSession()
  const pendingSendRef = useRef<{
    prompt: string
    attachments: File[]
    previousMessages: typeof chat.messages
    previousThreadMessages: ReturnType<typeof uiMessagesToThreadMessages>
    conversationId: string
    submittedModel: string
    submittedAttachmentMessageId?: string
  } | null>(null)
  const pendingRetryRef = useRef<{ conversationId: string; model: string } | null>(null)
  const handledChatErrorRef = useRef<{ conversationId: string; message: string } | null>(null)
  const revokeSubmittedAttachmentPreviews = useCallback((previews: SubmittedAttachmentPreview[]) => {
    for (const preview of previews) {
      revokeObjectUrl(preview.objectUrl)
      submittedAttachmentUrlsRef.current.delete(preview.objectUrl)
    }
  }, [])
  const createSubmittedAttachmentPreviews = useCallback(
    (attachments: File[], conversationId: string, messageId: string): SubmittedAttachmentPreview[] => {
      const counters: Record<SubmittedAttachmentKind, number> = { image: 0, pdf: 0, audio: 0, file: 0 }

      return attachments.map((attachment, index) => {
        const kind = getSubmittedAttachmentKind(attachment)
        counters[kind] += 1
        const objectUrl = createObjectUrl(attachment)
        if (objectUrl) submittedAttachmentUrlsRef.current.add(objectUrl)

        return {
          id: `${attachment.name}-${attachment.lastModified}-${index}`,
          conversationId,
          messageId,
          label: getSubmittedAttachmentLabel(kind, counters[kind]),
          kind,
          fileName: attachment.name || getSubmittedAttachmentLabel(kind, counters[kind]),
          mimeType: attachment.type,
          objectUrl
        }
      })
    },
    []
  )
  const removeSubmittedAttachmentPreviewsForConversation = useCallback(
    (conversationId: string) => {
      setSubmittedAttachmentsByMessageId((current) => {
        let changed = false
        const next = { ...current }

        for (const [messageId, previews] of Object.entries(current)) {
          const removedPreviews = previews.filter((preview) => preview.conversationId === conversationId)
          if (removedPreviews.length === 0) continue

          const keptPreviews = previews.filter((preview) => preview.conversationId !== conversationId)
          revokeSubmittedAttachmentPreviews(removedPreviews)
          if (keptPreviews.length > 0) {
            next[messageId] = keptPreviews
          } else {
            delete next[messageId]
          }
          changed = true
        }

        return changed ? next : current
      })
      setSelectedAttachmentPreview((current) => (current?.conversationId === conversationId ? null : current))
    },
    [revokeSubmittedAttachmentPreviews]
  )
  const removeSubmittedAttachmentPreviewsForMessage = useCallback(
    (messageId: string) => {
      setSubmittedAttachmentsByMessageId((current) => {
        const previews = current[messageId]
        if (!previews) return current

        revokeSubmittedAttachmentPreviews(previews)
        const next = { ...current }
        delete next[messageId]
        return next
      })
      setSelectedAttachmentPreview((current) => (current?.messageId === messageId ? null : current))
    },
    [revokeSubmittedAttachmentPreviews]
  )
  const displayedConversationId = activeConversationKey || chatConversationId
  const composerConversationId = displayedConversationId || draftConversationId
  const composerDraft = useMemo<ConversationComposerDraft>(() => {
    return composerDrafts[composerConversationId] ?? { prompt: '', attachments: [] }
  }, [composerConversationId, composerDrafts])
  const setComposerDraft = useCallback((conversationId: string, draft: ConversationComposerDraft) => {
    setComposerDrafts((current) => ({ ...current, [conversationId]: draft }))
  }, [])
  const updateComposerPrompt = useCallback(
    (nextPrompt: string) => {
      setComposerDrafts((current) => {
        const currentDraft = current[composerConversationId] ?? { prompt: '', attachments: [] }
        return { ...current, [composerConversationId]: { ...currentDraft, prompt: nextPrompt } }
      })
    },
    [composerConversationId]
  )
  const updateComposerAttachments = useCallback(
    (update: (current: File[]) => File[]) => {
      setComposerDrafts((current) => {
        const currentDraft = current[composerConversationId] ?? { prompt: '', attachments: [] }
        return {
          ...current,
          [composerConversationId]: { ...currentDraft, attachments: update(currentDraft.attachments) }
        }
      })
    },
    [composerConversationId]
  )
  const clearComposerDraft = useCallback(
    (conversationId: string) => setComposerDraft(conversationId, { prompt: '', attachments: [] }),
    [setComposerDraft]
  )
  const selectedConversationHasActiveLane = displayedConversationId === chatConversationId
  const composerIsStreaming = selectedConversationHasActiveLane && isStreaming
  const composerShouldQueue = isStreaming || (liveMode && !selectedConversationHasActiveLane)

  const options = useMemo<ModelSelectOption[]>(
    () => [
      AUTO_MODEL_OPTION,
      ...selectableModels.map((item) => ({
        value: item.name,
        label: item.name,
        meta: `${item.family} · ${item.context}`,
        status: modelStatusBadge(item)
      }))
    ],
    [selectableModels]
  )
  const canRetry = hasLastUserTurn(activeMessages.map((message) => ({ role: message.messageRole })))

  const inspectMessage = (message: TransparencyMessage) => {
    if (!transparencyTabEnabled) return

    setInspectedMessage(message)
    setSidebarTab('transparency')
  }
  const selectConversation = (conversation: Conversation) => {
    if (liveMode && isStreaming && liveMessagesWithModels.length > 0) {
      updateThread(chatConversationId, liveMessagesWithModels)
    }
    persistConversationSelection(conversation.id)
    setInspectedMessage(undefined)
    setSidebarTab('conversations')
  }
  const focusComposer = useCallback(() => {
    const focus = () => composerTextareaRef.current?.focus()
    if (typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(focus)
      return
    }

    window.setTimeout(focus, 0)
  }, [])
  const removeQueuedSubmissionsForConversation = useCallback((conversationId: string) => {
    setQueuedSubmissions((current) => {
      const next = current.filter((submission) => submission.conversationId !== conversationId)
      queuedSubmissionsRef.current = next
      return next
    })
  }, [])
  const clearStoppedConversation = useCallback((conversationId: string) => {
    setStoppedConversationIds((current) => {
      if (!current.has(conversationId)) return current

      const next = new Set(current)
      next.delete(conversationId)
      return next
    })
  }, [])
  const requestDeleteConversation = useCallback((conversation: Conversation, options?: DeleteConversationOptions) => {
    deleteDialogReturnFocusRef.current = options?.returnFocusElement ?? null
    setConversationPendingDelete(conversation)
  }, [])
  const openSystemPromptDialog = useCallback(() => {
    setSystemPromptDraft(systemPrompt)
    setSystemPromptDialogOpen(true)
  }, [systemPrompt])
  const updateSystemPromptDialogOpen = useCallback(
    (open: boolean) => {
      if (open) setSystemPromptDraft(systemPrompt)
      setSystemPromptDialogOpen(open)
    },
    [systemPrompt]
  )
  const saveSystemPrompt = useCallback(
    (value: string) => {
      setSystemPrompt(value)
    },
    [setSystemPrompt]
  )
  const confirmDeleteSelectedConversation = useCallback(() => {
    const conversation = conversationPendingDelete
    if (!conversation) return

    const deletingSelectedConversation = conversation.id === activeConversationKey
    const deletingLiveConversation = liveMode && conversation.id === chatConversationId

    if (deletingLiveConversation) {
      if (isStreaming) chat.stop()
      chat.setMessages([])
      pendingSendRef.current = null
      pendingRetryRef.current = null
    }

    if (deletingSelectedConversation) {
      clearComposerDraft(conversation.id)
      setAttachmentProcessingStatus(null)
      setFailedSubmission(null)
      setSelectedAttachmentPreview(null)
      handledChatErrorRef.current = null
      pendingRetryRef.current = null
      setInspectedMessage(undefined)
    }

    removeQueuedSubmissionsForConversation(conversation.id)
    removeSubmittedAttachmentPreviewsForConversation(conversation.id)
    clearStoppedConversation(conversation.id)
    deleteConversation(conversation.id)
    setSidebarTab('conversations')
    focusComposer()
  }, [
    activeConversationKey,
    chat,
    chatConversationId,
    clearStoppedConversation,
    clearComposerDraft,
    conversationPendingDelete,
    deleteConversation,
    focusComposer,
    isStreaming,
    liveMode,
    removeQueuedSubmissionsForConversation,
    removeSubmittedAttachmentPreviewsForConversation
  ])
  const retryLiveData = useCallback(() => {
    void modelsQuery.refetch()
  }, [modelsQuery])
  const switchToTestData = useCallback(() => setMode('harness'), [setMode])

  useEffect(() => {
    setSessionModel(activeModelName)
  }, [activeModelName, setSessionModel])

  useEffect(() => {
    if (chatConversationId) focusComposer()
  }, [chatConversationId, focusComposer])

  useEffect(() => {
    const submittedAttachmentUrls = submittedAttachmentUrlsRef.current
    return () => {
      for (const objectUrl of submittedAttachmentUrls) {
        revokeObjectUrl(objectUrl)
      }
      submittedAttachmentUrls.clear()
    }
  }, [])

  useEffect(() => {
    const pendingSend = pendingSendRef.current
    if (!pendingSend) return

    const nextMessages = chat.messages.slice(pendingSend.previousMessages.length)
    const submittedUserMessage = nextMessages.find((message) => message.role === 'user')
    let submittedAttachmentMessageId = pendingSend.submittedAttachmentMessageId
    if (
      submittedUserMessage &&
      pendingSend.attachments.length > 0 &&
      !submittedAttachmentsByMessageId[submittedUserMessage.id]
    ) {
      const previews = createSubmittedAttachmentPreviews(
        pendingSend.attachments,
        pendingSend.conversationId,
        submittedUserMessage.id
      )
      setSubmittedAttachmentsByMessageId((current) => ({ ...current, [submittedUserMessage.id]: previews }))
      submittedAttachmentMessageId = submittedUserMessage.id
      pendingSendRef.current = { ...pendingSend, submittedAttachmentMessageId: submittedUserMessage.id }
    }
    const submittedMessageIds = nextMessages
      .filter((message) => message.role === 'user' || message.role === 'assistant')
      .map((message) => message.id)
    if (submittedMessageIds.length > 0) {
      setMessageModels((current) => {
        let changed = false
        const next = { ...current }
        for (const messageId of submittedMessageIds) {
          if (next[messageId] === pendingSend.submittedModel) continue
          next[messageId] = pendingSend.submittedModel
          changed = true
        }
        return changed ? next : current
      })
    }

    if (nextMessages.some((message) => message.role === 'assistant' && getMessageTextContent(message) !== '')) {
      pendingSendRef.current = null
      return
    }

    if (chat.error) {
      setComposerDraft(pendingSend.conversationId, {
        prompt: pendingSend.prompt,
        attachments: pendingSend.attachments
      })
      chat.setMessages(pendingSend.previousMessages)
      updateThread(pendingSend.conversationId, pendingSend.previousThreadMessages)
      if (submittedAttachmentMessageId) {
        removeSubmittedAttachmentPreviewsForMessage(submittedAttachmentMessageId)
      }
      handledChatErrorRef.current = { conversationId: pendingSend.conversationId, message: chat.error.message }
      setFailedSubmission({
        id: `failed-${createChatDraftConversationId()}`,
        prompt: pendingSend.prompt,
        attachments: pendingSend.attachments,
        timestamp: new Date().toISOString(),
        conversationId: pendingSend.conversationId,
        errorMessage: chat.error.message,
        model: pendingSend.submittedModel,
        includeUserRow: true
      })
      pendingSendRef.current = null
    }
  }, [
    chat,
    chat.error,
    chat.messages,
    createSubmittedAttachmentPreviews,
    removeSubmittedAttachmentPreviewsForMessage,
    setComposerDraft,
    setMessageModels,
    submittedAttachmentsByMessageId,
    updateThread
  ])

  useEffect(() => {
    const pendingRetry = pendingRetryRef.current
    if (!chat.error || pendingSendRef.current || !pendingRetry) return

    const handledError = handledChatErrorRef.current
    if (handledError?.conversationId === pendingRetry.conversationId && handledError.message === chat.error.message) {
      pendingRetryRef.current = null
      return
    }

    handledChatErrorRef.current = { conversationId: pendingRetry.conversationId, message: chat.error.message }
    setFailedSubmission({
      id: `failed-${createChatDraftConversationId()}`,
      prompt: '',
      attachments: [],
      timestamp: new Date().toISOString(),
      conversationId: pendingRetry.conversationId,
      errorMessage: chat.error.message,
      model: pendingRetry.model,
      includeUserRow: false
    })
    pendingRetryRef.current = null
  }, [chat.error])

  const ensureConversation = useCallback(
    (conversationId = activeConversationKey || chatConversationId) => {
      if (activeConversationKey) return activeConversationKey

      createConversation(conversationId)
      setDraftConversationId(createChatDraftConversationId())
      return conversationId
    },
    [activeConversationKey, chatConversationId, createConversation, setDraftConversationId]
  )

  const submitPromptNow = useCallback(
    async (submission: ComposerSubmission, conversationId = activeConversationKey || chatConversationId) => {
      const promptSnapshot = submission.prompt
      const attachmentsSnapshot = [...submission.attachments]
      const ensuredConversationId = ensureConversation(conversationId)
      clearStoppedConversation(ensuredConversationId)
      setFailedSubmission(null)
      handledChatErrorRef.current = null
      pendingRetryRef.current = null
      pendingSendRef.current = {
        prompt: promptSnapshot,
        attachments: attachmentsSnapshot,
        previousMessages: chat.messages,
        previousThreadMessages: uiMessagesToThreadMessages(chat.messages),
        conversationId: ensuredConversationId,
        submittedModel: activeModelName
      }
      clearComposerDraft(ensuredConversationId)
      if (attachmentsSnapshot.length > 0) {
        const usesBrowserAnalyzer = attachmentsSnapshot.some(usesBrowserAnalyzerForAttachment)
        const browserAnalyzerReady = usesBrowserAnalyzer && isBrowserVisionModelLoaded()
        setAttachmentProcessingStatus({
          conversationId: ensuredConversationId,
          stage: browserAnalyzerReady || !usesBrowserAnalyzer ? 'processing' : 'downloading',
          attachmentCount: attachmentsSnapshot.length,
          prompt: promptSnapshot,
          usesBrowserAnalyzer,
          browserAnalyzerReady
        })
      }

      try {
        const content = await buildComposerMessageContent(submission.prompt, submission.attachments, {
          describeImage: describeImageForPrompt,
          extractPdfText: extractPdfTextFromFile,
          describeScannedPdf,
          onProcessingStage: (stage) => {
            setAttachmentProcessingStatus((current) => {
              if (!current || current.conversationId !== ensuredConversationId) return current
              if (ATTACHMENT_PROCESSING_ORDER[stage] < ATTACHMENT_PROCESSING_ORDER[current.stage]) return current
              return { ...current, stage }
            })
          }
        })
        setAttachmentProcessingStatus((current) => (current?.conversationId === ensuredConversationId ? null : current))
        await chat.sendMessage(content)
      } catch (error) {
        setAttachmentProcessingStatus((current) => (current?.conversationId === ensuredConversationId ? null : current))
        const pendingSend = pendingSendRef.current
        if (pendingSend) {
          setComposerDraft(pendingSend.conversationId, {
            prompt: promptSnapshot,
            attachments: attachmentsSnapshot
          })
          chat.setMessages(pendingSend.previousMessages)
          updateThread(pendingSend.conversationId, pendingSend.previousThreadMessages)
          pendingSendRef.current = null
        }
        const errorMessage = error instanceof Error ? error.message : String(error)
        handledChatErrorRef.current = { conversationId: ensuredConversationId, message: errorMessage }
        setFailedSubmission({
          id: `failed-${createChatDraftConversationId()}`,
          prompt: promptSnapshot,
          attachments: attachmentsSnapshot,
          timestamp: new Date().toISOString(),
          conversationId: ensuredConversationId,
          errorMessage,
          model: activeModelName,
          includeUserRow: true
        })
      }
    },
    [
      activeConversationKey,
      activeModelName,
      chat,
      chatConversationId,
      clearComposerDraft,
      clearStoppedConversation,
      ensureConversation,
      setComposerDraft,
      updateThread
    ]
  )

  const sendPrompt = useCallback(async () => {
    const nextPrompt = composerDraft.prompt.trim()
    if (!nextPrompt && composerDraft.attachments.length === 0) return

    const submission: ComposerSubmission = { prompt: composerDraft.prompt, attachments: [...composerDraft.attachments] }

    if (composerShouldQueue) {
      const queued: QueuedSubmission = {
        ...submission,
        id: createQueuedSubmissionId(),
        timestamp: new Date().toISOString(),
        conversationId: composerConversationId
      }
      setQueuedSubmissions((current) => {
        const next = [...current, queued]
        queuedSubmissionsRef.current = next
        return next
      })
      setFailedSubmission(null)
      handledChatErrorRef.current = null
      pendingRetryRef.current = null
      clearComposerDraft(composerConversationId)
      return
    }

    await submitPromptNow(submission, composerConversationId)
  }, [clearComposerDraft, composerConversationId, composerDraft, composerShouldQueue, submitPromptNow])

  useEffect(() => {
    queuedSubmissionsRef.current = queuedSubmissions
  }, [queuedSubmissions])

  useEffect(() => {
    if (isStreaming || queueDrainInFlightRef.current || queuedSubmissions.length === 0) return

    const nextSubmission = queuedSubmissions.find((submission) => submission.conversationId === chatConversationId)
    if (!nextSubmission) return

    queueDrainInFlightRef.current = true
    setQueuedSubmissions((current) => {
      const next = current.filter((submission) => submission.id !== nextSubmission.id)
      queuedSubmissionsRef.current = next
      return next
    })
    void (async () => {
      try {
        await submitPromptNow(
          { prompt: nextSubmission.prompt, attachments: [...nextSubmission.attachments] },
          nextSubmission.conversationId
        )
      } finally {
        queueDrainInFlightRef.current = false
      }
    })()
  }, [chatConversationId, isStreaming, queuedSubmissions, submitPromptNow])

  const removeQueuedSubmission = useCallback(
    (submissionId: string) => {
      setQueuedSubmissions((current) => {
        const next = current.filter((submission) => submission.id !== submissionId)
        queuedSubmissionsRef.current = next
        return next
      })
      focusComposer()
    },
    [focusComposer]
  )

  const retryLastResponse = useCallback(async () => {
    if (!canRetry) return
    const ensuredConversationId = ensureConversation()
    clearStoppedConversation(ensuredConversationId)
    setFailedSubmission(null)
    handledChatErrorRef.current = null
    pendingRetryRef.current = { conversationId: ensuredConversationId, model: activeModelName }
    await chat.reload()
  }, [activeModelName, canRetry, chat, clearStoppedConversation, ensureConversation])

  const stopStreamingResponse = useCallback(() => {
    const latestLiveMessage = liveMessagesWithModels.at(-1)
    if (liveMode && latestLiveMessage?.messageRole !== 'assistant') {
      const stoppedMessage = createStoppedAssistantThreadMessage(activeModelName)
      updateThread(chatConversationId, [...liveMessagesWithModels, stoppedMessage])
      chat.setMessages([
        ...chat.messages,
        {
          id: stoppedMessage.id,
          role: 'assistant',
          createdAt: new Date(stoppedMessage.timestamp),
          parts: [{ type: 'text', content: '' }]
        }
      ])
    } else if (liveMode && latestLiveMessage?.messageRole === 'assistant') {
      updateThread(chatConversationId, liveMessagesWithModels)
    }
    chat.stop()
    setStoppedConversationIds((current) => {
      if (current.has(chatConversationId)) return current

      const next = new Set(current)
      next.add(chatConversationId)
      return next
    })
  }, [activeModelName, chat, chatConversationId, liveMessagesWithModels, liveMode, updateThread])

  const visibleFailedSubmission =
    failedSubmission && failedSubmission.conversationId === displayedConversationId ? failedSubmission : null
  const visibleQueuedSubmissions = queuedSubmissions.filter(
    (submission) => submission.conversationId === displayedConversationId
  )
  const visibleAttachmentProcessingStatus =
    attachmentProcessingStatus && attachmentProcessingStatus.conversationId === displayedConversationId
      ? attachmentProcessingStatus
      : null
  const composerIsPreparingAttachments =
    attachmentProcessingStatus?.conversationId === composerConversationId &&
    attachmentProcessingStatus.attachmentCount > 0
  const activeConversationIsStreaming = streamingConversationIds.includes(displayedConversationId)
  const lastActiveMessage = activeMessages.at(-1)
  const lastMessageIsEmptyAssistant =
    lastActiveMessage?.messageRole === 'assistant' && lastActiveMessage.body.trim() === ''
  const lastMessageHasAssistantText =
    lastActiveMessage?.messageRole === 'assistant' && lastActiveMessage.body.trim() !== ''
  const showStreamingPlaceholder =
    activeConversationIsStreaming && !lastMessageIsEmptyAssistant && !lastMessageHasAssistantText

  const sidebar = (
    <ChatSidebar
      tab={sidebarTab}
      onTabChange={setSidebarTab}
      conversations={conversations.conversations}
      conversationGroups={conversations.conversationGroups}
      activeId={conversations.activeConversationId || activeConversation?.id}
      messageCounts={messageCounts}
      streamingConversationIds={streamingConversationIds}
      onSelectConversation={selectConversation}
      onRenameConversation={(conversation, title) => renameConversation(conversation.id, title)}
      onDeleteConversation={requestDeleteConversation}
      onNewChat={() => {
        if (liveMode && isStreaming && liveMessagesWithModels.length > 0) {
          updateThread(chatConversationId, liveMessagesWithModels)
        }
        const nextConversationId = createConversation(draftConversationId)
        setDraftConversationId(createChatDraftConversationId())
        clearComposerDraft(nextConversationId)
        setAttachmentProcessingStatus(null)
        setFailedSubmission(null)
        setSelectedAttachmentPreview(null)
        handledChatErrorRef.current = null
        pendingRetryRef.current = null
        setInspectedMessage(undefined)
        setSidebarTab('conversations')
        focusComposer()
      }}
      transparency={<TransparencyPane message={inspectedMessage} nodes={data.transparencyNodes} />}
      showTransparency={transparencyTabEnabled}
    />
  )

  const actions = (
    <>
      {data.actionMetrics.map((metric) => (
        <ChatMetricBadge key={metric.id} metric={metric} />
      ))}
      <div className="flex min-w-0 flex-1 basis-full items-center gap-2 sm:basis-auto md:flex-none">
        <span className="hidden shrink-0 whitespace-nowrap text-[length:var(--density-type-caption)] text-fg-faint md:inline">
          {data.modelLabel}
        </span>
        <ModelSelect options={options} value={activeModelName} onChange={setModel} />
      </div>
    </>
  )

  if (showLiveError) {
    return (
      <LiveDataUnavailableOverlay
        debugTitle="Could not reach the local model catalog"
        title="Live chat models are unavailable"
        debugDescription="Chat could not fetch the initial model catalog from the configured API target. Start the backend, verify the endpoint, or switch Data source back to Harness in Tweaks while debugging."
        productionDescription="Chat is waiting for the live model catalog before starting a conversation. Keep the page open while the service recovers, or switch Data source back to Harness in Tweaks to inspect sample conversations."
        onRetry={retryLiveData}
        onSwitchToTestData={switchToTestData}
      >
        <ChatLiveLoadingGhost />
      </LiveDataUnavailableOverlay>
    )
  }

  if (showLiveLoading) {
    return <ChatLiveLoadingGhost />
  }

  return (
    <>
      <DestructiveActionDialog
        open={conversationPendingDelete !== null}
        onOpenChange={(open) => {
          if (!open) setConversationPendingDelete(null)
        }}
        title={`Delete "${conversationPendingDelete?.title ?? 'chat'}"?`}
        description="This permanently removes the selected chat and its message history from local storage. This action cannot be undone."
        destructiveLabel="Delete chat"
        onConfirm={confirmDeleteSelectedConversation}
        returnFocusRef={deleteDialogReturnFocusRef}
      />
      <TextInputDialog
        open={systemPromptDialogOpen}
        onOpenChange={updateSystemPromptDialogOpen}
        title="Set system prompt"
        description="Saved instructions are sent before every chat message in this browser. Leave it empty to use the model defaults."
        label="System prompt"
        value={systemPromptDraft}
        onValueChange={setSystemPromptDraft}
        onSave={saveSystemPrompt}
        placeholder="You are a careful mesh-llm operator. Keep answers grounded in the current cluster state."
        saveLabel="Save prompt"
        returnFocusRef={systemPromptButtonRef}
      />
      <AttachmentPreviewDialog
        attachment={selectedAttachmentPreview}
        onOpenChange={(open) => {
          if (!open) setSelectedAttachmentPreview(null)
        }}
      />
      <ChatLayout
        sidebar={sidebar}
        hideSidebar={conversations.conversations.length === 0}
        title={data.title}
        subtitle={activeConversation?.title}
        actions={actions}
        composer={
          <Composer
            key={composerConversationId}
            value={composerDraft.prompt}
            onChange={updateComposerPrompt}
            onAttach={(files) => {
              setFailedSubmission(null)
              handledChatErrorRef.current = null
              pendingRetryRef.current = null
              updateComposerAttachments((current) => [...current, ...files])
            }}
            attachmentCount={composerDraft.attachments.length}
            disabled={composerIsPreparingAttachments || !canChat}
            isPreparingAttachments={composerIsPreparingAttachments}
            preparingStage={attachmentProcessingStatus?.stage}
            preparingAttachmentCount={attachmentProcessingStatus?.attachmentCount ?? 0}
            onSystemPrompt={openSystemPromptDialog}
            onSend={() => void sendPrompt()}
            onStop={stopStreamingResponse}
            onRetry={() => void retryLastResponse()}
            canRetry={canRetry}
            isStreaming={composerIsStreaming}
            sendMode={composerShouldQueue ? 'queue' : 'send'}
            textareaRef={composerTextareaRef}
            systemPromptButtonRef={systemPromptButtonRef}
            showSystemPromptButton={systemPromptButtonEnabled}
            placeholder={canChat ? 'Ask me anything...' : 'Waiting for a warm model...'}
          />
        }
        onMessageAreaClick={() => setInspectedMessage(undefined)}
      >
        {activeMessages.length === 0 &&
        !showStreamingPlaceholder &&
        visibleQueuedSubmissions.length === 0 &&
        !visibleAttachmentProcessingStatus &&
        !visibleFailedSubmission ? (
          <EmptyState
            tone="accent"
            icon={<MessageSquareMore aria-hidden={true} className="size-10" strokeWidth={1.4} />}
            title="Start Chatting"
            description={
              conversations.conversations.length === 0 ? (
                'Type a message below to begin. Your chats stay in this browser, and the mesh routes requests automatically.'
              ) : (
                <>
                  No messages yet. Send a message to begin a fresh conversation; replies use{' '}
                  <span className="font-mono text-fg">{activeModelName}</span> unless you choose another model.
                </>
              )
            }
          />
        ) : null}
        {activeMessages.map((message) => {
          const transparencyMessage = message.inspectMessage
          const messageAttachments = submittedAttachmentsByMessageId[message.id] ?? []
          const isLatestAssistantMessage = message.messageRole === 'assistant' && message.id === lastActiveMessage?.id
          const messageIsStreamingResponse = activeConversationIsStreaming && isLatestAssistantMessage
          const messageWasStopped = stoppedConversationIds.has(displayedConversationId) && isLatestAssistantMessage
          return (
            <MessageRow
              key={message.id}
              messageRole={message.messageRole}
              timestamp={message.timestamp}
              model={message.model}
              state={messageIsStreamingResponse ? 'streaming' : messageWasStopped ? 'stopped' : 'default'}
              body={message.body}
              route={message.route}
              routeNode={message.routeNode}
              showRouteMetadata={transparencyTabEnabled}
              tokens={message.tokens}
              tokPerSec={message.tokPerSec}
              ttft={message.ttft}
              inspect={
                transparencyTabEnabled && transparencyMessage ? () => inspectMessage(transparencyMessage) : undefined
              }
              inspectLabel={message.inspectLabel}
              inspected={transparencyMessage != null && inspectedMessage?.id === transparencyMessage.id}
              onStopStreaming={stopStreamingResponse}
              attachments={messageAttachments.map((attachment): MessageAttachmentAction => {
                return {
                  id: attachment.id,
                  label: attachment.label,
                  kind: attachment.kind,
                  fileName: attachment.fileName,
                  onOpen: () => setSelectedAttachmentPreview(attachment)
                }
              })}
            />
          )
        })}
        {visibleAttachmentProcessingStatus ? (
          <AttachmentProcessingPanel status={visibleAttachmentProcessingStatus} />
        ) : null}
        {visibleFailedSubmission ? (
          <>
            {visibleFailedSubmission.includeUserRow ? (
              <MessageRow
                key={`${visibleFailedSubmission.id}-user`}
                messageRole="user"
                timestamp={visibleFailedSubmission.timestamp}
                model={visibleFailedSubmission.model}
                state="default"
                body={getSubmissionBody(visibleFailedSubmission)}
                showRouteMetadata={false}
              />
            ) : null}
            <MessageRow
              key={`${visibleFailedSubmission.id}-error`}
              messageRole="assistant"
              timestamp={visibleFailedSubmission.timestamp}
              model={visibleFailedSubmission.model}
              state="error"
              body={visibleFailedSubmission.errorMessage}
              showRouteMetadata={false}
            />
          </>
        ) : null}
        {showStreamingPlaceholder ? (
          <MessageRow
            key="streaming-response-placeholder"
            messageRole="assistant"
            timestamp="Now"
            model={activeModelName}
            state="streaming"
            body=""
            showRouteMetadata={false}
            onStopStreaming={stopStreamingResponse}
          />
        ) : null}
        {visibleQueuedSubmissions.map((submission) => (
          <MessageRow
            key={submission.id}
            messageRole="user"
            timestamp={submission.timestamp}
            model="Queued"
            state="queued"
            body={getQueuedSubmissionBody(submission)}
            showRouteMetadata={false}
            onRemoveQueued={() => removeQueuedSubmission(submission.id)}
          />
        ))}
      </ChatLayout>
    </>
  )
}

export function ChatPage(props: ChatPageProps = {}) {
  const existingSession = useOptionalChatSession()
  if (existingSession) {
    return <ChatPageContent {...props} />
  }

  return (
    <ChatSessionProvider data={props.data ?? CHAT_HARNESS}>
      <ChatPageContent {...props} />
    </ChatSessionProvider>
  )
}
