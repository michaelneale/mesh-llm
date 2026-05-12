import { configure, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { MultimodalContent } from '@tanstack/ai-client'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { CHAT_HARNESS } from '@/features/app-tabs/data'
import { APP_STORAGE_KEYS } from '@/features/app-tabs/data'
import { ChatSessionProvider } from '@/features/chat/api/chat-session'
import { loadChatState, saveChatState, trimThreadMessages } from '@/features/chat/api/chat-storage'
import { ChatLayout } from '@/features/chat/layouts/ChatLayout'
import { ChatPage, ChatPageContent } from '@/features/chat/pages/ChatPage'
import { adaptModelsToSummary } from '@/features/network/api/models-adapter'
import { DataModeProvider } from '@/lib/data-mode/DataModeContext'
import { FeatureFlagProvider } from '@/lib/feature-flags'

const scrollIntoViewMock = vi.fn()
const createObjectUrlMock = vi.fn((file: File) => `blob:preview/${file.name}`)
const revokeObjectUrlMock = vi.fn()

configure({ asyncUtilTimeout: 4_000 })

function installPointerCaptureShim() {
  Object.defineProperty(HTMLElement.prototype, 'hasPointerCapture', {
    configurable: true,
    value: () => false
  })
  Object.defineProperty(HTMLElement.prototype, 'setPointerCapture', {
    configurable: true,
    value: () => undefined
  })
  Object.defineProperty(HTMLElement.prototype, 'releasePointerCapture', {
    configurable: true,
    value: () => undefined
  })
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
    configurable: true,
    value: scrollIntoViewMock
  })
}

function installImageFallbackShim() {
  class TestImage {
    width = 0
    height = 0
    onload: (() => void) | null = null
    onerror: (() => void) | null = null

    set src(value: string) {
      void value
      window.setTimeout(() => this.onerror?.(), 0)
    }
  }

  Object.defineProperty(globalThis, 'Image', {
    configurable: true,
    value: TestImage
  })
}

function installObjectUrlShim() {
  Object.defineProperty(URL, 'createObjectURL', {
    configurable: true,
    value: createObjectUrlMock
  })
  Object.defineProperty(URL, 'revokeObjectURL', {
    configurable: true,
    value: revokeObjectUrlMock
  })
}

const chatMock = vi.hoisted(() => {
  function createUiMessage(id: string, role: 'user' | 'assistant', body: string) {
    return {
      id,
      role,
      createdAt: new Date('2026-05-06T00:00:00.000Z'),
      parts: [{ type: 'text' as const, content: body }]
    }
  }

  const state = {
    messagesByConversation: new Map<string, ReturnType<typeof createUiMessage>[]>(),
    statusByConversation: new Map<string, 'ready' | 'submitted' | 'streaming' | 'error'>(),
    errorByConversation: new Map<string, Error | undefined>(),
    sendAssistantText: 'Partial assistant reply',
    sendResponseMetadata: undefined as
      | undefined
      | {
          model?: string
          usage?: { input_tokens: number; output_tokens: number; total_tokens?: number }
          timings?: { decode_time_ms?: number; ttft_ms?: number; total_time_ms?: number }
          servedBy?: string
        },
    sendStatus: 'streaming' as 'ready' | 'submitted' | 'streaming',
    sendErrorMessage: undefined as string | undefined,
    sendErrorResolves: false,
    sendOptimisticStatusBeforeError: false,
    sendOptimisticUserMessageBeforeError: false,
    sendOptimisticAssistantPlaceholderBeforeError: false,
    reloadAssistantText: 'Retried assistant reply',
    reloadStatus: 'ready' as const,
    reloadErrorMessage: undefined as string | undefined,
    stopCalls: [] as string[],
    sendCalls: [] as Array<{
      conversationId: string
      content: string | MultimodalContent
      model: string
      systemPrompt: string
    }>,
    reloadCalls: [] as string[],
    hookConversationIds: [] as string[],
    hookSystemPrompts: [] as Array<{ conversationId: string; systemPrompt: string }>,
    hookUnmounts: [] as string[],
    reset() {
      state.messagesByConversation.clear()
      state.statusByConversation.clear()
      state.errorByConversation.clear()
      state.sendAssistantText = 'Partial assistant reply'
      state.sendResponseMetadata = undefined
      state.sendStatus = 'streaming'
      state.sendErrorMessage = undefined
      state.sendErrorResolves = false
      state.sendOptimisticStatusBeforeError = false
      state.sendOptimisticUserMessageBeforeError = false
      state.sendOptimisticAssistantPlaceholderBeforeError = false
      state.reloadAssistantText = 'Retried assistant reply'
      state.reloadStatus = 'ready'
      state.reloadErrorMessage = undefined
      state.stopCalls = []
      state.sendCalls = []
      state.reloadCalls = []
      state.hookConversationIds = []
      state.hookSystemPrompts = []
      state.hookUnmounts = []
    },
    createUiMessage
  }

  return state
})

const attachmentPreprocessingMock = vi.hoisted(() => ({
  describeImageForPrompt: vi.fn(),
  extractPdfTextFromFile: vi.fn(),
  describeScannedPdf: vi.fn(),
  isBrowserVisionModelLoaded: vi.fn()
}))

type TestAttachmentProcessingStage = 'downloading' | 'starting' | 'processing'

function createDeferred<T>() {
  let resolveDeferred: ((value: T) => void) | undefined
  let rejectDeferred: ((reason?: unknown) => void) | undefined
  const promise = new Promise<T>((resolve, reject) => {
    resolveDeferred = resolve
    rejectDeferred = reject
  })

  return {
    promise,
    resolve(value: T) {
      resolveDeferred?.(value)
    },
    reject(reason?: unknown) {
      rejectDeferred?.(reason)
    }
  }
}

vi.mock('@/features/chat/api/chat-storage', () => ({
  MAX_CHAT_CONVERSATIONS: 80,
  clearChatState: vi.fn(),
  loadChatState: vi.fn(),
  saveChatState: vi.fn(),
  trimThreadMessages: vi.fn((messages: Array<unknown>) => messages)
}))

vi.mock('@/features/network/api/use-models-query', () => ({
  useModelsQuery: vi.fn(() => ({ data: { mesh_models: [] }, isFetching: false, isError: false, refetch: vi.fn() }))
}))

vi.mock('@/features/network/api/use-status-query', () => ({
  useStatusQuery: vi.fn(() => ({ data: undefined }))
}))

vi.mock('@/features/network/api/models-adapter', () => ({
  adaptModelsToSummary: vi.fn(() => CHAT_HARNESS.models)
}))

vi.mock('@/features/chat/api/attachment-preprocessing', () => ({
  describeImageForPrompt: attachmentPreprocessingMock.describeImageForPrompt,
  extractPdfTextFromFile: attachmentPreprocessingMock.extractPdfTextFromFile,
  describeScannedPdf: attachmentPreprocessingMock.describeScannedPdf,
  isBrowserVisionModelLoaded: attachmentPreprocessingMock.isBrowserVisionModelLoaded
}))

vi.mock('@/features/chat/api/use-chat', async () => {
  const React = await import('react')

  function threadToUiMessage(message: {
    id: string
    messageRole: 'user' | 'assistant'
    timestamp: string
    body: string
  }) {
    return {
      id: message.id,
      role: message.messageRole,
      createdAt: new Date(message.timestamp),
      parts: [{ type: 'text' as const, content: message.body }]
    }
  }

  return {
    useMeshChat: vi.fn(
      ({
        conversationId,
        model,
        systemPrompt,
        initialMessages,
        onResponseMetadata
      }: {
        conversationId: string
        model: string
        systemPrompt: string
        initialMessages: Array<{ id: string; messageRole: 'user' | 'assistant'; timestamp: string; body: string }>
        onResponseMetadata?: (metadata: {
          messageId: string
          model?: string
          usage?: { input_tokens: number; output_tokens: number; total_tokens?: number }
          timings?: { decode_time_ms?: number; ttft_ms?: number; total_time_ms?: number }
          servedBy?: string
        }) => void
      }) => {
        React.useEffect(() => {
          chatMock.hookConversationIds.push(conversationId)
          return () => {
            chatMock.hookUnmounts.push(conversationId)
          }
        }, [conversationId])

        React.useEffect(() => {
          chatMock.hookSystemPrompts.push({ conversationId, systemPrompt })
        }, [conversationId, systemPrompt])

        const initialUiMessages = React.useMemo(() => initialMessages.map(threadToUiMessage), [initialMessages])
        const [messages, setMessages] = React.useState(
          () => chatMock.messagesByConversation.get(conversationId) ?? initialUiMessages
        )
        const [status, setStatus] = React.useState<'ready' | 'submitted' | 'streaming' | 'error'>(
          () => chatMock.statusByConversation.get(conversationId) ?? 'ready'
        )
        const [error, setError] = React.useState<Error | undefined>(() =>
          chatMock.errorByConversation.get(conversationId)
        )
        const messagesRef = React.useRef(messages)

        React.useEffect(() => {
          const nextMessages = chatMock.messagesByConversation.get(conversationId) ?? initialUiMessages
          setMessages(nextMessages)
          setStatus(chatMock.statusByConversation.get(conversationId) ?? 'ready')
          setError(chatMock.errorByConversation.get(conversationId))
        }, [conversationId, initialUiMessages])

        React.useEffect(() => {
          messagesRef.current = messages
          chatMock.messagesByConversation.set(conversationId, messages)
        }, [conversationId, messages])

        React.useEffect(() => {
          chatMock.statusByConversation.set(conversationId, status)
        }, [conversationId, status])

        React.useEffect(() => {
          chatMock.errorByConversation.set(conversationId, error)
        }, [conversationId, error])

        return {
          messages,
          sendMessage: vi.fn(async (content: string | MultimodalContent) => {
            chatMock.sendCalls.push({ conversationId, content, model, systemPrompt })
            const body =
              typeof content === 'string'
                ? content
                : (() => {
                    if (typeof content.content === 'string') return content.content
                    const textPart = content.content.find(
                      (part): part is { type: 'text'; content: string } => part.type === 'text'
                    )
                    return textPart?.content ?? ''
                  })()
            if (chatMock.sendErrorMessage) {
              if (chatMock.sendOptimisticUserMessageBeforeError) {
                const optimisticMessages = [
                  ...messagesRef.current,
                  chatMock.createUiMessage(`user-${chatMock.sendCalls.length}`, 'user', body)
                ]
                if (chatMock.sendOptimisticAssistantPlaceholderBeforeError) {
                  optimisticMessages.push(
                    chatMock.createUiMessage(`assistant-${chatMock.sendCalls.length}`, 'assistant', '')
                  )
                }
                setMessages(optimisticMessages)
              }
              if (chatMock.sendOptimisticStatusBeforeError) {
                setStatus(chatMock.sendStatus)
                await Promise.resolve()
              }
              const sendError = new Error(chatMock.sendErrorMessage)
              setError(sendError)
              setStatus('error')
              if (chatMock.sendErrorResolves) {
                return
              }
              throw sendError
            }
            const userMessageId = `user-${chatMock.sendCalls.length}`
            const assistantMessageId = `assistant-${chatMock.sendCalls.length}`
            const nextMessages = [
              ...messagesRef.current,
              chatMock.createUiMessage(userMessageId, 'user', body),
              chatMock.createUiMessage(assistantMessageId, 'assistant', chatMock.sendAssistantText)
            ]
            setMessages(nextMessages)
            if (chatMock.sendResponseMetadata) {
              onResponseMetadata?.({ messageId: assistantMessageId, ...chatMock.sendResponseMetadata })
            }
            setStatus(chatMock.sendStatus)
            setError(undefined)
          }),
          reload: vi.fn(async () => {
            chatMock.reloadCalls.push(conversationId)
            const currentMessages = messagesRef.current
            let lastUserIndex = -1
            for (let index = currentMessages.length - 1; index >= 0; index -= 1) {
              if (currentMessages[index]?.role === 'user') {
                lastUserIndex = index
                break
              }
            }
            if (lastUserIndex < 0) return

            const nextMessages = [
              ...currentMessages.slice(0, lastUserIndex + 1),
              chatMock.createUiMessage(
                `assistant-retry-${chatMock.reloadCalls.length}`,
                'assistant',
                chatMock.reloadAssistantText
              )
            ]

            setMessages(nextMessages)
            setStatus(chatMock.reloadStatus)
            setError(chatMock.reloadErrorMessage ? new Error(chatMock.reloadErrorMessage) : undefined)
          }),
          stop: vi.fn(() => {
            chatMock.stopCalls.push(conversationId)
            chatMock.statusByConversation.set(conversationId, 'ready')
            setStatus('ready')
          }),
          status,
          error,
          isLoading: status === 'submitted' || status === 'streaming',
          setMessages,
          append: vi.fn(),
          addToolResult: vi.fn(),
          addToolApprovalResponse: vi.fn(),
          isSubscribed: false,
          connectionStatus: 'disconnected',
          sessionGenerating: false,
          clear: vi.fn()
        }
      }
    )
  }
})

function renderChatPage({
  transparencyTabEnabled = false,
  systemPromptButtonEnabled = false,
  mode = 'harness'
}: {
  transparencyTabEnabled?: boolean
  systemPromptButtonEnabled?: boolean
  mode?: 'live' | 'harness'
} = {}) {
  if (transparencyTabEnabled || systemPromptButtonEnabled) {
    window.localStorage.setItem(
      APP_STORAGE_KEYS.featureFlagOverrides,
      JSON.stringify({
        chat: { transparencyTab: transparencyTabEnabled, systemPromptButton: systemPromptButtonEnabled }
      })
    )
  }

  render(
    <FeatureFlagProvider>
      <DataModeProvider initialMode={mode} persist={false}>
        <ChatPage />
      </DataModeProvider>
    </FeatureFlagProvider>
  )
}

function renderPersistentChatRoute(showChat: boolean) {
  return render(
    <FeatureFlagProvider>
      <DataModeProvider initialMode="live" persist={false}>
        <ChatSessionProvider>
          {showChat ? <ChatPageContent /> : <div data-testid="network-route">Network route</div>}
        </ChatSessionProvider>
      </DataModeProvider>
    </FeatureFlagProvider>
  )
}

function queryAllByTextContent(text: string) {
  return screen.queryAllByText((_, element) => element?.textContent?.includes(text) ?? false, { selector: 'span' })
}

function shortTimestamp(date: Date) {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit'
  }).format(date)
}

function setLocalTime(date: Date, hours: number, minutes: number) {
  const timestamp = new Date(date)
  timestamp.setHours(hours, minutes, 0, 0)

  return timestamp
}

describe('ChatPage', () => {
  beforeEach(() => {
    scrollIntoViewMock.mockClear()
    createObjectUrlMock.mockClear()
    revokeObjectUrlMock.mockClear()
    installPointerCaptureShim()
    installImageFallbackShim()
    installObjectUrlShim()
    window.localStorage.removeItem(APP_STORAGE_KEYS.featureFlagOverrides)
    window.localStorage.removeItem(APP_STORAGE_KEYS.chatSystemPrompt)
    vi.mocked(loadChatState).mockResolvedValue(undefined)
    vi.mocked(saveChatState).mockResolvedValue(undefined)
    vi.mocked(trimThreadMessages).mockImplementation((messages) => messages)
    vi.mocked(adaptModelsToSummary).mockReturnValue(CHAT_HARNESS.models)
    attachmentPreprocessingMock.describeImageForPrompt.mockReset()
    attachmentPreprocessingMock.extractPdfTextFromFile.mockReset()
    attachmentPreprocessingMock.describeScannedPdf.mockReset()
    attachmentPreprocessingMock.isBrowserVisionModelLoaded.mockReset()
    attachmentPreprocessingMock.isBrowserVisionModelLoaded.mockReturnValue(false)
    attachmentPreprocessingMock.describeImageForPrompt.mockResolvedValue({
      imageDescription: '[Image description: A tabby cat]'
    })
    attachmentPreprocessingMock.extractPdfTextFromFile.mockResolvedValue({
      text: '',
      pageCount: 1,
      pagesWithText: 0,
      wordCount: 0
    })
    attachmentPreprocessingMock.describeScannedPdf.mockResolvedValue('[Page 1]\n[Image description: A scanned receipt]')
    chatMock.reset()
  })

  it('deselects the inspected message when the message area background is clicked', async () => {
    const user = userEvent.setup()

    renderChatPage({ transparencyTabEnabled: true })

    await user.click(screen.getByRole('button', { name: 'Inspect transparency' }))

    expect(screen.queryByText('No message selected')).not.toBeInTheDocument()

    await user.click(screen.getByTestId('chat-message-list'))

    expect(screen.getByText('No message selected')).toBeInTheDocument()
  })

  it('bounds the chat layout and makes the transcript the styled scroll container', () => {
    render(
      <ChatLayout
        actions={null}
        composer={<textarea aria-label="Prompt" />}
        sidebar={<div role="tablist" aria-label="Chat sidebar views" />}
        title="Chat"
      >
        <div data-testid="message-content">Messages</div>
      </ChatLayout>
    )

    expect(screen.getByTestId('chat-layout')).toHaveStyle({
      height: 'calc(100dvh - 180px)',
      maxHeight: 'calc(100dvh - 180px)'
    })
    expect(screen.getByTestId('chat-message-list')).toHaveClass(
      'chat-message-scrollbar',
      'overflow-y-auto',
      'overflow-x-hidden'
    )
  })

  it('falls back to harness conversations when persisted chat state is malformed', async () => {
    vi.mocked(loadChatState).mockResolvedValue({} as Awaited<ReturnType<typeof loadChatState>>)

    renderChatPage()

    expect(await screen.findAllByText('Routing latency notes')).not.toHaveLength(0)
  })

  it('does not show harness-scoped persisted conversations when rendering live mode', async () => {
    vi.mocked(loadChatState).mockImplementation(async (scope) => {
      if (scope !== 'harness') return undefined
      return {
        conversations: [{ id: 'persisted-harness', title: 'Harness persisted only', subtitle: '', updatedAt: 'Now' }],
        conversationGroups: [
          { title: 'Today', conversationIds: ['persisted-harness'] },
          { title: 'Earlier', conversationIds: [] }
        ],
        threads: { 'persisted-harness': [] },
        activeConversationId: 'persisted-harness'
      }
    })

    renderChatPage({ mode: 'live' })

    expect(await screen.findByText('Start your first conversation')).toBeInTheDocument()
    expect(screen.queryByText('Harness persisted only')).not.toBeInTheDocument()
  })

  it('shows auto as the first chat model selector option by default', async () => {
    const user = userEvent.setup()

    renderChatPage()

    const trigger = screen.getByRole('combobox', { name: 'Select model' })
    expect(trigger).toHaveTextContent('Auto (router picks best)')

    await user.click(trigger)

    const options = await screen.findAllByRole('option')
    expect(options[0]).toHaveTextContent('Auto')
  })

  it('excludes cold live models from the chat model selector', async () => {
    const user = userEvent.setup()
    vi.mocked(adaptModelsToSummary).mockReturnValue([
      { ...CHAT_HARNESS.models[0], name: 'warm-model', status: 'warm' },
      { ...CHAT_HARNESS.models[1], name: 'cold-model', status: 'offline' }
    ])

    renderChatPage({ mode: 'live' })

    await user.click(screen.getByRole('combobox', { name: 'Select model' }))

    const options = await screen.findAllByRole('option')
    expect(options.map((option) => option.textContent)).toEqual([
      expect.stringContaining('Auto'),
      expect.stringContaining('warm-model')
    ])
    expect(screen.queryByText('cold-model')).not.toBeInTheDocument()
  })

  it('loads the default system prompt when no user override is stored', async () => {
    const user = userEvent.setup()
    localStorage.clear()

    renderChatPage({ systemPromptButtonEnabled: true })

    const systemPromptButton = screen.getByRole('button', { name: 'System prompt' })
    await user.click(systemPromptButton)
    const dialog = screen.getByRole('dialog', { name: 'Set system prompt' })

    const textarea = within(dialog).getByLabelText('System prompt') as HTMLTextAreaElement
    expect(textarea.value).toContain('You are a helpful assistant running inside MeshLLM.')
  })

  it('hides the system prompt button until the chat feature flag is enabled', () => {
    renderChatPage()

    expect(screen.queryByRole('button', { name: 'System prompt' })).not.toBeInTheDocument()
  })

  it('opens, saves, prefills, and sends the chat-wide system prompt for new chats', async () => {
    const user = userEvent.setup()

    renderChatPage({ systemPromptButtonEnabled: true })

    const systemPromptButton = screen.getByRole('button', { name: 'System prompt' })
    await user.click(systemPromptButton)
    const dialog = screen.getByRole('dialog', { name: 'Set system prompt' })
    expect(dialog).toBeInTheDocument()

    const textarea = within(dialog).getByLabelText('System prompt')
    fireEvent.change(textarea, { target: { value: '' } })

    await user.type(textarea, 'Answer as a mesh-llm operator.')
    await user.click(screen.getByRole('button', { name: 'Save prompt' }))

    await waitFor(() => {
      expect(window.localStorage.getItem(APP_STORAGE_KEYS.chatSystemPrompt)).toBe('Answer as a mesh-llm operator.')
      expect(systemPromptButton).toHaveFocus()
    })

    await user.click(systemPromptButton)
    expect(
      within(screen.getByRole('dialog', { name: 'Set system prompt' })).getByLabelText('System prompt')
    ).toHaveValue('Answer as a mesh-llm operator.')
    await user.click(screen.getByRole('button', { name: 'Cancel' }))

    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.type(screen.getByLabelText('Prompt'), 'Summarize active models')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls[0]).toMatchObject({
        content: 'Summarize active models',
        model: 'auto',
        systemPrompt: 'Answer as a mesh-llm operator.'
      })
    })
  })

  it('does not send a persisted system prompt while the feature flag is disabled', async () => {
    const user = userEvent.setup()
    window.localStorage.setItem(APP_STORAGE_KEYS.chatSystemPrompt, 'Hidden saved instruction')

    renderChatPage()

    expect(screen.queryByRole('button', { name: 'System prompt' })).not.toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Route without hidden instructions')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls[0]).toMatchObject({
        content: 'Route without hidden instructions',
        model: 'auto',
        systemPrompt: ''
      })
    })
  })

  it('treats whitespace-only system prompts as cleared state', async () => {
    const user = userEvent.setup()

    renderChatPage({ systemPromptButtonEnabled: true })

    const systemPromptButton = screen.getByRole('button', { name: 'System prompt' })
    await user.click(systemPromptButton)
    const dialog = screen.getByRole('dialog', { name: 'Set system prompt' })

    const wsTextarea = within(dialog).getByLabelText('System prompt')
    fireEvent.change(wsTextarea, { target: { value: '' } })

    await user.type(wsTextarea, '   {Enter}  ')
    await user.click(screen.getByRole('button', { name: 'Save prompt' }))

    await waitFor(() => {
      expect(systemPromptButton).toHaveFocus()
    })

    await user.click(systemPromptButton)
    expect(
      within(screen.getByRole('dialog', { name: 'Set system prompt' })).getByLabelText('System prompt')
    ).toHaveValue('')
    await user.click(screen.getByRole('button', { name: 'Cancel' }))

    await user.type(screen.getByLabelText('Prompt'), 'Send without a system prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls[0]).toMatchObject({
        content: 'Send without a system prompt',
        model: 'auto',
        systemPrompt: ''
      })
    })
  })

  it('keeps auto selected and sends auto with a single live model', async () => {
    const user = userEvent.setup()
    vi.mocked(adaptModelsToSummary).mockReturnValue(CHAT_HARNESS.models.slice(0, 1))

    renderChatPage({ mode: 'live' })

    expect(screen.getByRole('combobox', { name: 'Select model' })).toHaveTextContent('Auto (router picks best)')

    await user.type(screen.getByLabelText('Prompt'), 'Use the router')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls[0]).toMatchObject({ content: 'Use the router', model: 'auto' })
    })
  })

  it('shows conversation metadata as message count followed by localized timestamp', async () => {
    const yesterday = new Date()
    yesterday.setDate(yesterday.getDate() - 1)

    renderChatPage()

    expect(
      await screen.findByText(`4 messages · ${shortTimestamp(setLocalTime(new Date(), 9, 42))}`)
    ).toBeInTheDocument()
    expect(screen.getByText(`2 messages · ${shortTimestamp(yesterday)}`)).toBeInTheDocument()
  })

  it('renames and deletes conversations from the row action menu', async () => {
    const user = userEvent.setup()

    renderChatPage()

    await user.click(await screen.findByRole('button', { name: 'Open actions for Routing latency notes' }))
    await user.click(await screen.findByRole('menuitem', { name: /rename/i }))

    const renameInput = screen.getByLabelText('Rename Routing latency notes')
    expect(renameInput).toHaveFocus()
    await user.clear(renameInput)
    await user.type(renameInput, 'Renamed route audit')
    await user.click(screen.getByRole('button', { name: 'Save chat title' }))

    expect(await screen.findAllByText('Renamed route audit')).not.toHaveLength(0)
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.conversations[0]).toMatchObject({ id: 'c1', title: 'Renamed route audit' })
    })

    await user.click(screen.getByRole('button', { name: 'Open actions for Renamed route audit' }))
    await user.click(await screen.findByRole('menuitem', { name: /delete/i }))

    const deleteDialog = await screen.findByRole('alertdialog', { name: 'Delete "Renamed route audit"?' })
    expect(deleteDialog).toHaveTextContent('This permanently removes the selected chat and its message history')
    await user.click(screen.getByRole('button', { name: 'Delete chat' }))

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.conversations.some((conversation) => conversation.id === 'c1')).toBe(false)
      expect(latestState?.threads.c1).toBeUndefined()
      expect(latestState?.activeConversationId).toBe('c2')
    })
    expect(screen.queryByText('Renamed route audit')).not.toBeInTheDocument()
  })

  it('restores the persisted live conversation selection and thread after reload', async () => {
    vi.mocked(loadChatState).mockImplementation(async (scope) => {
      if (scope !== 'live') return undefined

      return {
        conversations: [
          { id: 'live-a', title: 'Live first', subtitle: '', updatedAt: 'Now' },
          { id: 'live-b', title: 'Live restored', subtitle: '', updatedAt: 'Later' }
        ],
        conversationGroups: [
          { title: 'Today', conversationIds: ['live-a', 'live-b'] },
          { title: 'Earlier', conversationIds: [] }
        ],
        threads: {
          'live-a': [{ id: 'msg-a', messageRole: 'assistant', timestamp: 'Now', body: 'Wrong live thread' }],
          'live-b': [{ id: 'msg-b', messageRole: 'assistant', timestamp: 'Later', body: 'Restored live thread body' }]
        },
        activeConversationId: 'live-b'
      }
    })

    renderChatPage({ mode: 'live' })

    expect(await screen.findAllByText(/Live restored/)).not.toHaveLength(0)
    expect(screen.getByText('Restored live thread body')).toBeInTheDocument()
    expect(screen.queryByText('Wrong live thread')).not.toBeInTheDocument()
  })

  it('does not show stale live lane messages while switching between persisted live conversations', async () => {
    const user = userEvent.setup()
    vi.mocked(loadChatState).mockImplementation(async (scope) => {
      if (scope !== 'live') return undefined

      return {
        conversations: [
          { id: 'live-a', title: 'Live first', subtitle: '', updatedAt: 'Now' },
          { id: 'live-b', title: 'Live restored', subtitle: '', updatedAt: 'Later' }
        ],
        conversationGroups: [
          { title: 'Today', conversationIds: ['live-a', 'live-b'] },
          { title: 'Earlier', conversationIds: [] }
        ],
        threads: {
          'live-a': [{ id: 'msg-a', messageRole: 'assistant', timestamp: 'Now', body: 'First persisted body' }],
          'live-b': [{ id: 'msg-b', messageRole: 'assistant', timestamp: 'Later', body: 'Restored persisted body' }]
        },
        activeConversationId: 'live-b'
      }
    })

    renderChatPage({ mode: 'live' })

    await waitFor(() => expect(screen.getByText('Restored persisted body')).toBeInTheDocument())

    await user.click((await screen.findAllByRole('button', { name: /Live first/i }))[0])

    expect(screen.queryByText('Restored persisted body')).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('First persisted body')).toBeInTheDocument())
  })

  it('creates a live thread on send, enables Stop while streaming, preserves partial text on stop, and retries with reload semantics', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    const retryButton = screen.getByRole('button', { name: 'Retry last' })
    expect(retryButton).toBeDisabled()

    await user.type(screen.getByLabelText('Prompt'), 'Hello from live mode')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(screen.getByLabelText('Prompt')).toHaveValue('')
    expect(screen.getByRole('button', { name: 'Stop' })).toBeInTheDocument()
    expect(screen.getByText(/Generating response/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop streaming' })).toHaveTextContent('Streaming response...')

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.activeConversationId).toBeTruthy()
      expect(chatMock.sendCalls).toHaveLength(1)
      expect(chatMock.sendCalls[0]?.content).toBe('Hello from live mode')
      expect(latestState?.conversations).toHaveLength(1)
      expect(latestState?.conversations[0]?.id).toBe(latestState?.activeConversationId)
      expect(chatMock.sendCalls[0]?.conversationId).toBe(latestState?.activeConversationId)
      expect(chatMock.hookConversationIds).toContain(latestState?.activeConversationId)
      expect(Object.keys(latestState?.threads ?? {})).toEqual([latestState?.activeConversationId])
      expect(latestState?.threads[latestState.activeConversationId].map((message) => message.body)).toEqual([
        'Hello from live mode',
        'Partial assistant reply'
      ])
    })

    expect(retryButton).toBeEnabled()

    await user.click(screen.getByRole('button', { name: 'Stop streaming' }))

    expect(chatMock.stopCalls).toHaveLength(1)
    expect(await screen.findByRole('button', { name: 'Send' })).toBeInTheDocument()
    expect(screen.getByText('Partial assistant reply')).toBeInTheDocument()
    expect(screen.getByText('(stopped)')).toBeInTheDocument()

    chatMock.reloadAssistantText = 'Retried assistant reply'
    chatMock.reloadErrorMessage = 'Retry failed after replacing the last assistant reply'

    await user.click(screen.getByRole('button', { name: 'Retry last' }))

    expect(await screen.findByText('Retried assistant reply')).toBeInTheDocument()
    expect(screen.queryByText('Partial assistant reply')).not.toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('Retry failed after replacing the last assistant reply')

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.threads[latestState.activeConversationId].map((message) => message.body)).toEqual([
        'Hello from live mode',
        'Retried assistant reply'
      ])
    })
  })

  it('renders streamed thinking separately, formats final markdown, and persists the raw assistant body', async () => {
    const user = userEvent.setup()
    const streamedBody = 'Reasoning text.</think> The capital of France is **Paris**.'
    chatMock.sendAssistantText = streamedBody

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Show final answer formatting')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByText('Thinking trace')).toBeInTheDocument()
    expect(screen.getByText('Reasoning text.')).toBeInTheDocument()

    const paris = screen.getByText('Paris')
    expect(paris.tagName.toLowerCase()).toBe('strong')
    expect(paris.closest('.select-text')).toHaveTextContent('The capital of France is Paris.')
    expect(screen.getByRole('button', { name: 'Stop streaming' })).toHaveTextContent('Streaming response...')

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.threads[latestState.activeConversationId].map((message) => message.body)).toEqual([
        'Show final answer formatting',
        streamedBody
      ])
    })
  })

  it('keeps live streaming mounted and marked in the sidebar when the chat route unmounts', async () => {
    const user = userEvent.setup()
    const { rerender } = renderPersistentChatRoute(true)

    await user.type(screen.getByLabelText('Prompt'), 'Continue this while I leave')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByText('Partial assistant reply')).toBeInTheDocument()
    expect(screen.getByLabelText('Generating response')).toBeInTheDocument()

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.activeConversationId).toBeTruthy()
      expect(latestState?.threads[latestState.activeConversationId].map((message) => message.body)).toEqual([
        'Continue this while I leave',
        'Partial assistant reply'
      ])
    })

    const streamingConversationId = vi.mocked(saveChatState).mock.calls.at(-1)?.[1].activeConversationId
    if (!streamingConversationId) throw new Error('Expected streaming conversation id')
    const hookUnmountsBeforeRouteChange = chatMock.hookUnmounts.length

    rerender(
      <FeatureFlagProvider>
        <DataModeProvider initialMode="live" persist={false}>
          <ChatSessionProvider>
            <div data-testid="network-route">Network route</div>
          </ChatSessionProvider>
        </DataModeProvider>
      </FeatureFlagProvider>
    )

    expect(screen.getByTestId('network-route')).toBeInTheDocument()
    expect(chatMock.stopCalls).toHaveLength(0)
    expect(chatMock.hookUnmounts).toHaveLength(hookUnmountsBeforeRouteChange)

    rerender(
      <FeatureFlagProvider>
        <DataModeProvider initialMode="live" persist={false}>
          <ChatSessionProvider>
            <ChatPageContent />
          </ChatSessionProvider>
        </DataModeProvider>
      </FeatureFlagProvider>
    )

    expect(await screen.findByText('Partial assistant reply')).toBeInTheDocument()
    expect(screen.getByLabelText('Generating response')).toBeInTheDocument()
    expect(chatMock.hookConversationIds).toContain(streamingConversationId)
  })

  it('keeps a streaming conversation active when selecting another chat in the sidebar', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Write a long story')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByText('Partial assistant reply')).toBeInTheDocument()
    expect(screen.getByLabelText('Generating response')).toBeInTheDocument()

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.activeConversationId).toBeTruthy()
      expect(latestState?.threads[latestState.activeConversationId].map((message) => message.body)).toEqual([
        'Write a long story',
        'Partial assistant reply'
      ])
    })

    const streamingConversationId = vi.mocked(saveChatState).mock.calls.at(-1)?.[1].activeConversationId
    if (!streamingConversationId) throw new Error('Expected streaming conversation id')
    const hookUnmountsBeforeSwitch = chatMock.hookUnmounts.length

    await user.click(screen.getByRole('button', { name: 'New' }))

    expect(chatMock.stopCalls).toHaveLength(0)
    expect(chatMock.hookUnmounts.slice(hookUnmountsBeforeSwitch)).not.toContain(streamingConversationId)
    expect(screen.getByLabelText('Generating response')).toBeInTheDocument()
    expect(screen.getByText('Start with a clean routing context')).toBeInTheDocument()
    expect(screen.queryByText('Partial assistant reply')).not.toBeInTheDocument()

    await user.click(screen.getAllByRole('button', { name: /Write a long story/i })[0])

    expect(await screen.findByText('Partial assistant reply')).toBeInTheDocument()
    expect(screen.getByLabelText('Generating response')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop streaming' })).toHaveTextContent('Streaming response...')
    expect(chatMock.stopCalls).toHaveLength(0)
    expect(chatMock.hookUnmounts.slice(hookUnmountsBeforeSwitch)).not.toContain(streamingConversationId)
    expect(chatMock.hookConversationIds).toContain(streamingConversationId)
  })

  it('sends from the newly selected live conversation while another conversation is streaming', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First streaming prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByText('Partial assistant reply')).toBeInTheDocument()

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      const activeConversationId = latestState?.activeConversationId
      expect(activeConversationId).toBeTruthy()
      expect(chatMock.sendCalls).toHaveLength(1)
      expect(chatMock.sendCalls[0]?.conversationId).toBe(activeConversationId)
      expect(latestState?.threads[activeConversationId ?? ''].map((message) => message.body)).toEqual([
        'First streaming prompt',
        'Partial assistant reply'
      ])
    })

    const streamingConversationId = vi.mocked(saveChatState).mock.calls.at(-1)?.[1].activeConversationId
    if (!streamingConversationId) throw new Error('Expected streaming conversation id')
    const hookUnmountsBeforeNewChat = chatMock.hookUnmounts.length

    await user.click(screen.getByRole('button', { name: 'New' }))

    let newConversationId = ''
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      const activeConversationId = latestState?.activeConversationId ?? ''
      expect(activeConversationId).toBeTruthy()
      expect(activeConversationId).not.toBe(streamingConversationId)
      expect(latestState?.threads[activeConversationId]).toEqual([])
      newConversationId = activeConversationId
    })

    expect(chatMock.stopCalls).toHaveLength(0)
    expect(chatMock.hookUnmounts.slice(hookUnmountsBeforeNewChat)).not.toContain(streamingConversationId)
    expect(screen.getByRole('button', { name: 'Send' })).toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Second prompt for the new chat')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(2)
      expect(chatMock.sendCalls[1]?.conversationId).toBe(newConversationId)
      expect(chatMock.sendCalls[1]?.conversationId).not.toBe(streamingConversationId)
      expect(chatMock.sendCalls[1]?.content).toBe('Second prompt for the new chat')
    })
  })

  it('does not retarget either hidden lane when creating a third chat while two chats stream', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First stream stays alive')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByText('Partial assistant reply')).toBeInTheDocument()

    let firstStreamingConversationId = ''
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      firstStreamingConversationId = latestState?.activeConversationId ?? ''
      expect(firstStreamingConversationId).toBeTruthy()
      expect(chatMock.sendCalls).toHaveLength(1)
      expect(chatMock.sendCalls[0]?.conversationId).toBe(firstStreamingConversationId)
    })

    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.type(screen.getByLabelText('Prompt'), 'Second stream also stays alive')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    let secondStreamingConversationId = ''
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      secondStreamingConversationId = latestState?.activeConversationId ?? ''
      expect(secondStreamingConversationId).toBeTruthy()
      expect(secondStreamingConversationId).not.toBe(firstStreamingConversationId)
      expect(chatMock.sendCalls).toHaveLength(2)
      expect(chatMock.sendCalls[1]?.conversationId).toBe(secondStreamingConversationId)
    })

    const hookUnmountsBeforeThirdChat = chatMock.hookUnmounts.length

    await user.click(screen.getByRole('button', { name: 'New' }))

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.activeConversationId).toBeTruthy()
      expect(latestState?.activeConversationId).not.toBe(firstStreamingConversationId)
      expect(latestState?.activeConversationId).not.toBe(secondStreamingConversationId)
    })

    expect(chatMock.stopCalls).toHaveLength(0)
    expect(chatMock.hookUnmounts.slice(hookUnmountsBeforeThirdChat)).not.toContain(firstStreamingConversationId)
    expect(chatMock.hookUnmounts.slice(hookUnmountsBeforeThirdChat)).not.toContain(secondStreamingConversationId)
    expect(screen.getByText('Start with a clean routing context')).toBeInTheDocument()
  })

  it('keeps the third chat composer draft isolated while both live lanes stream', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First stream stays alive')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByText('Partial assistant reply')).toBeInTheDocument()

    let firstStreamingConversationId = ''
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      firstStreamingConversationId = latestState?.activeConversationId ?? ''
      expect(firstStreamingConversationId).toBeTruthy()
      expect(chatMock.sendCalls).toHaveLength(1)
    })

    await user.click(screen.getByRole('button', { name: 'New' }))
    await user.type(screen.getByLabelText('Prompt'), 'Second stream also stays alive')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    let secondStreamingConversationId = ''
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      secondStreamingConversationId = latestState?.activeConversationId ?? ''
      expect(secondStreamingConversationId).toBeTruthy()
      expect(secondStreamingConversationId).not.toBe(firstStreamingConversationId)
      expect(chatMock.sendCalls).toHaveLength(2)
    })

    await user.click(screen.getByRole('button', { name: 'New' }))

    let thirdConversationId = ''
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      thirdConversationId = latestState?.activeConversationId ?? ''
      expect(thirdConversationId).toBeTruthy()
      expect(thirdConversationId).not.toBe(firstStreamingConversationId)
      expect(thirdConversationId).not.toBe(secondStreamingConversationId)
    })

    const thirdDraft = 'Draft belongs only to the third chat'
    await user.type(screen.getByLabelText('Prompt'), thirdDraft)

    expect(screen.getByLabelText('Prompt')).toHaveValue(thirdDraft)
    expect(screen.getByRole('button', { name: 'Queue' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Stop' })).not.toBeInTheDocument()

    await user.click(screen.getAllByRole('button', { name: /First stream stays alive/i })[0])

    expect(screen.getByLabelText('Prompt')).not.toHaveValue(thirdDraft)
    expect(screen.getByRole('button', { name: 'Stop' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /^New chat/i }))

    expect(screen.getByLabelText('Prompt')).toHaveValue(thirdDraft)
    expect(screen.getByRole('button', { name: 'Queue' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Queue' }))

    expect(chatMock.sendCalls).toHaveLength(2)
    expect(screen.getByText(thirdDraft)).toBeInTheDocument()

    await user.click(screen.getAllByRole('button', { name: /First stream stays alive/i })[0])

    expect(screen.queryByText(thirdDraft)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Stop' }))

    expect(chatMock.sendCalls).toHaveLength(2)

    await user.click(screen.getByRole('button', { name: /^New chat/i }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(3)
      expect(chatMock.sendCalls[2]?.conversationId).toBe(thirdConversationId)
      expect(chatMock.sendCalls[2]?.content).toBe(thirdDraft)
    })
  })

  it('does not clear a hidden streaming conversation when deleting another selected chat', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Write a long story')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByText('Partial assistant reply')).toBeInTheDocument()

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.activeConversationId).toBeTruthy()
    })

    const streamingConversationId = vi.mocked(saveChatState).mock.calls.at(-1)?.[1].activeConversationId
    if (!streamingConversationId) throw new Error('Expected streaming conversation id')
    const hookUnmountsBeforeDelete = chatMock.hookUnmounts.length

    await user.click(screen.getByRole('button', { name: 'New' }))
    expect(screen.getByText('Start with a clean routing context')).toBeInTheDocument()
    expect(screen.queryByText('Partial assistant reply')).not.toBeInTheDocument()

    await user.click(await screen.findByRole('button', { name: 'Open actions for New chat' }))
    await user.click(await screen.findByRole('menuitem', { name: /delete/i }))
    await user.click(await screen.findByRole('button', { name: 'Delete chat' }))

    expect(chatMock.stopCalls).toHaveLength(0)
    expect(chatMock.hookUnmounts.slice(hookUnmountsBeforeDelete)).not.toContain(streamingConversationId)
    expect(await screen.findByText('Partial assistant reply')).toBeInTheDocument()
    expect(screen.getByLabelText('Generating response')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop streaming' })).toHaveTextContent('Streaming response...')

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.conversations).toHaveLength(1)
      expect(latestState?.activeConversationId).toBe(streamingConversationId)
      expect(latestState?.threads[streamingConversationId].map((message) => message.body)).toEqual([
        'Write a long story',
        'Partial assistant reply'
      ])
    })
  })

  it('persists an empty stopped assistant turn when a stream is stopped before tokens arrive', async () => {
    const user = userEvent.setup()
    chatMock.sendAssistantText = ''

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Stop before any token')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(screen.getByRole('button', { name: 'Stop streaming' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Stop streaming' }))

    expect(chatMock.stopCalls).toHaveLength(1)
    expect(await screen.findByText('(stopped)')).toBeInTheDocument()

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      expect(latestState?.activeConversationId).toBeTruthy()
      expect(latestState?.threads[latestState.activeConversationId].map((message) => message.body)).toEqual([
        'Stop before any token',
        ''
      ])
    })
  })

  it('keeps newly inserted chat messages pinned fully into view at the bottom', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    const messageList = screen.getByTestId('chat-message-list')
    Object.defineProperty(messageList, 'scrollHeight', { configurable: true, value: 1400 })
    Object.defineProperty(messageList, 'clientHeight', { configurable: true, value: 420 })

    await user.type(screen.getByLabelText('Prompt'), 'Follow the latest message')
    scrollIntoViewMock.mockClear()

    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByText('Partial assistant reply')).toBeInTheDocument()
    await waitFor(() => expect(scrollIntoViewMock).toHaveBeenCalledWith({ block: 'end' }))

    const scrollTarget = scrollIntoViewMock.mock.contexts.at(-1) as HTMLElement | undefined
    expect(scrollTarget).toHaveAttribute('data-chat-scroll-anchor', 'true')
    expect(messageList.scrollTop).toBe(1400)
  })

  it('renders and persists completed response metadata on live assistant messages', async () => {
    const user = userEvent.setup()
    chatMock.sendAssistantText = 'Response with measured metadata'
    chatMock.sendResponseMetadata = {
      model: 'unsloth/MiniMax-M2.5-GGUF:Q4_K_M',
      usage: { input_tokens: 9, output_tokens: 27, total_tokens: 36 },
      timings: { decode_time_ms: 1765, ttft_ms: 1116, total_time_ms: 2881 },
      servedBy: 'lemony-28'
    }

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Measure this response')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    const userHeader = screen.getByText('You').parentElement

    expect(userHeader).toHaveTextContent(chatMock.sendCalls[0]?.model ?? '')
    expect(userHeader).not.toHaveTextContent('2026-05-06')
    expect(await screen.findByText('Response with measured metadata')).toBeInTheDocument()
    expect(screen.getByText('unsloth/MiniMax-M2.5-GGUF:Q4_K_M')).toBeInTheDocument()
    expect(await screen.findByText((_, element) => element?.textContent === '27 tok')).toBeInTheDocument()
    expect(await screen.findByText((_, element) => element?.textContent === '15.3 tok/s')).toBeInTheDocument()
    expect(await screen.findByText((_, element) => element?.textContent === 'TTFT 1116ms')).toBeInTheDocument()

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      const activeThread = latestState?.activeConversationId
        ? latestState.threads[latestState.activeConversationId]
        : undefined
      expect(activeThread?.at(-1)).toMatchObject({
        id: 'assistant-1',
        messageRole: 'assistant',
        model: 'unsloth/MiniMax-M2.5-GGUF:Q4_K_M',
        route: 'lemony-28',
        routeNode: 'lemony-28',
        tokens: '27 tok',
        tokPerSec: '15.3 tok/s',
        ttft: '1116ms'
      })
    })
  })

  it('shows the streaming placeholder and drains a queued prompt with the latest selected model', async () => {
    const user = userEvent.setup()
    chatMock.sendAssistantText = ''

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First live prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByText('Streaming response...')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Queue' })).toBeDisabled()

    await user.type(screen.getByLabelText('Prompt'), 'Run this next')
    await user.click(screen.getByRole('button', { name: 'Queue' }))

    expect(screen.getByLabelText('Prompt')).toHaveValue('')
    expect(screen.getByText('Run this next')).toBeInTheDocument()
    expect(screen.getByText('Queued')).toBeInTheDocument()

    await user.click(screen.getByRole('combobox', { name: 'Select model' }))
    await user.click(await screen.findByText('Qwen3.5-0.8B-UD'))
    await user.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(2)
      expect(chatMock.sendCalls[1]?.content).toBe('Run this next')
      expect(chatMock.sendCalls[1]?.model).toBe('Qwen3.5-0.8B-UD')
    })

    expect(screen.queryByText('Queued')).not.toBeInTheDocument()
    expect(screen.getAllByText('Qwen3.5-0.8B-UD')).not.toHaveLength(0)
  })

  it('removes a queued prompt before the stream drains it', async () => {
    const user = userEvent.setup()
    chatMock.sendAssistantText = ''

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First live prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    expect(await screen.findByText('Streaming response...')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Do not send this')
    await user.click(screen.getByRole('button', { name: 'Queue' }))

    expect(screen.getByText('Do not send this')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Remove queued message' }))

    expect(screen.queryByText('Do not send this')).not.toBeInTheDocument()
    expect(screen.queryByText('Queued')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(1)
      expect(chatMock.sendCalls[0]?.content).toBe('First live prompt')
    })
  })

  it('keeps multiple queued prompts visible and removes only the selected queued item', async () => {
    const user = userEvent.setup()
    chatMock.sendAssistantText = ''

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First live prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    expect(await screen.findByText('Streaming response...')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Queued alpha')
    await user.click(screen.getByRole('button', { name: 'Queue' }))
    await user.type(screen.getByLabelText('Prompt'), 'Queued beta')
    await user.click(screen.getByRole('button', { name: 'Queue' }))

    expect(screen.getByText('Queued alpha')).toBeInTheDocument()
    expect(screen.getByText('Queued beta')).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: 'Remove queued message' })).toHaveLength(2)

    await user.click(screen.getAllByRole('button', { name: 'Remove queued message' })[0])

    expect(screen.queryByText('Queued alpha')).not.toBeInTheDocument()
    expect(screen.getByText('Queued beta')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(2)
      expect(chatMock.sendCalls[1]?.content).toBe('Queued beta')
    })
    expect(chatMock.sendCalls.map((call) => call.content)).not.toContain('Queued alpha')
  })

  it('drains multiple queued prompts one at a time in FIFO order', async () => {
    const user = userEvent.setup()
    chatMock.sendAssistantText = ''

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'First live prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))
    expect(await screen.findByText('Streaming response...')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Queued alpha')
    await user.click(screen.getByRole('button', { name: 'Queue' }))
    await user.type(screen.getByLabelText('Prompt'), 'Queued beta')
    await user.click(screen.getByRole('button', { name: 'Queue' }))

    await user.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(2)
      expect(chatMock.sendCalls[1]?.content).toBe('Queued alpha')
    })
    expect(screen.getAllByRole('button', { name: 'Remove queued message' })).toHaveLength(1)
    expect(screen.getByText('Queued beta')).toBeInTheDocument()
    expect(chatMock.sendCalls.map((call) => call.content)).not.toContain('Queued beta')

    await user.click(screen.getByRole('button', { name: 'Stop' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(3)
      expect(chatMock.sendCalls[2]?.content).toBe('Queued beta')
    })
    expect(screen.queryByText('Queued')).not.toBeInTheDocument()
  })

  it('persists submitted live message model labels with the conversation thread', async () => {
    const user = userEvent.setup()
    const submittedModel = 'auto'

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Persist the submitted model')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      const activeConversationId = latestState?.activeConversationId
      expect(activeConversationId).toBeTruthy()
      expect(latestState?.threads[activeConversationId ?? '']).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ body: 'Persist the submitted model', messageRole: 'user', model: submittedModel }),
          expect.objectContaining({ body: 'Partial assistant reply', messageRole: 'assistant', model: submittedModel })
        ])
      )
    })
  })

  it('hides the transparency tab by default', () => {
    renderChatPage()

    expect(screen.queryByRole('tab', { name: /transparency/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Inspect transparency' })).not.toBeInTheDocument()
  })

  it('opens the responsive chat sidebar popover from the floating control', async () => {
    const user = userEvent.setup()

    render(
      <ChatLayout
        actions={null}
        composer={<textarea aria-label="Prompt" />}
        sidebar={<div role="tablist" aria-label="Chat sidebar views" />}
        sidebarMode="compact"
        title="Chat"
      >
        <div data-testid="message-content">Messages</div>
      </ChatLayout>
    )

    expect(screen.queryByRole('tablist', { name: 'Chat sidebar views' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Open chat sidebar' }))

    expect(screen.getAllByRole('tablist', { name: 'Chat sidebar views' })).toHaveLength(1)

    await user.keyboard('{Escape}')

    expect(screen.queryByRole('tablist', { name: 'Chat sidebar views' })).not.toBeInTheDocument()
  })

  it('hides route disclosure text by default', () => {
    renderChatPage()

    expect(queryAllByTextContent('sent to lemony-28')).toHaveLength(0)
    expect(queryAllByTextContent('sent to carrack')).toHaveLength(0)
    expect(queryAllByTextContent('routed via carrack')).toHaveLength(0)
  })

  it('shows the transparency tab when the feature flag is enabled', () => {
    renderChatPage({ transparencyTabEnabled: true })

    expect(screen.getByRole('tab', { name: /transparency/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Inspect transparency' })).toBeInTheDocument()
    expect(queryAllByTextContent('sent to lemony-28').length).toBeGreaterThan(0)
    expect(queryAllByTextContent('sent to carrack').length).toBeGreaterThan(0)
    expect(queryAllByTextContent('routed via carrack').length).toBeGreaterThan(0)
  })

  it('creates and selects a new empty live conversation without copying previous messages', async () => {
    const user = userEvent.setup()
    chatMock.sendStatus = 'ready'

    renderChatPage({ mode: 'live' })

    await user.type(screen.getByLabelText('Prompt'), 'Hello!')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByText('Partial assistant reply')).toBeInTheDocument()
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      const activeConversationId = latestState?.activeConversationId
      expect(activeConversationId).toBeTruthy()
      expect(latestState?.threads[activeConversationId ?? ''].map((message) => message.body)).toEqual([
        'Hello!',
        'Partial assistant reply'
      ])
    })

    await user.click(screen.getByRole('button', { name: /new/i }))

    expect(await screen.findAllByText(/New chat/)).not.toHaveLength(0)
    expect(await screen.findByText('Start with a clean routing context')).toBeInTheDocument()
    expect(screen.queryByText('Partial assistant reply')).not.toBeInTheDocument()
    expect(screen.getByText(`0 messages · ${shortTimestamp(new Date())}`)).toBeInTheDocument()
    await waitFor(() => expect(screen.getByLabelText('Prompt')).toHaveFocus())
    await waitFor(() => {
      const latestState = vi.mocked(saveChatState).mock.calls.at(-1)?.[1]
      const activeConversationId = latestState?.activeConversationId
      expect(activeConversationId).toBeTruthy()
      expect(latestState?.threads[activeConversationId ?? '']).toEqual([])
    })
  })

  it('opens the hidden file picker from Attach', async () => {
    const user = userEvent.setup()
    const clickSpy = vi.spyOn(HTMLInputElement.prototype, 'click')

    renderChatPage({ mode: 'live' })

    await user.click(screen.getByRole('button', { name: 'Attach' }))

    expect(clickSpy).toHaveBeenCalled()
  })

  it('sends legacy-compatible attachment content for image and scanned pdf attachments', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    const picker = document.querySelector('input[type="file"]') as HTMLInputElement
    const image = new File(['image-bytes'], 'cat.png', { type: 'image/png' })
    const pdf = new File(['pdf-bytes'], 'scan.pdf', { type: 'application/pdf' })

    await user.upload(picker, [image, pdf])
    expect(screen.getByText('2 attachments ready')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'Summarize these')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(1)
    })

    const content = chatMock.sendCalls[0]?.content
    expect(typeof content).not.toBe('string')
    expect((content as MultimodalContent).content).toEqual([
      { type: 'text', content: 'Summarize these' },
      { type: 'text', content: '[Image description: A tabby cat]' },
      { type: 'text', content: '[Content from scan.pdf]\n\n[Page 1]\n[Image description: A scanned receipt]' }
    ])
    expect(attachmentPreprocessingMock.describeImageForPrompt).toHaveBeenCalledTimes(1)
    expect(attachmentPreprocessingMock.extractPdfTextFromFile).toHaveBeenCalledWith(pdf)
    expect(attachmentPreprocessingMock.describeScannedPdf).toHaveBeenCalledWith(pdf, expect.any(Function))
  })

  it('shows submitted attachment chips on the user message and opens an image preview', async () => {
    const user = userEvent.setup()

    renderChatPage({ mode: 'live' })

    const picker = document.querySelector('input[type="file"]') as HTMLInputElement
    const image = new File(['image-bytes'], 'cat.png', { type: 'image/png' })

    await user.upload(picker, image)
    await user.type(screen.getByLabelText('Prompt'), 'Describe this image to me')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(screen.getByText('Describe this image to me')).toBeInTheDocument()
    })

    await waitFor(() => expect(createObjectUrlMock).toHaveBeenCalledWith(image))
    const chip = await screen.findByRole('button', { name: 'Open cat.png' })
    expect(chip).toHaveTextContent('Image 1')

    await user.click(chip)

    const dialog = await screen.findByRole('dialog', { name: /cat\.png/i })
    expect(within(dialog).getByText('Image 1 · image/png')).toBeInTheDocument()
    expect(within(dialog).getByRole('img', { name: 'cat.png' })).toHaveAttribute('src', 'blob:preview/cat.png')
  })

  it('shows staged attachment preparation feedback before the chat prompt is submitted', async () => {
    const user = userEvent.setup()
    const imageDescription = createDeferred<{ imageDescription?: string }>()
    attachmentPreprocessingMock.describeImageForPrompt.mockImplementation(
      async (_dataUrl: string, onStage?: (stage: TestAttachmentProcessingStage) => void) => {
        onStage?.('starting')
        await Promise.resolve()
        onStage?.('processing')
        return imageDescription.promise
      }
    )

    renderChatPage({ mode: 'live' })

    const picker = document.querySelector('input[type="file"]') as HTMLInputElement
    const image = new File(['image-bytes'], 'slow.png', { type: 'image/png' })

    await user.upload(picker, image)
    await user.type(screen.getByLabelText('Prompt'), 'What is in this image?')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    const status = await screen.findByLabelText('Attachment preparation status')
    expect(status).toHaveTextContent('Preparing attachments')
    expect(status).toHaveTextContent('Downloading')
    expect(status).toHaveTextContent('Starting')
    expect(status).toHaveTextContent('Processing')
    expect(status).toHaveTextContent('Prompt waiting: What is in this image?')
    await waitFor(() => expect(status).toHaveTextContent('Processing attachment content'))
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
    expect(screen.getByText('Processing attachments…')).toBeInTheDocument()
    expect(chatMock.sendCalls).toHaveLength(0)

    imageDescription.resolve({ imageDescription: '[Image description: A slow diagram]' })

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(1)
    })
    expect(screen.queryByLabelText('Attachment preparation status')).not.toBeInTheDocument()
  })

  it('reuses the loaded browser analyzer state for later attachment submissions', async () => {
    const user = userEvent.setup()
    const imageDescription = createDeferred<{ imageDescription?: string }>()
    attachmentPreprocessingMock.isBrowserVisionModelLoaded.mockReturnValue(true)
    attachmentPreprocessingMock.describeImageForPrompt.mockImplementation(
      async (_dataUrl: string, onStage?: (stage: TestAttachmentProcessingStage) => void) => {
        onStage?.('processing')
        return imageDescription.promise
      }
    )

    renderChatPage({ mode: 'live' })

    const picker = document.querySelector('input[type="file"]') as HTMLInputElement
    const image = new File(['image-bytes'], 'cached.png', { type: 'image/png' })

    await user.upload(picker, image)
    await user.type(screen.getByLabelText('Prompt'), 'Use the cached analyzer')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    const status = await screen.findByLabelText('Attachment preparation status')
    expect(status).toHaveTextContent('Processing attachment content')
    expect(status).toHaveTextContent('Cached')
    expect(status).toHaveTextContent('Reusing the browser analyzer already loaded in this tab.')
    expect(status).toHaveTextContent('Ready')
    expect(status).toHaveTextContent('The local vision and document pipeline is already warm.')
    expect(status).not.toHaveTextContent('Downloading browser model')
    expect(status).not.toHaveTextContent('Fetching the browser analyzer and attachment assets.')
    expect(status).not.toHaveTextContent('Warming the local vision and document pipeline.')
    expect(chatMock.sendCalls).toHaveLength(0)

    imageDescription.resolve({ imageDescription: '[Image description: Cached run]' })

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(1)
    })
  })

  it('preserves the prompt and queued attachment when upload fails before send completes', async () => {
    const user = userEvent.setup()
    chatMock.sendErrorMessage = 'Upload failed: 503'
    chatMock.sendErrorResolves = true
    chatMock.sendStatus = 'submitted'
    chatMock.sendOptimisticStatusBeforeError = true
    chatMock.sendOptimisticUserMessageBeforeError = true
    chatMock.sendOptimisticAssistantPlaceholderBeforeError = true

    renderChatPage({ mode: 'live' })

    const picker = document.querySelector('input[type="file"]') as HTMLInputElement
    const audio = new File(['audio-bytes'], 'clip.mp3', { type: 'audio/mpeg' })

    await user.upload(picker, audio)
    await user.type(screen.getByLabelText('Prompt'), 'Keep this prompt')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Upload failed: 503')
    await waitFor(() => expect(createObjectUrlMock).toHaveBeenCalledWith(audio))
    await waitFor(() => expect(revokeObjectUrlMock).toHaveBeenCalledWith('blob:preview/clip.mp3'))
    expect(screen.getByTestId('chat-message-list')).toHaveTextContent('Keep this prompt')
    await waitFor(() => {
      expect(screen.getByLabelText('Prompt')).toHaveValue('Keep this prompt')
    })
    expect(chatMock.sendCalls).toHaveLength(1)

    chatMock.sendErrorMessage = undefined
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(2)
    })

    const retriedContent = chatMock.sendCalls[1]?.content
    expect(typeof retriedContent).not.toBe('string')
    expect((retriedContent as MultimodalContent).content).toEqual([
      { type: 'text', content: 'Keep this prompt' },
      {
        type: 'audio',
        source: { type: 'data', value: 'YXVkaW8tYnl0ZXM=', mimeType: 'audio/mpeg' },
        metadata: { fileName: 'clip.mp3' }
      }
    ])
  })

  it('restores the prompt and attachment when a send request fails generically', async () => {
    const user = userEvent.setup()
    chatMock.sendErrorMessage = 'Network failed'

    renderChatPage({ mode: 'live' })

    const picker = document.querySelector('input[type="file"]') as HTMLInputElement
    const image = new File(['image-bytes'], 'diagram.png', { type: 'image/png' })

    await user.upload(picker, image)
    await user.type(screen.getByLabelText('Prompt'), 'Keep this generic failure draft')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Network failed')
    expect(screen.getByTestId('chat-message-list')).toHaveTextContent('Keep this generic failure draft')
    await waitFor(() => {
      expect(screen.getByLabelText('Prompt')).toHaveValue('Keep this generic failure draft')
    })
    expect(chatMock.sendCalls).toHaveLength(1)

    chatMock.sendErrorMessage = undefined
    await user.click(screen.getByRole('button', { name: 'Send' }))

    await waitFor(() => {
      expect(chatMock.sendCalls).toHaveLength(2)
    })
    const retriedContent = chatMock.sendCalls[1]?.content
    expect(typeof retriedContent).not.toBe('string')
    expect((retriedContent as MultimodalContent).content).toEqual([
      { type: 'text', content: 'Keep this generic failure draft' },
      { type: 'text', content: '[Image description: A tabby cat]' }
    ])
  })
})
