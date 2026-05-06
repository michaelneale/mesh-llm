import { useCallback, useMemo, useRef, useState } from 'react'
import { Cpu, HardDrive } from 'lucide-react'
import { LoadingGhostBlock } from '@/components/ui/LoadingGhostBlock'
import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { LiveRefreshPill } from '@/components/ui/LiveRefreshPill'
import { useLoadingGhostShimmer } from '@/components/ui/useLoadingGhostShimmer'
import { ChatSidebar } from '@/features/chat/components/ChatSidebar'
import { Composer } from '@/features/chat/components/Composer'
import { MessageRow } from '@/features/chat/components/MessageRow'
import { ModelSelect } from '@/features/chat/components/ModelSelect'
import { TransparencyPane } from '@/features/chat/components/transparency/TransparencyPane'
import { ChatLayout } from '@/features/chat/layouts/ChatLayout'
import { useMeshChat } from '@/features/chat/api/use-chat'
import { useChatMessages } from '@/features/chat/api/use-chat-messages'
import { useConversations } from '@/features/chat/api/use-conversations'
import { useModelsQuery } from '@/features/network/api/use-models-query'
import { adaptModelsToSummary } from '@/features/network/api/models-adapter'
import { useDataMode } from '@/lib/data-mode'
import { useBooleanFeatureFlag } from '@/lib/feature-flags'
import { QueryProvider } from '@/lib/query/QueryProvider'
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

function ChatLiveLoadingGhost() {
  const rootRef = useRef<HTMLDivElement | null>(null)
  useLoadingGhostShimmer(rootRef)
  const conversations = ['conv-a', 'conv-b', 'conv-c', 'conv-d']
  const messages = ['msg-a', 'msg-b', 'msg-c', 'msg-d', 'msg-e']

  const sidebar = (
    <aside className="panel-shell flex min-h-0 flex-col overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
      <div className="border-b border-border-soft px-3.5 py-2.5">
        <LoadingGhostBlock className="h-4 w-32" shimmer />
        <LoadingGhostBlock className="mt-2 h-3 w-44" shimmer />
      </div>
      <div className="space-y-2 p-3">
        {conversations.map((conversation) => (
          <LoadingGhostBlock key={conversation} className="h-14" shimmer />
        ))}
      </div>
    </aside>
  )

  return (
    <div ref={rootRef}>
      <ChatLayout
        sidebar={sidebar}
        title="Live chat"
        subtitle="Connecting to the backend model catalog"
        actions={
          <>
            <LoadingGhostBlock className="h-6 w-20 rounded-full" shimmer />
            <LoadingGhostBlock className="h-8 w-44" shimmer />
          </>
        }
        composer={<LoadingGhostBlock className="h-11" shimmer />}
      >
        <div className="space-y-4">
          {messages.map((message, index) => (
            <div key={message} className={index % 2 === 0 ? 'mr-auto max-w-[72%]' : 'ml-auto max-w-[68%]'}>
              <LoadingGhostBlock className="h-4 w-32" shimmer />
              <LoadingGhostBlock className="mt-2 h-16" shimmer />
            </div>
          ))}
        </div>
      </ChatLayout>
    </div>
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

export function ChatPageContent({ data = CHAT_HARNESS }: ChatPageProps) {
  const { mode, setMode } = useDataMode()
  const liveMode = mode === 'live'
  const modelsQuery = useModelsQuery({ enabled: mode === 'live' })
  const liveModels = modelsQuery.data ? adaptModelsToSummary(modelsQuery.data.mesh_models) : undefined
  const resolvedModels = liveMode ? liveModels : data.models
  const displayModels = resolvedModels ?? data.models
  const showLiveError = liveMode && !liveModels && !modelsQuery.isFetching && modelsQuery.isError
  const showLiveLoading = liveMode && !liveModels && !showLiveError
  const transparencyTabEnabled = useBooleanFeatureFlag('chat/transparencyTab')
  const showLiveRefresh = liveMode && Boolean(liveModels) && modelsQuery.isFetching
  const conversationFallback = useMemo<ChatHarnessData>(() => {
    if (!liveMode) return data
    return { ...data, conversations: [], conversationGroups: [], threads: {} }
  }, [data, liveMode])

  const [activeConversationId, setActiveConversationId] = useState(data.conversations[0]?.id ?? '')
  const [sidebarTab, setSidebarTab] = useState<'conversations' | 'transparency'>('conversations')
  const [inspectedMessage, setInspectedMessage] = useState<TransparencyMessage | undefined>()
  const [prompt, setPrompt] = useState('')
  const [model, setModel] = useState(data.models[0]?.name ?? '')
  const activeModelName = displayModels.some((item) => item.name === model) ? model : (displayModels[0]?.name ?? '')

  const conversations = useConversations(conversationFallback)
  const activeConversation =
    conversations.conversations.find((c) => c.id === activeConversationId) ?? conversations.conversations[0]
  const activeConversationKey = activeConversation?.id ?? ''
  const chat = useMeshChat(activeModelName)
  const liveMessages = useChatMessages(chat.messages)
  const resolvedThreads =
    liveMessages.length > 0
      ? { ...conversations.threads, [activeConversationKey]: liveMessages }
      : conversations.threads

  const options: ModelSelectOption[] = displayModels.map((item) => ({
    value: item.name,
    label: item.name,
    meta: `${item.family} · ${item.context}`,
    status: modelStatusBadge(item)
  }))
  const activeMessages = resolvedThreads[activeConversationKey] ?? []

  const inspectMessage = (message: TransparencyMessage) => {
    if (!transparencyTabEnabled) return

    setInspectedMessage(message)
    setSidebarTab('transparency')
  }
  const selectConversation = (conversation: Conversation) => {
    setActiveConversationId(conversation.id)
    setInspectedMessage(undefined)
    setSidebarTab('conversations')
  }
  const retryLiveData = useCallback(() => {
    void modelsQuery.refetch()
  }, [modelsQuery])
  const switchToTestData = useCallback(() => setMode('harness'), [setMode])

  const sidebar = (
    <ChatSidebar
      tab={sidebarTab}
      onTabChange={setSidebarTab}
      conversations={conversations.conversations}
      conversationGroups={conversations.conversationGroups}
      activeId={activeConversation?.id}
      onSelectConversation={selectConversation}
      onNewChat={() => setPrompt('')}
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
      {showLiveRefresh ? <LiveRefreshPill className="mb-3">Refreshing model catalog</LiveRefreshPill> : null}
      {conversations.conversations.length === 0 && !showLiveLoading && <div>No conversations yet</div>}
      <ChatLayout
        sidebar={sidebar}
        title={data.title}
        subtitle={activeConversation?.title}
        actions={actions}
        composer={
          <Composer
            value={prompt}
            onChange={setPrompt}
            onSend={() => {
              void chat.sendMessage(prompt)
              setPrompt('')
            }}
          />
        }
        onMessageAreaClick={() => setInspectedMessage(undefined)}
      >
        {activeMessages.map((message) => {
          const transparencyMessage = message.inspectMessage
          return (
            <MessageRow
              key={message.id}
              messageRole={message.messageRole}
              timestamp={message.timestamp}
              model={message.model}
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
            />
          )
        })}
      </ChatLayout>
    </>
  )
}

export function ChatPage(props: ChatPageProps = {}) {
  return (
    <QueryProvider>
      <ChatPageContent {...props} />
    </QueryProvider>
  )
}
