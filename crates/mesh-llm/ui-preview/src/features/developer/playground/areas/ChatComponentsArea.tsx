import { useState } from 'react'
import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { ChatSidebar } from '@/features/chat/components/ChatSidebar'
import { Composer } from '@/features/chat/components/Composer'
import { MessageRow, type MessageAttachmentAction } from '@/features/chat/components/MessageRow'
import { ModelSelect } from '@/features/chat/components/ModelSelect'
import { TransparencyPane } from '@/features/chat/components/transparency/TransparencyPane'
import { CHAT_HARNESS } from '@/features/app-tabs/data'
import { useBooleanFeatureFlag } from '@/lib/feature-flags'
import { PlaygroundPanel, SidebarTabs, TextAreaField, TextField, ToggleChip } from '../primitives'
import type { DeveloperPlaygroundState } from '../useDeveloperPlaygroundState'

type TransparencyPreview = 'empty' | 'user' | 'assistant'

const PLAYGROUND_ATTACHMENTS: MessageAttachmentAction[] = [
  {
    id: 'rack-photo',
    label: 'rack-photo.png',
    kind: 'image',
    fileName: 'rack-photo.png',
    onOpen: () => undefined
  },
  {
    id: 'placement-plan',
    label: 'placement-plan.pdf',
    kind: 'pdf',
    fileName: 'placement-plan.pdf',
    onOpen: () => undefined
  }
]

const MARKDOWN_THINKING_RESPONSE = `<think>Check whether the selected peer already hosts a warm Qwen route before recommending a split.</think>
### Route note

- Prefer **carrack** for the first token check.
- Keep lemony-28 warm for fallback.

Use the [local status endpoint](http://localhost:3131/api/status) before changing placement.`

const STREAMING_THINKING_RESPONSE = `<think>Inspecting peer load, model warmth, and route latency.`

export function ChatComponentsArea({ state }: { state: DeveloperPlaygroundState }) {
  const transparencyTabEnabled = useBooleanFeatureFlag('chat/transparencyTab')
  const [transparencyPreview, setTransparencyPreview] = useState<TransparencyPreview>('empty')
  const transparencyMessage =
    transparencyPreview === 'user'
      ? state.activeUserMessage?.inspectMessage
      : transparencyPreview === 'assistant'
        ? state.activeAssistantMessage?.inspectMessage
        : undefined

  return (
    <SidebarTabs
      ariaLabel="Chat component previews"
      defaultValue="workspace"
      tabs={[
        {
          value: 'workspace',
          label: 'Workspace',
          content: (
            <>
              <PlaygroundPanel
                title="Chat preview controls"
                description="Edit the header and message copy that drive the preview so sidebar, rows, transparency, and composer stay in sync."
                actions={
                  <button
                    className="ui-control inline-flex items-center rounded-[var(--radius)] border px-2.5 py-1 text-[length:var(--density-type-caption)] font-medium"
                    onClick={() => state.setPrompt('Draft a short onboarding note for a new mesh peer.')}
                    type="button"
                  >
                    Load sample prompt
                  </button>
                }
              >
                <div className="grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
                  <div className="space-y-3">
                    <TextField label="Header title" value={state.chatHeaderTitle} onChange={state.setChatHeaderTitle} />
                    <TextField
                      label="Conversation label"
                      value={state.activeDraft.conversationLabel}
                      onChange={(value) => state.updateActiveChatDraft('conversationLabel', value)}
                    />
                    <TextAreaField
                      label="User message"
                      rows={3}
                      value={state.activeDraft.userBody}
                      onChange={(value) => state.updateActiveChatDraft('userBody', value)}
                    />
                    <TextAreaField
                      label="Assistant message"
                      rows={4}
                      value={state.activeDraft.assistantBody}
                      onChange={(value) => state.updateActiveChatDraft('assistantBody', value)}
                    />
                  </div>
                  <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                    <div className="type-label text-fg-faint">Active thread</div>
                    <div className="mt-2 font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
                      {state.activeDraft.conversationLabel}
                    </div>
                    <div className="mt-1 text-[length:var(--density-type-caption)] text-fg-dim">
                      Use the sidebar to swap threads. Edits only affect the current preview.
                    </div>
                  </div>
                </div>
              </PlaygroundPanel>

              <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
                <ChatSidebar
                  tab={state.sidebarTab}
                  onTabChange={state.setSidebarTab}
                  conversations={state.previewConversations}
                  conversationGroups={CHAT_HARNESS.conversationGroups}
                  activeId={state.activeConversation?.id}
                  onNewChat={() => {
                    state.setPrompt('')
                    state.setInspectedMessage(undefined)
                    state.setSidebarTab('conversations')
                  }}
                  onSelectConversation={(conversation) => {
                    state.setActiveConversationId(conversation.id)
                    state.setInspectedMessage(undefined)
                    state.setSidebarTab('conversations')
                  }}
                  showTransparency={transparencyTabEnabled}
                  transparency={
                    <TransparencyPane message={state.inspectedMessage} nodes={CHAT_HARNESS.transparencyNodes} />
                  }
                />

                <section className="flex min-h-0 flex-col overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
                  <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border-soft px-3.5 py-2.5">
                    <div>
                      <h2 className="type-panel-title text-foreground">{state.chatHeaderTitle}</h2>
                      <div className="mt-1 text-[length:var(--density-type-label)] text-fg-faint">
                        {state.activeDraft.conversationLabel}
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[length:var(--density-type-caption)] text-fg-faint">
                        {CHAT_HARNESS.modelLabel}
                      </span>
                      <ModelSelect
                        options={state.chatOptions}
                        value={state.selectedChatModel}
                        onChange={state.setSelectedChatModel}
                      />
                    </div>
                  </header>
                  <div className="flex-1 space-y-4 overflow-auto px-5 py-4">
                    {state.previewMessages.map((message) => (
                      <MessageRow
                        key={message.id}
                        body={message.body}
                        inspect={
                          transparencyTabEnabled && message.inspectMessage
                            ? () => {
                                state.setInspectedMessage(message.inspectMessage)
                                state.setSidebarTab('transparency')
                              }
                            : undefined
                        }
                        inspectLabel={message.inspectLabel}
                        inspected={Boolean(
                          message.inspectMessage && state.inspectedMessage?.id === message.inspectMessage.id
                        )}
                        messageRole={message.messageRole}
                        model={message.model}
                        route={message.route}
                        routeNode={message.routeNode}
                        showRouteMetadata={transparencyTabEnabled}
                        timestamp={message.timestamp}
                        tokens={message.tokens}
                        tokPerSec={message.tokPerSec}
                        ttft={message.ttft}
                      />
                    ))}
                  </div>
                  <div className="border-t border-border-soft bg-panel px-4 py-3">
                    <Composer value={state.prompt} onChange={state.setPrompt} onSend={() => state.setPrompt('')} />
                  </div>
                </section>
              </div>
            </>
          )
        },
        {
          value: 'rows',
          label: 'Message rows',
          content: (
            <PlaygroundPanel
              title="Row focus"
              description="Inspect the current user and assistant rows in isolation while keeping the same live text edits."
            >
              <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
                <div className="space-y-4">
                  {state.previewMessages
                    .filter(
                      (message) =>
                        message.id === state.activeUserMessage?.id || message.id === state.activeAssistantMessage?.id
                    )
                    .map((message) => (
                      <MessageRow
                        key={message.id}
                        body={message.body}
                        inspect={
                          transparencyTabEnabled && message.inspectMessage
                            ? () => {
                                state.setInspectedMessage(message.inspectMessage)
                                state.setSidebarTab('transparency')
                              }
                            : undefined
                        }
                        inspectLabel={message.inspectLabel}
                        inspected={Boolean(
                          message.inspectMessage && state.inspectedMessage?.id === message.inspectMessage.id
                        )}
                        messageRole={message.messageRole}
                        model={message.model}
                        route={message.route}
                        routeNode={message.routeNode}
                        showRouteMetadata={transparencyTabEnabled}
                        timestamp={message.timestamp}
                        tokens={message.tokens}
                        tokPerSec={message.tokPerSec}
                        ttft={message.ttft}
                      />
                    ))}
                </div>
                <TransparencyPane message={state.inspectedMessage} nodes={CHAT_HARNESS.transparencyNodes} />
              </div>
            </PlaygroundPanel>
          )
        },
        {
          value: 'states',
          label: 'States',
          content: (
            <>
              <PlaygroundPanel
                title="Transparency states"
                description="Switch between empty, outbound user, and inbound assistant transparency payloads without depending on message-row clicks."
              >
                <div className="mb-4 flex flex-wrap gap-1.5">
                  <ToggleChip
                    label="No message"
                    pressed={transparencyPreview === 'empty'}
                    onToggle={() => setTransparencyPreview('empty')}
                  />
                  <ToggleChip
                    label="User route"
                    pressed={transparencyPreview === 'user'}
                    onToggle={() => setTransparencyPreview('user')}
                  />
                  <ToggleChip
                    label="Assistant route"
                    pressed={transparencyPreview === 'assistant'}
                    onToggle={() => setTransparencyPreview('assistant')}
                  />
                </div>
                <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
                  <div className="space-y-4">
                    {state.previewMessages.slice(0, 2).map((message) => (
                      <MessageRow
                        key={message.id}
                        body={message.body}
                        inspectLabel={message.inspectLabel}
                        inspected={message.inspectMessage?.id === transparencyMessage?.id}
                        messageRole={message.messageRole}
                        model={message.model}
                        route={message.route}
                        routeNode={message.routeNode}
                        showRouteMetadata={transparencyTabEnabled}
                        timestamp={message.timestamp}
                        tokens={message.tokens}
                        tokPerSec={message.tokPerSec}
                        ttft={message.ttft}
                      />
                    ))}
                  </div>
                  <TransparencyPane message={transparencyMessage} nodes={CHAT_HARNESS.transparencyNodes} />
                </div>
              </PlaygroundPanel>

              <PlaygroundPanel
                title="Message row states"
                description="Queue removal, streaming stop, stopped stats, error copy, markdown, thinking traces, and attachment actions are visible without sending a live request."
              >
                <div className="space-y-4">
                  <MessageRow
                    attachments={PLAYGROUND_ATTACHMENTS}
                    body="Compare this rack photo against the planned split and note any peer that should stay client-only."
                    inspect={() => setTransparencyPreview('user')}
                    inspectLabel="Inspect attachment dispatch"
                    messageRole="user"
                    model="Multimodal dispatch"
                    routeNode="carrack"
                    state="default"
                    timestamp="14:31"
                  />
                  <MessageRow
                    body="Queued behind a local warmup. Remove this prompt if the operator changes model targets."
                    messageRole="user"
                    model="Qwen3.6-27B-UD"
                    onRemoveQueued={() => undefined}
                    routeNode="lemony-28"
                    state="queued"
                    timestamp="14:32"
                  />
                  <MessageRow
                    body={STREAMING_THINKING_RESPONSE}
                    messageRole="assistant"
                    model="Qwen3.6-27B-UD"
                    onStopStreaming={() => undefined}
                    route="mesh route"
                    routeNode="carrack"
                    state="streaming"
                    timestamp="14:33"
                    tokens="42 tok"
                    tokPerSec="18.4 tok/s"
                    ttft="181ms"
                  />
                  <MessageRow
                    body={MARKDOWN_THINKING_RESPONSE}
                    messageRole="assistant"
                    model="Qwen3.6-27B-UD"
                    route="mesh route"
                    routeNode="carrack"
                    state="stopped"
                    timestamp="14:34"
                    tokens="218 tok"
                    tokPerSec="21.7 tok/s"
                    ttft="164ms"
                  />
                  <MessageRow
                    body="The selected model is no longer reachable on any advertised peer."
                    messageRole="user"
                    model="Qwen3.6-35B-A3B-UD"
                    routeNode="lemony-29"
                    state="error"
                    timestamp="14:35"
                  />
                </div>
              </PlaygroundPanel>

              <LiveDataUnavailableOverlay
                debugTitle="Chat transport is offline"
                title="Live chat is unavailable"
                debugDescription="The playground keeps the chat workspace visible while showing the overlay copy used when a live streaming transport cannot connect."
                productionDescription="The chat workspace is waiting for the mesh chat service. Operators can retry or continue inspecting harness conversations."
                onRetry={() => undefined}
                onSwitchToTestData={() => undefined}
              >
                <section className="flex min-h-[360px] flex-col overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
                  <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border-soft px-3.5 py-2.5">
                    <div>
                      <h2 className="type-panel-title text-foreground">{state.chatHeaderTitle}</h2>
                      <div className="mt-1 text-[length:var(--density-type-label)] text-fg-faint">
                        Unavailable live transport preview
                      </div>
                    </div>
                    <ModelSelect
                      options={state.chatOptions}
                      value={state.selectedChatModel}
                      onChange={state.setSelectedChatModel}
                    />
                  </header>
                  <div className="flex-1 space-y-4 overflow-auto px-5 py-4">
                    {state.previewMessages.slice(0, 2).map((message) => (
                      <MessageRow
                        key={message.id}
                        body={message.body}
                        messageRole={message.messageRole}
                        model={message.model}
                        route={message.route}
                        routeNode={message.routeNode}
                        showRouteMetadata={transparencyTabEnabled}
                        timestamp={message.timestamp}
                        tokens={message.tokens}
                        tokPerSec={message.tokPerSec}
                        ttft={message.ttft}
                      />
                    ))}
                  </div>
                </section>
              </LiveDataUnavailableOverlay>
            </>
          )
        }
      ]}
    />
  )
}
