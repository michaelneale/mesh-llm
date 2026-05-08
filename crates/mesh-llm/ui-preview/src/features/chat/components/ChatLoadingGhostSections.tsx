import { LoadingGhostBlock } from '@/components/ui/LoadingGhostBlock'

const CONVERSATION_GHOST_ROWS = ['conv-a', 'conv-b', 'conv-c', 'conv-d']
const MESSAGE_GHOST_ROWS = ['msg-a', 'msg-b', 'msg-c', 'msg-d', 'msg-e']

export function ChatSidebarLoadingGhost() {
  return (
    <aside className="panel-shell flex min-h-0 flex-col overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
      <div className="border-b border-border-soft px-3.5 py-2.5">
        <LoadingGhostBlock className="h-4 w-32" shimmer />
        <LoadingGhostBlock className="mt-2 h-3 w-44" shimmer />
      </div>
      <div className="space-y-2 p-3">
        {CONVERSATION_GHOST_ROWS.map((conversation) => (
          <LoadingGhostBlock key={conversation} className="h-14" shimmer />
        ))}
      </div>
    </aside>
  )
}

export function ChatActionsLoadingGhost() {
  return (
    <>
      <LoadingGhostBlock className="h-6 w-20 rounded-full" shimmer />
      <LoadingGhostBlock className="h-8 w-44" shimmer />
    </>
  )
}

export function ChatComposerLoadingGhost() {
  return <LoadingGhostBlock className="h-11" shimmer />
}

export function ChatMessagesLoadingGhost() {
  return (
    <div className="space-y-4">
      {MESSAGE_GHOST_ROWS.map((message, index) => (
        <div key={message} className={index % 2 === 0 ? 'mr-auto max-w-[72%]' : 'ml-auto max-w-[68%]'}>
          <LoadingGhostBlock className="h-4 w-32" shimmer />
          <LoadingGhostBlock className="mt-2 h-16" shimmer />
        </div>
      ))}
    </div>
  )
}
