import { useEffect, useRef, useState, type ReactNode } from 'react'
import { ArrowRightLeft, Check, MessageSquare, MoreVertical, Pencil, Plus, Trash2, X } from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/DropdownMenu'
import {
  SidebarNavigation,
  type SidebarNavigationItem,
  type SidebarNavigationSection
} from '@/components/ui/SidebarNavigation'
import { TabPanel, type TabPanelItem } from '@/components/ui/TabPanel'
import { buildConversationGroups } from '@/features/chat/api/conversation-groups'
import type { Conversation, ConversationGroup } from '@/features/app-tabs/types'

type SidebarTab = 'conversations' | 'transparency'
type DeleteConversationOptions = { returnFocusElement?: HTMLElement | null }

type ChatSidebarProps = {
  tab: SidebarTab
  onTabChange: (tab: SidebarTab) => void
  conversations: Conversation[]
  conversationGroups?: ConversationGroup[]
  activeId?: string
  messageCounts?: Record<string, number>
  streamingConversationIds?: readonly string[]
  onSelectConversation?: (conversation: Conversation) => void
  onRenameConversation?: (conversation: Conversation, title: string) => void
  onDeleteConversation?: (conversation: Conversation, options?: DeleteConversationOptions) => void
  onNewChat?: () => void
  transparency: ReactNode
  showTransparency?: boolean
}

const conversationTimestampFormatter = new Intl.DateTimeFormat(undefined, {
  month: 'short',
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit'
})

function formatConversationTimestamp(value: string): string {
  const trimmedValue = value.trim()
  const today = new Date()
  const timeOnlyMatch = trimmedValue.match(/^(\d{1,2}):(\d{2})(?:\s*([ap])\.?m\.?)?$/i)

  if (!trimmedValue || /^now$/i.test(trimmedValue)) {
    return conversationTimestampFormatter.format(today)
  }

  if (timeOnlyMatch) {
    const hoursValue = Number(timeOnlyMatch[1])
    const minutesValue = Number(timeOnlyMatch[2])
    const period = timeOnlyMatch[3]?.toLowerCase()

    if (hoursValue <= 23 && minutesValue <= 59) {
      const timestamp = new Date(today)
      const hours =
        period === 'p' && hoursValue < 12 ? hoursValue + 12 : period === 'a' && hoursValue === 12 ? 0 : hoursValue
      timestamp.setHours(hours, minutesValue, 0, 0)

      return conversationTimestampFormatter.format(timestamp)
    }
  }

  if (/^yesterday$/i.test(trimmedValue)) {
    const yesterday = new Date(today)
    yesterday.setDate(today.getDate() - 1)
    return conversationTimestampFormatter.format(yesterday)
  }

  const parsedDate = new Date(trimmedValue)
  if (!Number.isNaN(parsedDate.getTime())) {
    return conversationTimestampFormatter.format(parsedDate)
  }

  return trimmedValue
}

function formatMessageCount(count: number): string {
  return `${count} message${count === 1 ? '' : 's'}`
}

function GeneratingDots() {
  return (
    <output className="inline-flex items-center gap-1 text-accent" aria-label="Generating response">
      <span aria-hidden={true}>Generating</span>
      <span className="chat-generating-dots" aria-hidden={true}>
        <span />
        <span />
        <span />
      </span>
    </output>
  )
}

export function ChatSidebar({
  tab,
  onTabChange,
  conversations,
  conversationGroups,
  activeId,
  messageCounts = {},
  streamingConversationIds = [],
  onSelectConversation,
  onRenameConversation,
  onDeleteConversation,
  onNewChat,
  transparency,
  showTransparency = true
}: ChatSidebarProps) {
  const [editingConversationId, setEditingConversationId] = useState<string | undefined>()
  const [editingTitle, setEditingTitle] = useState('')
  const editingInputRef = useRef<HTMLInputElement | null>(null)
  const actionTriggerRefs = useRef(new Map<string, HTMLButtonElement>())
  const groups = conversationGroups ?? buildConversationGroups(conversations)
  const conversationsById = new Map(conversations.map((conversation) => [conversation.id, conversation]))
  const activeConversationId = activeId ?? conversations[0]?.id ?? ''
  const streamingConversationSet = new Set(streamingConversationIds)

  useEffect(() => {
    if (!editingConversationId) return
    const input = editingInputRef.current
    input?.focus()
    input?.select()
  }, [editingConversationId])

  function startRename(conversation: Conversation) {
    setEditingConversationId(conversation.id)
    setEditingTitle(conversation.title)
  }

  function cancelRename() {
    setEditingConversationId(undefined)
    setEditingTitle('')
  }

  function saveRename(conversation: Conversation) {
    onRenameConversation?.(conversation, editingTitle)
    cancelRename()
  }

  function renderConversationActions(conversation: Conversation) {
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            aria-label={`Open actions for ${conversation.title}`}
            className="grid size-7 place-items-center rounded-[var(--radius)] text-fg-faint outline-none transition-[background,color] hover:bg-panel-strong hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
            ref={(node) => {
              if (node) {
                actionTriggerRefs.current.set(conversation.id, node)
                return
              }

              actionTriggerRefs.current.delete(conversation.id)
            }}
            type="button"
          >
            <MoreVertical className="size-3.5" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem onSelect={() => startRename(conversation)}>
            <Pencil className="size-3.5" /> Rename
          </DropdownMenuItem>
          <DropdownMenuSeparator className="my-1 h-px bg-border-soft" />
          <DropdownMenuItem
            tone="destructive"
            onSelect={() =>
              onDeleteConversation?.(conversation, {
                returnFocusElement: actionTriggerRefs.current.get(conversation.id) ?? null
              })
            }
          >
            <Trash2 className="size-3.5" /> Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    )
  }

  function renderRenameEditor(conversation: Conversation) {
    return (
      <form
        className="flex items-center gap-1.5"
        onSubmit={(event) => {
          event.preventDefault()
          saveRename(conversation)
        }}
      >
        <input
          ref={editingInputRef}
          aria-label={`Rename ${conversation.title}`}
          className="min-w-0 flex-1 rounded-[var(--radius)] border border-border bg-panel-strong px-2 py-1 text-[length:var(--density-type-control-lg)] text-fg outline-none focus-visible:border-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
          onChange={(event) => setEditingTitle(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Escape') cancelRename()
          }}
          value={editingTitle}
        />
        <button
          aria-label="Save chat title"
          className="grid size-7 place-items-center rounded-[var(--radius)] text-fg-faint outline-none transition-[background,color] hover:bg-panel-strong hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
          type="submit"
        >
          <Check className="size-3.5" />
        </button>
        <button
          aria-label="Cancel chat rename"
          className="grid size-7 place-items-center rounded-[var(--radius)] text-fg-faint outline-none transition-[background,color] hover:bg-panel-strong hover:text-fg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
          onClick={cancelRename}
          type="button"
        >
          <X className="size-3.5" />
        </button>
      </form>
    )
  }

  const conversationSections: SidebarNavigationSection[] = groups.map((group, index) => ({
    id: `${group.title}-${index}`,
    title: group.title,
    items: group.conversationIds.reduce<SidebarNavigationItem[]>((items, id) => {
      const conversation = conversationsById.get(id)

      if (conversation) {
        const messageCount = messageCounts[conversation.id] ?? 0
        const isEditing = conversation.id === editingConversationId
        const isGenerating = streamingConversationSet.has(conversation.id)
        items.push({
          id: conversation.id,
          label: conversation.title,
          summary: (
            <span className="flex min-w-0 flex-wrap items-center gap-x-1.5 gap-y-0.5">
              <span>
                {formatMessageCount(messageCount)} · {formatConversationTimestamp(conversation.updatedAt)}
              </span>
              {isGenerating ? (
                <>
                  <span aria-hidden={true}>·</span>
                  <GeneratingDots />
                </>
              ) : null}
            </span>
          ),
          action: isEditing ? undefined : renderConversationActions(conversation),
          editingContent: isEditing ? renderRenameEditor(conversation) : undefined
        })
      }

      return items
    }, [])
  }))

  function handleSelectConversation(id: string) {
    const conversation = conversationsById.get(id)

    if (conversation) {
      onSelectConversation?.(conversation)
    }
  }

  const tabItems: TabPanelItem<SidebarTab>[] = [
    {
      value: 'conversations',
      label: 'Conversations',
      icon: MessageSquare,
      accessory: (
        <span className="rounded-full border border-border-soft px-[5px] font-mono text-[length:var(--density-type-annotation)] text-fg-faint">
          {conversations.length}
        </span>
      ),
      content: (
        <SidebarNavigation
          activeId={activeConversationId}
          ariaLabel="Conversations"
          navClassName="space-y-0"
          onSelect={handleSelectConversation}
          sectionItemsClassName="space-y-0"
          sections={conversationSections}
        />
      )
    },
    ...(showTransparency
      ? [{ value: 'transparency' as const, label: 'Transparency', icon: ArrowRightLeft, content: transparency }]
      : [])
  ]

  return (
    <TabPanel
      ariaLabel="Chat sidebar views"
      className="panel-shell flex min-h-0 flex-col overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel"
      contentClassName="flex-1 overflow-auto px-3.5 py-3"
      iconClassName="size-3"
      listClassName="min-w-0 flex-1"
      onValueChange={(value) => {
        onTabChange(value)
      }}
      tabBarAccessory={
        tab === 'conversations' && onNewChat ? (
          <button
            onClick={onNewChat}
            type="button"
            className="ui-control inline-flex h-7 items-center gap-1.5 rounded-[var(--radius)] border px-3 text-[length:var(--density-type-control-lg)] font-medium outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
          >
            <Plus className="size-3.5" /> New
          </button>
        ) : null
      }
      tabBarClassName="border-border-soft pl-0 pr-2.5"
      tabs={tabItems}
      value={tab}
    />
  )
}
