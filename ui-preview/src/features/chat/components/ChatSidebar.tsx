import type { ReactNode } from 'react'
import { MessageSquare, ArrowRightLeft, Plus } from 'lucide-react'
import { SidebarNavigation, type SidebarNavigationItem, type SidebarNavigationSection } from '@/components/ui/SidebarNavigation'
import { TabPanel, type TabPanelItem } from '@/components/ui/TabPanel'
import type { Conversation, ConversationGroup } from '@/features/app-tabs/types'

type SidebarTab = 'conversations' | 'transparency'

type ChatSidebarProps = { tab: SidebarTab; onTabChange: (tab: SidebarTab) => void; conversations: Conversation[]; conversationGroups?: ConversationGroup[]; activeId?: string; onSelectConversation?: (conversation: Conversation) => void; onNewChat?: () => void; transparency: ReactNode; showTransparency?: boolean }

function defaultConversationGroups(conversations: Conversation[]): ConversationGroup[] {
  return [
    { title: 'Today', conversationIds: conversations.slice(0, 2).map((conversation) => conversation.id) },
    { title: 'Earlier', conversationIds: conversations.slice(2).map((conversation) => conversation.id) },
  ]
}

export function ChatSidebar({ tab, onTabChange, conversations, conversationGroups, activeId, onSelectConversation, onNewChat, transparency, showTransparency = true }: ChatSidebarProps) {
  const groups = conversationGroups ?? defaultConversationGroups(conversations)
  const conversationsById = new Map(conversations.map((conversation) => [conversation.id, conversation]))
  const activeConversationId = activeId ?? conversations[0]?.id ?? ''
  const conversationSections: SidebarNavigationSection[] = groups.map((group, index) => ({
    id: `${group.title}-${index}`,
    title: group.title,
    items: group.conversationIds.reduce<SidebarNavigationItem[]>((items, id) => {
      const conversation = conversationsById.get(id)

      if (conversation) {
        items.push({ id: conversation.id, label: conversation.title, count: conversation.updatedAt })
      }

      return items
    }, []),
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
      accessory: <span className="rounded-full border border-border-soft px-[5px] font-mono text-[length:var(--density-type-annotation)] text-fg-faint">{conversations.length}</span>,
      content: (
        <SidebarNavigation
          activeId={activeConversationId}
          ariaLabel="Conversations"
          navClassName="space-y-0"
          onSelect={handleSelectConversation}
          sectionItemsClassName="space-y-0"
          sections={conversationSections}
        />
      ),
    },
    ...(showTransparency ? [{ value: 'transparency' as const, label: 'Transparency', icon: ArrowRightLeft, content: transparency }] : []),
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
      tabBarAccessory={tab === 'conversations' && onNewChat ? (
        <button
          onClick={onNewChat}
          type="button"
          className="ui-control inline-flex h-[22px] items-center gap-[5px] rounded-[var(--radius)] border px-[9px] text-[length:var(--density-type-caption)] font-medium outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
        >
          <Plus className="size-[11px]" /> New
        </button>
      ) : null}
      tabBarClassName="border-border-soft pl-0 pr-1.5"
      tabs={tabItems}
      value={tab}
    />
  )
}
