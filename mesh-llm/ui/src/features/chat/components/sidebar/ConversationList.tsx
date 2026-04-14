import { type Dispatch, type MutableRefObject, type Ref, type SetStateAction } from "react";
import { Check, MessageSquarePlus, Pencil, Trash2, X } from "lucide-react";

import { Button } from "../../../../components/ui/button";
import { ScrollArea } from "../../../../components/ui/scroll-area";
import { cn } from "../../../../lib/utils";
import type { ChatConversation } from "../../lib/chat-types";

export function ConversationList({
  conversations,
  activeConversationId,
  editingConversationId,
  editingTitle,
  editingTitleInputRef,
  setEditingTitle,
  hasChats,
  isSending,
  onConversationCreate,
  onConversationSelect,
  onConversationStartRename,
  onConversationSaveRename,
  onConversationCancelRename,
  onConversationDelete,
  onConversationsClear,
  onConversationAction,
}: {
  conversations: ChatConversation[];
  activeConversationId: string;
  editingConversationId: string | null;
  editingTitle: string;
  editingTitleInputRef: MutableRefObject<HTMLInputElement | null>;
  setEditingTitle: Dispatch<SetStateAction<string>>;
  hasChats: boolean;
  isSending: boolean;
  onConversationCreate: () => void;
  onConversationSelect: (conversationId: string) => void;
  onConversationStartRename: (conversation: ChatConversation) => void;
  onConversationSaveRename: () => void;
  onConversationCancelRename: () => void;
  onConversationDelete: (conversation: ChatConversation) => void;
  onConversationsClear: () => void;
  onConversationAction: () => void;
}) {
  return (
    <div className="space-y-3 p-3">
      <Button
        type="button"
        size="sm"
        className="w-full"
        onClick={() => {
          onConversationCreate();
          onConversationAction();
        }}
        disabled={isSending}
      >
        <MessageSquarePlus className="mr-1.5 h-4 w-4" />
        New chat
      </Button>
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs text-muted-foreground">Conversations</div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-7 px-2 text-xs"
          onClick={onConversationsClear}
          disabled={!hasChats || isSending}
        >
          <Trash2 className="mr-1.5 h-3.5 w-3.5" />
          Clear
        </Button>
      </div>
      <ScrollArea className="h-[calc(100svh_-_14rem)] md:h-[calc(100svh_-_24rem)]">
        <div className="space-y-1">
          {conversations.map((conversation) => {
            const isActive = conversation.id === activeConversationId;
            const isEditing = editingConversationId === conversation.id;
            return (
              <div
                key={conversation.id}
                className={cn(
                  "group flex items-center gap-2 rounded-md border p-2",
                  isActive
                    ? "border-primary/40 bg-muted/40"
                    : "border-transparent",
                )}
              >
                {isEditing ? (
                  <div className="min-w-0 flex-1 space-y-1">
                    <input
                      ref={editingTitleInputRef as Ref<HTMLInputElement>}
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          onConversationSaveRename();
                        } else if (e.key === "Escape") {
                          e.preventDefault();
                          onConversationCancelRename();
                        }
                      }}
                      className="h-7 w-full rounded-md border bg-background px-2 text-sm outline-none ring-offset-background focus:ring-2 focus:ring-ring"
                    />
                    <div className="text-xs text-muted-foreground">
                      {conversation.messages.length} message
                      {conversation.messages.length === 1 ? "" : "s"}
                    </div>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="min-w-0 flex-1 text-left"
                    onClick={() => {
                      onConversationSelect(conversation.id);
                      onConversationAction();
                    }}
                    disabled={isSending}
                  >
                    <div className="text-sm font-medium leading-5 [display:-webkit-box] [overflow:hidden] [-webkit-box-orient:vertical] [-webkit-line-clamp:3] [overflow-wrap:anywhere]">
                      {conversation.title}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {conversation.messages.length} message
                      {conversation.messages.length === 1 ? "" : "s"}
                    </div>
                  </button>
                )}
                {isEditing ? (
                  <>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 shrink-0 opacity-70 hover:opacity-100"
                      onClick={onConversationSaveRename}
                      aria-label="Save conversation name"
                    >
                      <Check className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 shrink-0 opacity-70 hover:opacity-100"
                      onClick={onConversationCancelRename}
                      aria-label="Cancel rename"
                    >
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </>
                ) : (
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 shrink-0 opacity-70 hover:opacity-100"
                    onClick={() => onConversationStartRename(conversation)}
                    disabled={isSending}
                    aria-label="Rename conversation"
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </Button>
                )}
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 shrink-0 opacity-70 hover:opacity-100"
                  onClick={() => onConversationDelete(conversation)}
                  disabled={isSending}
                  aria-label="Delete conversation"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
