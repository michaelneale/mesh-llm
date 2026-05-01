import { createChatId } from "./chat-id";
import type { ChatConversation, ChatMessage, ChatState } from "./chat-types";

const DEFAULT_CHAT_TITLE = "New chat";

export function createConversation(
  seed?: Partial<Pick<ChatConversation, "title" | "messages">>,
): ChatConversation {
  const now = Date.now();
  return {
    id: createChatId(),
    title: seed?.title || DEFAULT_CHAT_TITLE,
    createdAt: now,
    updatedAt: now,
    messages: seed?.messages || [],
  };
}

export function createInitialChatState(): ChatState {
  return { conversations: [], activeConversationId: "" };
}

export async function loadPersistedChatState(
  readPersistedState: () => Promise<ChatState | null>,
): Promise<ChatState> {
  const fromDb = await readPersistedState();
  return fromDb ?? createInitialChatState();
}

export function findLastUserMessageIndex(messages: ChatMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === "user") return i;
  }
  return -1;
}
