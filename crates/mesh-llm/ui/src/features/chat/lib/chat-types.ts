export type ChatAttachmentKind = "image" | "audio" | "file";

export type ChatAttachmentStatus = "pending" | "uploading" | "failed";

export type ChatMessageAudio = {
  dataUrl: string;
  mimeType: string;
  fileName?: string;
};

export type ChatAttachment = {
  id: string;
  kind: ChatAttachmentKind;
  dataUrl: string;
  mimeType: string;
  fileName?: string;
  status?: ChatAttachmentStatus;
  error?: string;
  extractedText?: string;
  extractionSummary?: string;
  renderedPageImages?: string[];
  imageDescription?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  model?: string;
  stats?: string;
  error?: boolean;
  image?: string;
  audio?: ChatMessageAudio;
  attachments?: ChatAttachment[];
};

export type ChatConversation = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
};

export type ChatState = {
  conversations: ChatConversation[];
  activeConversationId: string;
};

export type AttachmentStatePatch = Partial<
  Pick<
    ChatAttachment,
    "status" | "error" | "extractionSummary" | "imageDescription" | "renderedPageImages"
  >
>;
