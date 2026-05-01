import type { ChatAttachment, ChatMessage } from "./chat-types";

export function parseDataUrl(
  dataUrl: string,
): { mimeType: string; base64: string } | null {
  const match = /^data:([^;,]+);base64,(.+)$/s.exec(dataUrl);
  if (!match) return null;
  return { mimeType: match[1], base64: match[2] };
}

export function messageAttachments(message: ChatMessage): ChatAttachment[] {
  if (Array.isArray(message.attachments) && message.attachments.length > 0) {
    return message.attachments;
  }

  const attachments: ChatAttachment[] = [];
  if (message.image) {
    attachments.push({
      id: `${message.id}-image`,
      kind: "image",
      dataUrl: message.image,
      mimeType: parseDataUrl(message.image)?.mimeType || "image/jpeg",
      fileName: "image.jpg",
    });
  }
  if (message.audio) {
    attachments.push({
      id: `${message.id}-audio`,
      kind: "audio",
      dataUrl: message.audio.dataUrl,
      mimeType: message.audio.mimeType,
      fileName: message.audio.fileName,
    });
  }

  return attachments;
}
