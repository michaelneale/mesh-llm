import {
  describeImageAttachmentForPrompt,
  describeRenderedPagesAsText,
} from "./vision-describe";
import { messageAttachments, parseDataUrl } from "./chat-attachments";
import type { AttachmentStatePatch, ChatAttachment, ChatMessage } from "./chat-types";

async function uploadRequestObject(params: {
  requestId: string;
  dataUrl: string;
  fileName?: string;
}): Promise<{ token: string }> {
  const parsed = parseDataUrl(params.dataUrl);
  if (!parsed) throw new Error("Attachment is not a valid base64 data URL");
  const response = await fetch("/api/objects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      request_id: params.requestId,
      mime_type: parsed.mimeType,
      file_name: params.fileName,
      bytes_base64: parsed.base64,
    }),
  });
  if (!response.ok) {
    throw new Error(`Attachment upload failed (${response.status})`);
  }
  return (await response.json()) as { token: string };
}

export function attachmentForMessage(
  attachment: ChatAttachment,
): Omit<ChatAttachment, "status" | "error"> {
  const { status, error, ...persistedAttachment } = attachment;
  if (persistedAttachment.extractedText) {
    persistedAttachment.renderedPageImages = undefined;
  }
  return persistedAttachment;
}

export async function buildAttachmentBlocks(
  attachments: ChatAttachment[],
  requestId: string,
  clientId: string,
  onStatusChange?: (attachmentId: string, patch: AttachmentStatePatch) => void,
) {
  const contentBlocks: Array<Record<string, unknown>> = [];
  for (const sourceAttachment of attachments) {
    const attachment = { ...sourceAttachment };
    const extractedText = attachment.extractedText;
    if (extractedText) {
      const label = attachment.fileName
        ? `[Content from ${attachment.fileName}]`
        : "[Extracted PDF content]";
      contentBlocks.push({
        type: "input_text",
        text: `${label}\n\n${extractedText}`,
      });
      continue;
    }

    const renderedPageImages = attachment.renderedPageImages;
    if (renderedPageImages?.length) {
      onStatusChange?.(attachment.id, { status: "uploading", error: undefined });
      const label = attachment.fileName
        ? `[Content from ${attachment.fileName}]`
        : "[Extracted PDF content]";
      const text = await describeRenderedPagesAsText(renderedPageImages, {
        onProgress: (message) => {
          onStatusChange?.(attachment.id, {
            status: "uploading",
            error: undefined,
            extractionSummary: message,
          });
        },
      });
      attachment.extractedText = text;
      attachment.renderedPageImages = undefined;
      contentBlocks.push({
        type: "input_text",
        text: `${label}\n\n${text}`,
      });
      onStatusChange?.(attachment.id, {
        status: "pending",
        error: undefined,
        renderedPageImages: undefined,
        extractionSummary: "Described scanned PDF pages in browser",
      });
      continue;
    }

    if (attachment.kind === "image") {
      let imageDescription = attachment.imageDescription?.trim() ?? "";
      if (!imageDescription) {
        onStatusChange?.(attachment.id, {
          status: "uploading",
          error: undefined,
          extractionSummary: "Describing image...",
        });
        const result = await describeImageAttachmentForPrompt(attachment.dataUrl, {
          onProgress: (message) => {
            onStatusChange?.(attachment.id, {
              status: "uploading",
              error: undefined,
              extractionSummary: message,
            });
          },
        });
        imageDescription = result.imageDescription?.trim() ?? "";
        attachment.imageDescription = result.imageDescription;
        attachment.extractionSummary = result.extractionSummary;
        attachment.error = result.error;
        onStatusChange?.(attachment.id, {
          status: result.error ? "failed" : "pending",
          error: result.error,
          extractionSummary: result.extractionSummary,
          imageDescription: result.imageDescription,
        });
      }
      contentBlocks.push({
        type: "input_text",
        text: imageDescription || "[Image attached but could not be described]",
      });
      continue;
    }

    onStatusChange?.(attachment.id, { status: "uploading", error: undefined });
    try {
      const upload = await uploadRequestObject({
        requestId,
        dataUrl: attachment.dataUrl,
        fileName: attachment.fileName,
      });
      const url = `mesh://blob/${clientId}/${upload.token}`;
      if (attachment.kind === "audio") {
        contentBlocks.push({
          type: "input_audio",
          audio_url: url,
        });
      } else {
        contentBlocks.push({
          type: "input_file",
          url,
          mime_type: attachment.mimeType,
          file_name: attachment.fileName,
        });
      }
      onStatusChange?.(attachment.id, { status: "pending", error: undefined });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      onStatusChange?.(attachment.id, { status: "failed", error: message });
      throw error;
    }
  }
  return contentBlocks;
}

export async function buildResponsesInput(
  historyForRequest: ChatMessage[],
  requestId: string,
  clientId: string,
  prebuiltContentByMessageId?: Record<string, Array<Record<string, unknown>>>,
) {
  return Promise.all(
    historyForRequest.map(async (message) => {
      const contentBlocks: Array<Record<string, unknown>> =
        prebuiltContentByMessageId?.[message.id]?.slice() ?? [];
      const attachments = messageAttachments(message);
      if (message.content.trim()) {
        contentBlocks.push({ type: "input_text", text: message.content });
      }
      if (!prebuiltContentByMessageId?.[message.id] && attachments.length > 0) {
        contentBlocks.push(
          ...(await buildAttachmentBlocks(attachments, requestId, clientId)),
        );
      }
      return {
        role: message.role,
        content:
          contentBlocks.length === 1 &&
          attachments.length === 0 &&
          contentBlocks[0].type === "input_text"
            ? message.content
            : contentBlocks,
      };
    }),
  );
}
