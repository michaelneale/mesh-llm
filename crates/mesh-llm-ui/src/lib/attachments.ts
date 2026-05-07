export type AttachmentKind = "image" | "audio" | "file";

export const IMAGE_MAX_BYTES = 12 * 1024 * 1024;
export const AUDIO_MAX_BYTES = 24 * 1024 * 1024;
export const FILE_MAX_BYTES = 24 * 1024 * 1024;

type FileLike = {
  size: number;
  type: string;
};

type AttachmentSupportOptions = {
  /** The *effective* media kinds that still need model support.
   *  Images and PDFs are always handled in-browser (Florence-2 / pdf.js)
   *  so they should NOT appear here. Only audio and generic files need
   *  model support. */
  pendingKinds: ReadonlySet<AttachmentKind>;
  selectedModel: string;
  warmModels: readonly string[];
  audioModels: ReadonlySet<string>;
  multimodalModels: ReadonlySet<string>;
};

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function validateAttachmentFile(
  file: FileLike,
  kind: AttachmentKind,
): string | null {
  if (kind === "image") {
    if (!file.type.startsWith("image/")) return "Only image files are allowed here.";
    if (file.size > IMAGE_MAX_BYTES) {
      return `Image is too large (${formatBytes(file.size)}). Max is ${formatBytes(IMAGE_MAX_BYTES)}.`;
    }
    return null;
  }
  if (kind === "audio") {
    if (!file.type.startsWith("audio/")) return "Only audio files are allowed here.";
    if (file.size > AUDIO_MAX_BYTES) {
      return `Audio file is too large (${formatBytes(file.size)}). Max is ${formatBytes(AUDIO_MAX_BYTES)}.`;
    }
    return null;
  }
  if (file.size > FILE_MAX_BYTES) {
    return `File is too large (${formatBytes(file.size)}). Max is ${formatBytes(FILE_MAX_BYTES)}.`;
  }
  return null;
}

export function getAttachmentSendIssue({
  pendingKinds,
  selectedModel,
  warmModels,
  audioModels,
  multimodalModels,
}: AttachmentSupportOptions): string | null {
  // Images and PDFs are always handled in the browser (described via
  // Florence-2 / extracted via pdf.js) so they never require model support.
  // Only audio and generic file attachments still need a capable model.
  if (!pendingKinds.size) return null;

  const modelSupports = (modelName: string) =>
    (!pendingKinds.has("audio") || audioModels.has(modelName)) &&
    (!pendingKinds.has("file") || multimodalModels.has(modelName));

  if (selectedModel && selectedModel !== "auto") {
    if (modelSupports(selectedModel)) return null;

    if (pendingKinds.has("audio") && !audioModels.has(selectedModel)) {
      return `${selectedModel} doesn't support audio. Choose an audio-capable model or remove the attachment.`;
    }
    return "Selected model does not support the attached media. Choose a compatible model or remove the attachment.";
  }

  if (warmModels.some(modelSupports)) return null;

  // Auto mode but no warm model supports it.
  if (pendingKinds.has("audio")) {
    return "No warm model supports audio input. Warm an audio-capable model to use audio attachments.";
  }
  return "No warm model supports the attached media. Warm a compatible model or remove the attachment.";
}
