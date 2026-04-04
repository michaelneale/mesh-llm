import { describe, expect, it } from "vitest";

import {
  AUDIO_MAX_BYTES,
  FILE_MAX_BYTES,
  IMAGE_MAX_BYTES,
  getAttachmentSendIssue,
  validateAttachmentFile,
} from "./attachments";

describe("validateAttachmentFile", () => {
  it("rejects wrong image mime types", () => {
    expect(
      validateAttachmentFile(
        { size: 128, type: "text/plain" },
        "image",
      ),
    ).toBe("Only image files are allowed here.");
  });

  it("rejects oversized audio files", () => {
    expect(
      validateAttachmentFile(
        { size: AUDIO_MAX_BYTES + 1, type: "audio/mpeg" },
        "audio",
      ),
    ).toContain("Audio file is too large");
  });

  it("accepts generic files within size limits", () => {
    expect(
      validateAttachmentFile(
        { size: FILE_MAX_BYTES, type: "application/pdf" },
        "file",
      ),
    ).toBeNull();
  });

  it("rejects oversized images", () => {
    expect(
      validateAttachmentFile(
        { size: IMAGE_MAX_BYTES + 1, type: "image/png" },
        "image",
      ),
    ).toContain("Image is too large");
  });
});

describe("getAttachmentSendIssue", () => {
  it("returns selected-model mismatch errors", () => {
    expect(
      getAttachmentSendIssue({
        pendingKinds: new Set(["audio"]),
        selectedModel: "vision-only",
        warmModels: ["vision-only"],
        visionModels: new Set(["vision-only"]),
        audioModels: new Set<string>(),
        multimodalModels: new Set(["vision-only"]),
      }),
    ).toBe(
      "Selected model does not support the attached media. Choose a compatible model or remove the attachment.",
    );
  });

  it("returns no warm model errors for auto selection", () => {
    expect(
      getAttachmentSendIssue({
        pendingKinds: new Set(["file"]),
        selectedModel: "auto",
        warmModels: ["text-only"],
        visionModels: new Set<string>(),
        audioModels: new Set<string>(),
        multimodalModels: new Set<string>(),
      }),
    ).toBe(
      "No warm model supports the attached media. Warm a compatible model or remove the attachment.",
    );
  });

  it("allows supported attachments", () => {
    expect(
      getAttachmentSendIssue({
        pendingKinds: new Set(["image", "file"]),
        selectedModel: "auto",
        warmModels: ["multi"],
        visionModels: new Set(["multi"]),
        audioModels: new Set<string>(),
        multimodalModels: new Set(["multi"]),
      }),
    ).toBeNull();
  });
});
