import { describe, expect, it } from "vitest";

import { messageAttachments, parseDataUrl } from "./chat-attachments";

describe("parseDataUrl", () => {
  it("extracts the mime type and base64 payload", () => {
    expect(parseDataUrl("data:image/png;base64,abc123")).toEqual({
      mimeType: "image/png",
      base64: "abc123",
    });
  });

  it("returns null for invalid data urls", () => {
    expect(parseDataUrl("https://example.com/file.png")).toBeNull();
  });
});

describe("messageAttachments", () => {
  it("prefers normalized attachments when present", () => {
    const attachments = messageAttachments({
      id: "msg-1",
      role: "user",
      content: "",
      image: "data:image/png;base64,legacy",
      attachments: [
        {
          id: "att-1",
          kind: "file",
          dataUrl: "data:text/plain;base64,aGVsbG8=",
          mimeType: "text/plain",
          fileName: "hello.txt",
        },
      ],
    });

    expect(attachments).toHaveLength(1);
    expect(attachments[0].id).toBe("att-1");
  });

  it("derives legacy image and audio attachments consistently", () => {
    const attachments = messageAttachments({
      id: "msg-legacy",
      role: "user",
      content: "",
      image: "data:image/png;base64,image-data",
      audio: {
        dataUrl: "data:audio/wav;base64,audio-data",
        mimeType: "audio/wav",
        fileName: "note.wav",
      },
    });

    expect(attachments).toEqual([
      {
        id: "msg-legacy-image",
        kind: "image",
        dataUrl: "data:image/png;base64,image-data",
        mimeType: "image/png",
        fileName: "image.jpg",
      },
      {
        id: "msg-legacy-audio",
        kind: "audio",
        dataUrl: "data:audio/wav;base64,audio-data",
        mimeType: "audio/wav",
        fileName: "note.wav",
      },
    ]);
  });
});
