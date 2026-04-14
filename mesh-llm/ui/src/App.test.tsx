import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

import {
  App,
  attachmentForMessage,
  ChatPage,
  describeImageAttachmentForPrompt,
  describeRenderedPagesAsText,
} from "./App";

function buildProps(
  overrides: Partial<Parameters<typeof ChatPage>[0]> = {},
): Parameters<typeof ChatPage>[0] {
  return {
    status: {
      node_id: "node-1",
      token: "invite-token",
      node_status: "host",
      is_host: true,
      is_client: false,
      llama_ready: true,
      api_port: 9337,
      model_name: "model-a",
      model_size_gb: 1,
      inflight_requests: 0,
      my_vram_gb: 12,
      peers: [],
    },
    inviteToken: "invite-token",
    isPublicMesh: false,
    isFlyHosted: false,
    inflightRequests: 0,
    warmModels: ["model-a"],
    meshModelByName: {},
    modelStatsByName: {},
    selectedModel: "model-a",
    setSelectedModel: vi.fn(),
    selectedModelNodeCount: 1,
    selectedModelVramGb: 12,
    selectedModelAudio: true,
    selectedModelMultimodal: true,
    composerError: null,
    setComposerError: vi.fn(),
    attachmentSendIssue: null,
    attachmentPreparationMessage: null,
    pendingAttachments: [],
    setPendingAttachments: vi.fn(),
    conversations: [
      {
        id: "chat-1",
        title: "Chat 1",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [],
      },
    ],
    activeConversationId: "chat-1",
    onConversationCreate: vi.fn(),
    onConversationSelect: vi.fn(),
    onConversationRename: vi.fn(),
    onConversationDelete: vi.fn(),
    onConversationsClear: vi.fn(),
    messages: [],
    reasoningOpen: {},
    setReasoningOpen: vi.fn(),
    chatScrollRef: { current: null },
    input: "",
    setInput: vi.fn(),
    isSending: false,
    queuedText: null,
    canChat: true,
    canRegenerate: false,
    onStop: vi.fn(),
    onRegenerate: vi.fn(),
    onSubmit: vi.fn(),
    ...overrides,
  };
}

const statusTemplate = {
  version: "1.0.0",
  latest_version: null,
  node_id: "node-1",
  token: "token-123",
  node_status: "Host",
  is_host: true,
  is_client: false,
  llama_ready: true,
  model_name: "model-a",
  models: ["model-a"],
  available_models: ["model-a"],
  requested_models: [],
  serving_models: ["model-a"],
  hosted_models: ["model-a"],
  api_port: 9337,
  my_vram_gb: 16,
  model_size_gb: 8,
  mesh_name: "test-mesh",
  peers: [],
  inflight_requests: 0,
  nostr_discovery: false,
  my_hostname: "host.local",
  gpus: [] as unknown[],
};

let statusPayload = createStatusPayload();
let modelsPayload = { mesh_models: [] as Array<Record<string, unknown>> };
const mockFetch = vi.fn();

function createStatusPayload() {
  return {
    ...statusTemplate,
    peers: [] as typeof statusTemplate.peers,
    models: [] as typeof statusTemplate.models,
    available_models: [] as typeof statusTemplate.available_models,
    requested_models: [] as typeof statusTemplate.requested_models,
    serving_models: [...statusTemplate.serving_models],
    hosted_models: [...statusTemplate.hosted_models],
    gpus: [] as typeof statusTemplate.gpus,
  };
}

function createResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function getRequestUrl(input: RequestInfo | URL) {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.href;
  return input.url;
}

function setupFetchMock() {
  mockFetch.mockImplementation((input: RequestInfo | URL) => {
    const url = getRequestUrl(input);
    if (url.endsWith("/api/status")) {
      return Promise.resolve(createResponse(statusPayload));
    }
    if (url.endsWith("/api/models")) {
      return Promise.resolve(createResponse(modelsPayload));
    }
    return Promise.resolve(createResponse({}));
  });
  globalThis.fetch = mockFetch as typeof fetch;
}

function setPath(path: string) {
  window.history.replaceState({}, "", path);
}

class MockEventSource {
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  readyState = 0;
  withCredentials = false;

  constructor(public url: string) {
    queueMicrotask(() => {
      this.onopen?.(new Event("open"));
    });
  }

  close() {}

  addEventListener() {}

  removeEventListener() {}

  dispatchEvent() {
    return false;
  }
}

beforeAll(() => {
  const makeMatchMedia = () => ({
    matches: false,
    media: "",
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  });

  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: () => makeMatchMedia(),
  });

  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: vi.fn().mockResolvedValue(undefined) },
  });
});

beforeEach(() => {
  statusPayload = createStatusPayload();
  modelsPayload = { mesh_models: [] };
  setupFetchMock();
  Object.defineProperty(window, "EventSource", {
    configurable: true,
    writable: true,
    value: MockEventSource,
  });
  setPath("/");
});

afterEach(() => {
  vi.resetAllMocks();
  setPath("/");
});

describe("ChatPage", () => {
  it("allows attachment-only sends and renders attachment controls", () => {
    render(
      <ChatPage
        {...buildProps({
          pendingAttachments: [
            {
              id: "att-1",
              kind: "file",
              dataUrl: "data:text/plain;base64,aGVsbG8=",
              mimeType: "text/plain",
              fileName: "hello.txt",
              status: "pending",
            },
          ],
        })}
      />,
    );

    expect(screen.getByTestId("chat-file-input")).toBeInTheDocument();
    expect(screen.getByTestId("chat-image-input")).toBeInTheDocument();
    expect(screen.getByTestId("chat-audio-input")).toBeInTheDocument();
    expect(screen.getByTestId("chat-send")).toBeEnabled();
    expect(screen.getByText("hello.txt")).toBeInTheDocument();
  });

  it("renders attachment policy errors", () => {
    render(
      <ChatPage
        {...buildProps({
          attachmentSendIssue:
            "Selected model does not support the attached media. Choose a compatible model or remove the attachment.",
        })}
      />,
    );

    expect(screen.getByTestId("composer-error")).toHaveTextContent(
      "Selected model does not support the attached media.",
    );
  });

  it("shows attachment preparation progress and disables send", () => {
    render(
      <ChatPage
        {...buildProps({
          attachmentPreparationMessage: "Preparing PDF in browser…",
          pendingAttachments: [
            {
              id: "att-pdf",
              kind: "file",
              dataUrl: "data:application/pdf;base64,abc",
              mimeType: "application/pdf",
              fileName: "scan.pdf",
              status: "uploading",
            },
          ],
        })}
      />,
    );

    expect(screen.getByText("Preparing PDF in browser…")).toBeInTheDocument();
    expect(screen.getByTestId("chat-send")).toBeDisabled();
  });

  it("shows failed image-description state with retry affordance", () => {
    render(
      <ChatPage
        {...buildProps({
          pendingAttachments: [
            {
              id: "att-image-failed",
              kind: "image",
              dataUrl: "data:image/png;base64,abc",
              mimeType: "image/png",
              fileName: "legacy.png",
              status: "failed",
              extractionSummary: "Image description failed — retry or send placeholder text",
              error: "Image description failed: model init failed",
            },
          ],
        })}
      />,
    );

    expect(screen.getByText("Retry")).toBeInTheDocument();
    expect(
      screen.getByText("Image description failed: model init failed"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Image description failed — retry or send placeholder text"),
    ).toBeInTheDocument();
  });

  it("shows Queue button label and calls onSubmit when isSending=true", () => {
    const onSubmit = vi.fn();
    render(
      <ChatPage
        {...buildProps({ isSending: true, input: "next message", onSubmit })}
      />,
    );

    const btn = screen.getByTestId("chat-send");
    expect(btn).toHaveTextContent("Queue");
    btn.click();
    expect(onSubmit).toHaveBeenCalled();
  });

  it("renders queued bubble with the queued text when queuedText is set", () => {
    render(
      <ChatPage
        {...buildProps({
          isSending: true,
          queuedText: "queued message",
          messages: [
            {
              id: "msg-1",
              role: "user" as const,
              content: "hello",
            },
          ],
        })}
      />,
    );

    expect(screen.getByText("Queued")).toBeInTheDocument();
    expect(screen.getByText("queued message")).toBeInTheDocument();
  });

  it("shows Send button and no queued bubble when not sending", () => {
    render(<ChatPage {...buildProps({ isSending: false, queuedText: null })} />);

    expect(screen.getByTestId("chat-send")).toHaveTextContent("Send");
    expect(screen.queryByText("Queued")).not.toBeInTheDocument();
  });

  it("calls onSubmit for attachment-only queue (empty text, pending attachment, isSending=true)", () => {
    const onSubmit = vi.fn();
    render(
      <ChatPage
        {...buildProps({
          isSending: true,
          input: "",
          queuedText: "",
          pendingAttachments: [
            {
              id: "att-2",
              kind: "image",
              dataUrl: "data:image/png;base64,abc",
              mimeType: "image/png",
              fileName: "photo.png",
              status: "pending",
            },
          ],
          onSubmit,
        })}
      />,
    );

    const btn = screen.getByTestId("chat-send");
    expect(btn).toHaveTextContent("Queue");
    btn.click();
    expect(onSubmit).toHaveBeenCalled();
  });
});

describe("App routing and status", () => {
  it("desktop unknown path fallback resolves to dashboard behavior", async () => {
    setPath("/unknown-path");
    render(<App />);

    const networkLink = await screen.findByRole("link", { name: "Network" });
    expect(networkLink).toHaveAttribute("aria-current", "page");
    await waitFor(() => expect(window.location.pathname).toBe("/dashboard"));
    expect(screen.queryByRole("button", { name: /New chat/i })).not.toBeInTheDocument();
  });

  it("mobile unknown path fallback also syncs dashboard state", async () => {
    const previousInnerWidth = window.innerWidth;
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 640,
    });
    setPath("/unknown-path");

    try {
      render(<App />);

      const networkLink = await screen.findByRole("link", { name: "Network" });
      expect(networkLink).toHaveAttribute("aria-current", "page");
      await waitFor(() => expect(window.location.pathname).toBe("/dashboard"));
      expect(screen.queryByRole("button", { name: /New chat/i })).not.toBeInTheDocument();
    } finally {
      Object.defineProperty(window, "innerWidth", {
        configurable: true,
        writable: true,
        value: previousInnerWidth,
      });
    }
  });

  it("/dashboard route renders without redirecting to /config", async () => {
    setPath("/dashboard");
    render(<App />);

    const networkLink = await screen.findByRole("link", { name: "Network" });
    expect(networkLink).toHaveAttribute("aria-current", "page");
    await waitFor(() => expect(window.location.pathname).toBe("/dashboard"));
    expect(screen.queryByRole("button", { name: /New chat/i })).not.toBeInTheDocument();
  });

  it("/chat route renders chat section content", async () => {
    setPath("/chat");
    render(<App />);

    const chatLink = await screen.findByRole("link", { name: "Chat" });
    expect(chatLink).toHaveAttribute("aria-current", "page");
    await screen.findByRole("button", { name: /New chat/i });
    await waitFor(() => expect(window.location.pathname).toBe("/chat"));
    expect(
      screen.queryByRole("link", { current: "page", name: "Network" }),
    ).not.toBeInTheDocument();
  });

  it("boots /api/status on mount and consumes status payload", async () => {
    setPath("/dashboard");
    render(<App />);

    await waitFor(() =>
      expect(mockFetch.mock.calls.some((call) => call[0] === "/api/status")).toBe(
        true,
      ),
    );
    await screen.findByText("Mesh LLM v1.0.0");
  });

  it("keeps client chat disabled until /api/models reports a warm model", async () => {
    statusPayload = {
      ...createStatusPayload(),
      is_client: true,
      is_host: false,
      llama_ready: false,
      model_name: "ghost-model",
      hosted_models: [],
      serving_models: [],
    };
    setPath("/chat");
    render(<App />);

    const input = await screen.findByTestId("chat-input");
    await waitFor(() =>
      expect(mockFetch.mock.calls.some((call) => call[0] === "/api/models")).toBe(
        true,
      ),
    );
    expect(input).toBeDisabled();
    expect(input).toHaveAttribute("placeholder", "Waiting for a warm model...");
    expect(screen.getByTestId("chat-send")).toBeDisabled();
  });
});

describe("describeRenderedPagesAsText", () => {
  it("combines page descriptions and preserves failures as placeholders", async () => {
    const onProgress = vi.fn();
    const describe = vi
      .fn<
        (dataUrl: string) => Promise<{
          combinedText: string;
          description: string;
          ocrText: string;
        }>
      >()
      .mockResolvedValueOnce({
        combinedText: "First page OCR",
        description: "First page OCR",
        ocrText: "First page OCR",
      })
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValueOnce({
        combinedText: "",
        description: "",
        ocrText: "",
      });

    const text = await describeRenderedPagesAsText(
      [
        "data:image/png;base64,one",
        "data:image/png;base64,two",
        "data:image/png;base64,three",
      ],
      { describe, onProgress },
    );

    expect(text).toContain("[Page 1]\nFirst page OCR");
    expect(text).toContain("[Page 2]\n[Unable to describe page]");
    expect(text).toContain("[Page 3]\n[Unable to describe page]");
    expect(onProgress).toHaveBeenNthCalledWith(
      1,
      "Describing scanned PDF page 1/3...",
    );
    expect(onProgress).toHaveBeenNthCalledWith(
      2,
      "Describing scanned PDF page 2/3...",
    );
    expect(onProgress).toHaveBeenNthCalledWith(
      3,
      "Describing scanned PDF page 3/3...",
    );
  });
});

describe("describeImageAttachmentForPrompt", () => {
  it("returns image text and summary on success", async () => {
    const describe = vi.fn<typeof describeImageAttachmentForPrompt extends never ? never : any>().mockResolvedValue({
      combinedText: "A cat on a chair",
      description: "A cat on a chair",
      ocrText: "",
    });

    const result = await describeImageAttachmentForPrompt(
      "data:image/png;base64,abc",
      { describe },
    );

    expect(result).toEqual({
      imageDescription: "A cat on a chair",
      extractionSummary: "Described by local vision",
    });
  });

  it("returns a visible warning payload on failure", async () => {
    const describe = vi.fn().mockRejectedValue(new Error("boom"));

    const result = await describeImageAttachmentForPrompt(
      "data:image/png;base64,abc",
      { describe },
    );

    expect(result.imageDescription).toBeUndefined();
    expect(result.extractionSummary).toBe(
      "Image description failed — retry or send placeholder text",
    );
    expect(result.error).toContain("Image description failed: boom");
  });
});

describe("attachmentForMessage", () => {
  it("drops rendered page images once extracted text exists", () => {
    const attachment = attachmentForMessage({
      id: "att-pdf",
      kind: "file",
      dataUrl: "data:application/pdf;base64,abc",
      mimeType: "application/pdf",
      fileName: "scan.pdf",
      status: "pending",
      extractedText: "Recovered text",
      renderedPageImages: ["data:image/png;base64,one"],
      extractionSummary: "1 page described",
    });

    expect(attachment.extractedText).toBe("Recovered text");
    expect(attachment.renderedPageImages).toBeUndefined();
  });
});
