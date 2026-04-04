import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

import { App, ChatPage } from "./App";

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
    selectedModelVision: true,
    selectedModelAudio: true,
    selectedModelMultimodal: true,
    composerError: null,
    setComposerError: vi.fn(),
    attachmentSendIssue: null,
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

const modelsPayload = { mesh_models: [] };
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
      return Promise.resolve(createResponse(createStatusPayload()));
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
});

describe("App routing and status", () => {
  it("desktop unknown path fallback resolves to dashboard behavior", async () => {
    setPath("/unknown-path");
    render(<App />);

    const networkLink = await screen.findByRole("link", { name: "Network" });
    expect(networkLink).toHaveAttribute("aria-current", "page");
    await waitFor(() => expect(window.location.pathname).toBe("/dashboard"));
  });

  it("/dashboard route renders without redirecting to /config", async () => {
    setPath("/dashboard");
    render(<App />);

    const networkLink = await screen.findByRole("link", { name: "Network" });
    expect(networkLink).toHaveAttribute("aria-current", "page");
    await waitFor(() => expect(window.location.pathname).toBe("/dashboard"));
  });

  it("/chat route renders chat section content", async () => {
    setPath("/chat");
    render(<App />);

    const chatLink = await screen.findByRole("link", { name: "Chat" });
    expect(chatLink).toHaveAttribute("aria-current", "page");
    await screen.findByRole("button", { name: /New chat/i });
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
});
