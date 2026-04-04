import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ChatPage } from "./App";

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
