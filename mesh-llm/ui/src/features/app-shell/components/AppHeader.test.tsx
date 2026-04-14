// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { TooltipProvider } from "../../../components/ui/tooltip";
import { AppHeader } from "./AppHeader";

class MockResizeObserver {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

function renderHeader(overrides: Partial<Parameters<typeof AppHeader>[0]> = {}) {
  return render(
    <TooltipProvider>
      <AppHeader
        sections={[
          { key: "dashboard", label: "Network" },
          { key: "chat", label: "Chat" },
        ]}
        section="dashboard"
        setSection={vi.fn()}
        themeMode="auto"
        setThemeMode={vi.fn()}
        statusError={null}
        inviteWithModelCommand="mesh-llm --join invite-token --model GLM-4.7-Flash-Q4_K_M"
        inviteWithModelName="GLM-4.7-Flash-Q4_K_M"
        inviteClientCommand="mesh-llm --client --join invite-token"
        inviteToken="invite-token"
        apiDirectUrl=""
        isPublicMesh={false}
        {...overrides}
      />
    </TooltipProvider>,
  );
}

describe("AppHeader", () => {
  beforeAll(() => {
    Object.defineProperty(window, "ResizeObserver", {
      configurable: true,
      writable: true,
      value: MockResizeObserver,
    });

    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it("shows and copies the OpenCode launcher for public meshes", async () => {
    renderHeader({ isPublicMesh: true });

    fireEvent.click(screen.getByRole("button", { name: "API access" }));

    await screen.findByText("mesh-llm opencode");
    fireEvent.click(screen.getByRole("button", { name: "Copy opencode command" }));

    await waitFor(() =>
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith("mesh-llm opencode"),
    );
    expect(screen.getByText("mesh-llm claude")).toBeInTheDocument();
    expect(screen.getByText("mesh-llm goose")).toBeInTheDocument();
  });

  it("keeps private meshes focused on invite flows without OpenCode", async () => {
    renderHeader({ isPublicMesh: false, inviteToken: "private-token" });

    fireEvent.click(screen.getByRole("button", { name: "API access" }));

    await screen.findByText("mesh-llm claude --join private-token");
    expect(screen.getByText("mesh-llm goose --join private-token")).toBeInTheDocument();
    expect(screen.queryByText(/mesh-llm opencode/i)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Copy opencode command" }),
    ).not.toBeInTheDocument();
  });
});
