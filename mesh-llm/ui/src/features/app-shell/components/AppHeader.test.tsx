// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import { TooltipProvider } from "../../../components/ui/tooltip";
import { CommandBarModal } from "./command-bar/CommandBarModal";
import { CommandBarProvider } from "./command-bar/CommandBarProvider";
import type { CommandBarMode } from "./command-bar/command-bar-types";
import { useCommandBar } from "./command-bar/useCommandBar";
import { AppHeader } from "./AppHeader";

class MockResizeObserver {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

type CommandBarItem = {
  id: string;
  name: string;
};

function ModelsIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" {...props}>
      <circle cx="8" cy="8" r="7" />
    </svg>
  );
}

function NodesIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" {...props}>
      <rect x="2" y="2" width="12" height="12" />
    </svg>
  );
}

function createMode(
  id: string,
  label: string,
  source: CommandBarMode<CommandBarItem>["source"],
): CommandBarMode<CommandBarItem> {
  return {
    id,
    label,
    leadingIcon: id === "models" ? ModelsIcon : NodesIcon,
    source,
    getItemKey: (item) => item.id,
    getSearchText: (item) => item.name,
    onSelect: vi.fn(),
  };
}

function renderHeader({
  headerOverrides = {},
  behavior = "distinct",
  modes = [createMode("models", "Models", [{ id: "model-1", name: "Model one" }])],
}: {
  headerOverrides?: Partial<Parameters<typeof AppHeader>[0]>;
  behavior?: "distinct" | "combined";
  modes?: readonly CommandBarMode<CommandBarItem>[];
} = {}) {
  return render(
    <CommandBarProvider>
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
          {...headerOverrides}
        />
        <CommandBarModal
          modes={modes}
          behavior={behavior}
          defaultModeId="models"
          title="Switch models"
          description="Search the mesh model catalog and select a model without leaving the current view."
          placeholder="Search models"
          emptyMessage="No matching models."
        />
      </TooltipProvider>
      <CommandBarStateProbe />
    </CommandBarProvider>,
  );
}

function CommandBarStateProbe() {
  const { activeModeId, isOpen } = useCommandBar();

  return (
    <div
      data-testid="command-bar-state"
      data-active-mode-id={activeModeId ?? ""}
      data-open={isOpen ? "true" : "false"}
    />
  );
}

describe("AppHeader", () => {
  const originalUserAgent = navigator.userAgent;

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
    Object.defineProperty(navigator, "userAgent", {
      configurable: true,
      value: "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0)",
    });
  });

  afterEach(() => {
    Object.defineProperty(navigator, "userAgent", {
      configurable: true,
      value: originalUserAgent,
    });
    cleanup();
  });

  it("shows and copies the OpenCode launcher for public meshes", async () => {
    renderHeader({ headerOverrides: { isPublicMesh: true } });

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
    renderHeader({
      headerOverrides: { isPublicMesh: false, inviteToken: "private-token" },
    });

    fireEvent.click(screen.getByRole("button", { name: "API access" }));

    await screen.findByText("mesh-llm claude --join private-token");
    expect(screen.getByText("mesh-llm goose --join private-token")).toBeInTheDocument();
    expect(screen.queryByText(/mesh-llm opencode/i)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Copy opencode command" }),
    ).not.toBeInTheDocument();
  });

  it("opens the command bar in models mode from the header trigger", () => {
    renderHeader();

    fireEvent.click(screen.getByRole("button", { name: "Models" }));

    expect(screen.getByTestId("command-bar-state")).toHaveAttribute("data-open", "true");
    expect(screen.getByTestId("command-bar-state")).toHaveAttribute(
      "data-active-mode-id",
      "models",
    );
  });

  it("keeps the single-mode model picker concise when there are no results", async () => {
    renderHeader({
      modes: [createMode("models", "Models", [])],
    });

    fireEvent.click(screen.getByRole("button", { name: "Models" }));

    const dialog = await screen.findByRole("dialog", { name: "Switch models" });
    const listbox = screen.getByRole("listbox", { name: "Command bar results" });

    expect(listbox).toHaveAttribute("aria-label", "Command bar results");
    expect(listbox).not.toHaveAttribute("aria-activedescendant");
    expect(screen.getByText("No matching models.")).toBeInTheDocument();
    expect(within(dialog).queryByText("Modes")).not.toBeInTheDocument();
    expect(
      within(dialog).queryByRole("button", { name: /Models Ctrl\+1/i }),
    ).not.toBeInTheDocument();
  });

  it("keeps duplicate combined results distinguishable and updates listbox selection semantics", async () => {
    renderHeader({
      behavior: "combined",
      modes: [
        createMode("models", "Models", [{ id: "model-1", name: "Shared result" }]),
        createMode("nodes", "Nodes", [{ id: "node-1", name: "Shared result" }]),
      ],
    });

    fireEvent.click(screen.getByRole("button", { name: "Models" }));

    const dialog = await screen.findByRole("dialog", { name: "Switch models" });
    const input = screen.getByRole("textbox", { name: "Command bar search" });
    const listbox = screen.getByRole("listbox", { name: "Command bar results" });
    let options = within(listbox).getAllByRole("option");

    expect(within(listbox).getAllByText("Shared result")).toHaveLength(2);
    expect(within(options[0]).getByText("Models")).toBeInTheDocument();
    expect(within(options[1]).getByText("Nodes")).toBeInTheDocument();
    expect(options[0]).toHaveAttribute("aria-selected", "true");
    expect(options[1]).toHaveAttribute("aria-selected", "false");
    expect(options[0].id).not.toBe(options[1].id);
    expect(listbox).toHaveAttribute("aria-label", "Command bar results");
    expect(listbox).toHaveAttribute("aria-activedescendant", options[0].id);

    fireEvent.keyDown(input, { key: "ArrowDown" });

    options = within(listbox).getAllByRole("option");
    expect(options[0]).toHaveAttribute("aria-selected", "false");
    expect(options[1]).toHaveAttribute("aria-selected", "true");
    expect(listbox).toHaveAttribute("aria-activedescendant", options[1].id);
    expect(within(dialog).queryByRole("button", { name: /Models Ctrl\+1/i })).not.toBeInTheDocument();
  });

  it("shows explicit loading and error copy for async command-bar sources", async () => {
    let rejectRequest: ((reason?: unknown) => void) | undefined;
    const asyncSource = vi.fn().mockImplementation(
      () => new Promise<readonly CommandBarItem[]>((_, reject) => {
        rejectRequest = reject;
      }),
    );

    renderHeader({
      modes: [createMode("models", "Models", asyncSource)],
    });

    fireEvent.click(screen.getByRole("button", { name: "Models" }));

    expect(screen.getByText("Loading results")).toBeInTheDocument();

    rejectRequest?.(new Error("Catalog offline"));

    await waitFor(() => {
      expect(screen.getByText("Could not load results")).toBeInTheDocument();
    });
    expect(screen.getByText("Catalog offline")).toBeInTheDocument();
  });

  it("restores focus to the opener for both button and keyboard launches", async () => {
    renderHeader();

    const modelsButton = screen.getByRole("button", { name: "Models" });
    modelsButton.focus();
    fireEvent.click(modelsButton);

    let input = await screen.findByRole("textbox", { name: "Command bar search" });
    await waitFor(() => expect(input).toHaveFocus());
    fireEvent.keyDown(input, { key: "Escape" });

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "Switch models" })).not.toBeInTheDocument();
    });
    expect(modelsButton).toHaveFocus();

    const networkLink = screen.getByRole("link", { name: "Network" });
    networkLink.focus();
    fireEvent.keyDown(window, { key: "k", metaKey: true });

    input = await screen.findByRole("textbox", { name: "Command bar search" });
    await waitFor(() => expect(input).toHaveFocus());
    fireEvent.keyDown(input, { key: "Escape" });

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "Switch models" })).not.toBeInTheDocument();
    });
    expect(networkLink).toHaveFocus();
  });
});
