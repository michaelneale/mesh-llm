// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { TooltipProvider } from "../../../components/ui/tooltip";
import type { StatusPayload } from "../../app-shell/lib/status-types";
import { DashboardPage } from "./DashboardPage";

afterEach(() => {
  cleanup();
});

describe("WakeableCapacity", () => {
  it("renders sleeping and waking inventory outside live topology and peer rows", () => {
    renderDashboard(
      buildStatus({
        wakeable_nodes: [
          {
            logical_id: "vast-a100-1",
            state: "sleeping",
            models: ["Qwen2.5-72B-Instruct"],
            vram_gb: 80,
            provider: "Vast",
          },
          {
            logical_id: "runpod-h100-2",
            state: "waking",
            models: ["DeepSeek-R1", "Qwen3-32B"],
            vram_gb: 94,
            wake_eta_secs: 420,
          },
        ],
      }),
    );

    const section = screen.getByTestId("wakeable-capacity-section");
    expect(within(section).getByText("Wakeable Capacity")).toBeInTheDocument();
    expect(within(section).getByText("Sleeping")).toBeInTheDocument();
    expect(within(section).getByText("Waking")).toBeInTheDocument();
    expect(within(section).getByText("vast-a100-1")).toBeInTheDocument();
    expect(within(section).getByText("runpod-h100-2")).toBeInTheDocument();
    expect(within(section).getByText("Qwen2.5-72B-Instruct")).toBeInTheDocument();
    expect(within(section).getByText("DeepSeek-R1")).toBeInTheDocument();
    expect(within(section).getByText("Qwen3-32B")).toBeInTheDocument();
    expect(within(section).getByText("80.0 GB")).toBeInTheDocument();
    expect(within(section).getByText("94.0 GB")).toBeInTheDocument();
    expect(within(section).getByText("Vast")).toBeInTheDocument();
    expect(within(section).getByText("7 min")).toBeInTheDocument();

    expect(screen.getByText("No host or worker nodes visible yet.")).toBeInTheDocument();
    expect(screen.getByText("No peers connected")).toBeInTheDocument();
  });

  it("hides the section when wakeable_nodes is an empty array", () => {
    renderDashboard(buildStatus({ wakeable_nodes: [] }));

    expect(screen.queryByTestId("wakeable-capacity-section")).not.toBeInTheDocument();
    expect(screen.queryByText("Wakeable Capacity")).not.toBeInTheDocument();
  });

  it("hides the section when wakeable_nodes is absent", () => {
    renderDashboard(buildStatus());

    expect(screen.queryByTestId("wakeable-capacity-section")).not.toBeInTheDocument();
  });
});

function renderDashboard(status: StatusPayload) {
  return render(
    <TooltipProvider>
      <DashboardPage
        status={status}
        meshModels={[]}
        modelsLoading={false}
        topologyNodes={[]}
        selectedModel="all"
        meshModelByName={{}}
        themeMode="dark"
        isPublicMesh={false}
        inviteToken="invite-token"
        isLocalhost
      />
    </TooltipProvider>,
  );
}

function buildStatus(overrides: Partial<StatusPayload> = {}): StatusPayload {
  return {
    node_id: "node-1",
    token: "invite-token",
    node_state: "serving",
    node_status: "Serving",
    is_host: true,
    is_client: false,
    llama_ready: true,
    model_name: "Qwen2.5-32B",
    models: ["Qwen2.5-32B"],
    available_models: ["Qwen2.5-32B"],
    requested_models: [],
    serving_models: ["Qwen2.5-32B"],
    hosted_models: ["Qwen2.5-32B"],
    api_port: 3131,
    my_vram_gb: 24,
    model_size_gb: 16,
    peers: [],
    inflight_requests: 0,
    ...overrides,
  };
}
