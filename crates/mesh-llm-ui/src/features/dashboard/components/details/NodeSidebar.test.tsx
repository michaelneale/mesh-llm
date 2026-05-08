// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TooltipProvider } from "../../../../components/ui/tooltip";
import { Sheet, SheetContent } from "../../../../components/ui/sheet";
import type { LlamaRuntimePayload } from "../../../app-shell/lib/status-types";
import { NodeSidebar } from "./NodeSidebar";

afterEach(() => {
  cleanup();
});

describe("NodeSidebar llama runtime", () => {
  it("renders live metric samples and slot activity for the local node", () => {
    render(
      <TooltipProvider>
        <Sheet open>
          <SheetContent>
            <NodeSidebar
              node={{
                id: "local-node",
                title: "perseus.local",
                self: true,
                state: "serving",
                role: "Host",
                latencyLabel: "local",
                vramGb: 38.6,
                vramSharePct: 100,
                gpus: [],
                hostedModels: [],
                hotModels: [],
                servingModels: [],
                requestedModels: [],
                availableModels: [],
                version: "0.65.0-rc2",
                latestVersion: null,
                llamaReady: true,
                apiPort: 9337,
                inflightRequests: 0,
                owner: { status: "unsigned", verified: false },
                privacyLimited: false,
              }}
              meshModelByName={{}}
              llamaRuntime={buildRuntime()}
              onOpenModel={vi.fn()}
            />
          </SheetContent>
        </Sheet>
      </TooltipProvider>,
    );

    expect(screen.getByText("Llama runtime")).toBeInTheDocument();
    expect(screen.getByText("Metrics • Live")).toBeInTheDocument();
    expect(screen.getByText("Slots • Live")).toBeInTheDocument();
    expect(screen.getByText("1/2 slots busy")).toBeInTheDocument();
    expect(screen.getByText("requests processing")).toBeInTheDocument();
    expect(screen.getByText("tokens predicted seconds")).toBeInTheDocument();
    expect(screen.getByText("Slot context map")).toBeInTheDocument();
    expect(screen.getByText("Available")).toBeInTheDocument();
    expect(screen.getByText("Active")).toBeInTheDocument();
    expect(screen.getByLabelText("Llama slot context map. 1 of 2 slots active.")).toBeInTheDocument();
    expect(screen.getByLabelText("#0 · Active · context 16384")).toBeInTheDocument();
    expect(screen.getByLabelText("#1 · Available · context 16384")).toBeInTheDocument();
  });

  it("handles omitted empty runtime collections from the API", () => {
    renderNodeSidebar({
      metrics: {
        status: "unavailable",
        error: "metrics endpoint unavailable",
      },
      slots: {
        status: "unavailable",
        error: "slots endpoint unavailable",
      },
    });

    expect(screen.getByText("Metrics • Unavailable")).toBeInTheDocument();
    expect(screen.getByText("Slots • Unavailable")).toBeInTheDocument();
    expect(screen.getByText("0/0 slots busy")).toBeInTheDocument();
    expect(screen.getByText("Metrics: metrics endpoint unavailable")).toBeInTheDocument();
    expect(screen.getByText("No llama.cpp metric samples reported yet.")).toBeInTheDocument();
  });

  it("keeps metrics and slots endpoint failures independent", () => {
    const runtime = buildRuntime();
    renderNodeSidebar({
      ...runtime,
      slots: {
        ...runtime.slots,
        status: "error",
        last_attempt_unix_ms: 1_700_000_002_500,
        last_success_unix_ms: 1_700_000_000_000,
        error: "slots timeout",
      },
    });

    expect(screen.getByText("Metrics • Live")).toBeInTheDocument();
    expect(screen.getByText("Slots • Stale")).toBeInTheDocument();
    expect(screen.getByText("Slots: slots timeout")).toBeInTheDocument();
    expect(screen.queryByText("Metrics: slots timeout")).not.toBeInTheDocument();
  });
});

function renderNodeSidebar(llamaRuntime: LlamaRuntimePayload) {
  return render(
    <TooltipProvider>
      <Sheet open>
        <SheetContent>
          <NodeSidebar
            node={{
              id: "local-node",
              title: "perseus.local",
              self: true,
              state: "serving",
              role: "Host",
              latencyLabel: "local",
              vramGb: 38.6,
              vramSharePct: 100,
              gpus: [],
              hostedModels: [],
              hotModels: [],
              servingModels: [],
              requestedModels: [],
              availableModels: [],
              version: "0.65.0-rc2",
              latestVersion: null,
              llamaReady: true,
              apiPort: 9337,
              inflightRequests: 0,
              owner: { status: "unsigned", verified: false },
              privacyLimited: false,
            }}
            meshModelByName={{}}
            llamaRuntime={llamaRuntime}
            onOpenModel={vi.fn()}
          />
        </SheetContent>
      </Sheet>
    </TooltipProvider>,
  );
}

function buildRuntime(): LlamaRuntimePayload {
  return {
    metrics: {
      status: "ready",
      last_attempt_unix_ms: 1_700_000_000_000,
      last_success_unix_ms: 1_700_000_000_000,
      samples: [
        {
          name: "llamacpp:requests_processing",
          labels: {},
          value: 1,
        },
        {
          name: "llamacpp:tokens_predicted_seconds",
          labels: {},
          value: 75.5,
        },
      ],
    },
    slots: {
      status: "ready",
      last_attempt_unix_ms: 1_700_000_000_000,
      last_success_unix_ms: 1_700_000_000_000,
      slots: [
        {
          id: 0,
          n_ctx: 16_384,
          is_processing: true,
          speculative: false,
        },
        {
          id: 1,
          n_ctx: 16_384,
          is_processing: false,
          speculative: false,
        },
      ],
    },
    items: {
      metrics: [
        {
          name: "llamacpp:requests_processing",
          labels: {},
          value: 1,
        },
        {
          name: "llamacpp:tokens_predicted_seconds",
          labels: {},
          value: 75.5,
        },
      ],
      slots: [
        {
          index: 0,
          id: 0,
          n_ctx: 16_384,
          is_processing: true,
        },
        {
          index: 1,
          id: 1,
          n_ctx: 16_384,
          is_processing: false,
        },
      ],
      slots_total: 2,
      slots_busy: 1,
    },
  };
}
