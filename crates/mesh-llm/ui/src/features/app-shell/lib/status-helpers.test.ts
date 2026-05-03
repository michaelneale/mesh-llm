import { describe, expect, it } from "vitest";

import {
  displayVramGb,
  formatLiveNodeState,
  gpuInventoryVramGb,
  localRoutableModels,
  meshGpuVram,
  normalizeVramGb,
  overviewVramGb,
  topologyStatusTone,
  topologyStatusTooltip,
} from "./status-helpers";
import type { LiveNodeState, StatusPayload } from "./status-types";

describe("live node state helpers", () => {
  it("formats all supported live node states", () => {
    const cases: Array<{
      state: LiveNodeState;
      label: string;
      tone: "good" | "info" | "warn" | "bad" | "neutral";
      tooltip: string;
    }> = [
      {
        state: "client",
        label: "Client",
        tone: "info",
        tooltip: "Sends requests, but does not contribute VRAM.",
      },
      {
        state: "standby",
        label: "Standby",
        tone: "neutral",
        tooltip: "Connected, but not currently serving a model.",
      },
      {
        state: "loading",
        label: "Loading",
        tone: "warn",
        tooltip: "Initializing model work before it can serve requests.",
      },
      {
        state: "serving",
        label: "Serving",
        tone: "good",
        tooltip: "Actively serving a model.",
      },
    ];

    for (const testCase of cases) {
      expect(formatLiveNodeState(testCase.state)).toBe(testCase.label);
      expect(topologyStatusTone(testCase.state)).toBe(testCase.tone);
      expect(topologyStatusTooltip(testCase.state)).toBe(testCase.tooltip);
    }
  });

  it("handles invalid states gracefully without throwing", () => {
    expect(formatLiveNodeState(undefined as any)).toBe("Unknown");
    expect(formatLiveNodeState(null as any)).toBe("Unknown");
    
    expect(topologyStatusTone(undefined as any)).toBe("neutral");
    expect(topologyStatusTone(null as any)).toBe("neutral");
    
    expect(topologyStatusTooltip(undefined as any)).toBe("Node state unavailable");
    expect(topologyStatusTooltip(null as any)).toBe("Node state unavailable");
  });

  it("uses node_state as the local routable-model source of truth", () => {
    const baseStatus: StatusPayload = {
      node_id: "local-node",
      node_status: "Serving",
      node_state: "serving",
      token: "token",
      is_host: false,
      is_client: true,
      llama_ready: true,
      peers: [],
      model_name: "fallback-model",
      requested_models: [],
      available_models: [],
      serving_models: ["serving-model"],
      hosted_models: ["hosted-model"],
      my_vram_gb: 24,
      api_port: 3131,
      model_size_gb: 0,
      inflight_requests: 0,
      version: "test",
      latest_version: null,
      wakeable_nodes: [],
    };

    expect(localRoutableModels(baseStatus)).toEqual(["hosted-model"]);
    expect(localRoutableModels({ ...baseStatus, node_state: "client", is_client: false })).toEqual(
      [],
    );
  });

  it("prefers physical GPU inventory over effective capacity for VRAM display", () => {
    const physicalVramBytes = 17_094_934_528;

    expect(gpuInventoryVramGb([{ vram_bytes: physicalVramBytes }])).toBeCloseTo(15.92, 2);
    expect(displayVramGb(false, 29.4, [{ vram_bytes: physicalVramBytes }])).toBeCloseTo(
      15.92,
      2,
    );
  });

  it("falls back to capacity VRAM when GPU inventory is absent", () => {
    expect(displayVramGb(false, 29.4, [])).toBe(29.4);
    expect(displayVramGb(false, 29.4, undefined)).toBe(29.4);
    expect(displayVramGb(true, 29.4, [{ vram_bytes: 17_094_934_528 }])).toBe(0);
  });

  it("normalizes overview VRAM through the shared clamp helper", () => {
    expect(normalizeVramGb(null)).toBe(0);
    expect(normalizeVramGb(undefined)).toBe(0);
    expect(normalizeVramGb(-4)).toBe(0);
    expect(normalizeVramGb(12.5)).toBe(12.5);

    expect(overviewVramGb(false, -4)).toBe(normalizeVramGb(-4));
    expect(overviewVramGb(false, 12.5)).toBe(normalizeVramGb(12.5));
    expect(overviewVramGb(true, 12.5)).toBe(0);
  });

  it("uses physical GPU inventory for mesh VRAM totals when available", () => {
    const status: StatusPayload = {
      node_id: "local-node",
      node_status: "Serving",
      node_state: "serving",
      token: "token",
      is_host: true,
      is_client: false,
      llama_ready: true,
      peers: [
        {
          id: "peer-with-inventory",
          role: "Host",
          state: "serving",
          models: [],
          vram_gb: 48,
          gpus: [{ name: "GPU", vram_bytes: 12 * 1024 ** 3 }],
        },
        {
          id: "legacy-peer",
          role: "Host",
          state: "serving",
          models: [],
          vram_gb: 8,
          gpus: [],
        },
      ],
      model_name: "model",
      requested_models: [],
      available_models: [],
      serving_models: ["model"],
      hosted_models: ["model"],
      my_vram_gb: 29.4,
      gpus: [{ name: "RTX 5080", vram_bytes: 16 * 1024 ** 3 }],
      api_port: 3131,
      model_size_gb: 0,
      inflight_requests: 0,
      version: "test",
      latest_version: null,
      wakeable_nodes: [],
    };

    expect(meshGpuVram(status)).toBe(36);
  });
});
