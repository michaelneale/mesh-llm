import { describe, expect, it } from "vitest";

import {
  LIVE_NODE_STATE_LABELS,
  type LiveNodeState,
  type Peer,
  type StatusPayload,
  type WakeableNodeState,
} from "./status-types";

type OptionalKey<T, K extends keyof T> = {} extends Pick<T, K> ? true : false;

const STATUS_PAYLOAD_NODE_STATE_REQUIRED = false satisfies OptionalKey<StatusPayload, "node_state">;
const PEER_STATE_REQUIRED = false satisfies OptionalKey<Peer, "state">;
const WAKEABLE_NODES_OPTIONAL = true satisfies OptionalKey<StatusPayload, "wakeable_nodes">;

describe("status type contracts", () => {
  it("enumerates the allowed live and wakeable node states", () => {
    const liveStates = ["client", "standby", "loading", "serving"] as const satisfies readonly LiveNodeState[];
    const wakeableStates = ["sleeping", "waking"] as const satisfies readonly WakeableNodeState[];

    expect(liveStates).toEqual(["client", "standby", "loading", "serving"]);
    expect(wakeableStates).toEqual(["sleeping", "waking"]);
    expect(LIVE_NODE_STATE_LABELS).toEqual({
      client: "Client",
      standby: "Standby",
      loading: "Loading",
      serving: "Serving",
    });
  });

  it("requires live state on status payloads and peers while leaving wakeable_nodes optional", () => {
    const statusPayload: StatusPayload = {
      node_id: "node-1",
      token: "token-1",
      node_state: "serving",
      node_status: "Serving",
      is_host: true,
      is_client: false,
      llama_ready: true,
      model_name: "Qwen",
      api_port: 3131,
      my_vram_gb: 24,
      model_size_gb: 8,
      peers: [
        {
          id: "peer-1",
          role: "Host",
          state: "standby",
          models: [],
          vram_gb: 16,
        },
      ],
      inflight_requests: 0,
    };

    const peer = {
      id: "peer-2",
      role: "Worker",
      state: "loading",
      models: [],
      vram_gb: 12,
    } satisfies Peer;

    // @ts-expect-error StatusPayload must require node_state
    const missingNodeState: StatusPayload = {
      node_id: "node-1",
      token: "token-1",
      node_status: "Serving",
      is_host: true,
      is_client: false,
      llama_ready: true,
      model_name: "Qwen",
      api_port: 3131,
      my_vram_gb: 24,
      model_size_gb: 8,
      peers: [],
      inflight_requests: 0,
    };

    // @ts-expect-error Peer must require state
    const missingPeerState: Peer = {
      id: "peer-1",
      role: "Host",
      models: [],
      vram_gb: 12,
    };

    expect(statusPayload.node_state).toBe("serving");
    expect(statusPayload.wakeable_nodes).toBeUndefined();
    expect(peer.state).toBe("loading");
    expect(missingNodeState).toBeDefined();
    expect(missingPeerState).toBeDefined();
    expect(STATUS_PAYLOAD_NODE_STATE_REQUIRED).toBe(false);
    expect(PEER_STATE_REQUIRED).toBe(false);
    expect(WAKEABLE_NODES_OPTIONAL).toBe(true);
  });

  it("rejects legacy labels at the type layer", () => {
    // @ts-expect-error legacy labels are not valid live node states
    const idle: LiveNodeState = "Idle";
    // @ts-expect-error split-era labels are not valid live node states
    const splitServing: LiveNodeState = "Serving (split)";
    // @ts-expect-error topology role labels are not valid live node states
    const host: LiveNodeState = "Host";
    // @ts-expect-error wakeable state must not accept live state labels
    const standbyWakeable: WakeableNodeState = "standby";

    expect([idle, splitServing, host, standbyWakeable]).toHaveLength(4);
  });
});
