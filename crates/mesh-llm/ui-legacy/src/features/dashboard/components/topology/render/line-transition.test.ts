import { describe, expect, it } from "vitest";

import { narrowPairDiffToChangedComponent } from "./line-transition";

describe("narrowPairDiffToChangedComponent", () => {
  it("filters out unrelated symmetric-diff edges when a node joins", () => {
    const result = narrowPairDiffToChangedComponent({
      previousPairKeys: ["anchor::peer", "other-a::other-b"],
      currentPairKeys: ["anchor::new-node", "new-node::peer", "other-c::other-d"],
      addedNodeIds: ["new-node"],
      removedNodeIds: [],
    });

    expect([...result.outgoingPairKeys]).toEqual(["anchor::peer"]);
    expect([...result.incomingPairKeys].sort()).toEqual([
      "anchor::new-node",
      "new-node::peer",
    ]);
  });

  it("keeps join transitions local to the added-node neighborhood", () => {
    const result = narrowPairDiffToChangedComponent({
      previousPairKeys: [
        "anchor::peer",
        "peer::worker",
        "worker::remote",
        "stable-a::stable-b",
      ],
      currentPairKeys: [
        "anchor::new-node",
        "new-node::peer",
        "peer::worker-2",
        "worker::remote-2",
        "stable-a::stable-b",
      ],
      addedNodeIds: ["new-node"],
      removedNodeIds: [],
    });

    expect([...result.outgoingPairKeys].sort()).toEqual(["anchor::peer"]);
    expect([...result.incomingPairKeys].sort()).toEqual([
      "anchor::new-node",
      "new-node::peer",
    ]);
  });

  it("handles removed nodes by keeping only the affected changed component", () => {
    const result = narrowPairDiffToChangedComponent({
      previousPairKeys: ["anchor::peer", "peer::removed-node", "x::y"],
      currentPairKeys: ["anchor::peer", "peer::replacement", "z::w"],
      addedNodeIds: [],
      removedNodeIds: ["removed-node"],
    });

    expect([...result.outgoingPairKeys]).toEqual(["peer::removed-node"]);
    expect([...result.incomingPairKeys]).toEqual(["peer::replacement"]);
  });

  it("keeps removal transitions local to the removed-node neighborhood", () => {
    const result = narrowPairDiffToChangedComponent({
      previousPairKeys: ["peer::removed-node", "stable-a::stable-b"],
      currentPairKeys: [
        "peer::replacement",
        "replacement::worker-2",
        "worker-2::remote-2",
        "stable-a::stable-b",
      ],
      addedNodeIds: [],
      removedNodeIds: ["removed-node"],
    });

    expect([...result.outgoingPairKeys]).toEqual(["peer::removed-node"]);
    expect([...result.incomingPairKeys]).toEqual(["peer::replacement"]);
  });

  it("treats same-endpoint reroutes around a changed blocker as outgoing and incoming", () => {
    const result = narrowPairDiffToChangedComponent({
      previousPairKeys: ["anchor::peer", "stable-a::stable-b"],
      currentPairKeys: ["anchor::peer", "stable-a::stable-b"],
      previousPairRouteSignatures: new Map([
        ["anchor::peer", JSON.stringify({ mode: "straight", blockerIds: [] })],
        ["stable-a::stable-b", JSON.stringify({ mode: "straight", blockerIds: [] })],
      ]),
      currentPairRouteSignatures: new Map([
        [
          "anchor::peer",
          JSON.stringify({
            mode: "detour",
            axis: "y",
            side: "up",
            blockerIds: ["new-node"],
          }),
        ],
        ["stable-a::stable-b", JSON.stringify({ mode: "straight", blockerIds: [] })],
      ]),
      addedNodeIds: ["new-node"],
      removedNodeIds: [],
    });

    expect([...result.outgoingPairKeys]).toEqual(["anchor::peer"]);
    expect([...result.incomingPairKeys]).toEqual(["anchor::peer"]);
  });
});
