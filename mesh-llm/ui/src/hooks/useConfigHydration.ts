import { useEffect, useRef } from "react";

import type { OwnedNode } from "./useOwnedNodes";
import type { MeshConfig, ModelAssignment, PlacementMode } from "../types/config";
import type { ConfigAction } from "../pages/config/configReducer";

const STATUS_ONLY_MODEL_NAMES = new Set(["(idle)", "(client)", "(standby)"]);

/**
 * Merge runtime serving state into the authored config so the config page
 * shows what's actually running, not just what was last saved to TOML.
 */
function mergeRuntimeIntoConfig(
  saved: MeshConfig,
  ownedNodes: OwnedNode[],
): MeshConfig {
  const savedLookup = new Map(saved.nodes.map((n) => [n.node_id, n]));
  const nodes = ownedNodes.map((node) => {
    const savedNode = savedLookup.get(node.id);
    const runtimeModels = node.models.filter(
      (m) => !STATUS_ONLY_MODEL_NAMES.has(m),
    );

    if (runtimeModels.length === 0) {
      // Node is idle / standby — keep saved config intact (user may have
      // assignments queued that haven't started yet).
      return savedNode ?? { node_id: node.id, hostname: undefined, placement_mode: 'pooled' as PlacementMode, models: [] };
    }

    const runtimeNameSet = new Set(runtimeModels);
    const savedModels = savedNode?.models ?? [];

    // Keep all saved assignments whose model name appears in runtime
    // (preserves split blocks, ctx_size, etc.) and add stubs for runtime
    // models with no saved counterpart.
    const keptFromSaved = savedModels.filter((m) => runtimeNameSet.has(m.name));
    const savedNameSet = new Set(savedModels.map((m) => m.name));
    const stubs: ModelAssignment[] = runtimeModels
      .filter((name) => !savedNameSet.has(name))
      .map((name) => ({ name, ctx_size: 4096 }));

    return {
      node_id: node.id,
      hostname: savedNode?.hostname,
      placement_mode: savedNode?.placement_mode,
      models: [...keptFromSaved, ...stubs],
    };
  });

  return { ...saved, nodes };
}

export type UseConfigHydrationParams = {
  dispatch: React.Dispatch<ConfigAction>;
  isConfigLoading: boolean;
  ownedNodes: OwnedNode[];
  config: MeshConfig;
  isDirty: boolean;
  setSavedConfig: (config: MeshConfig) => void;
};

/**
 * Seeds the config from runtime state once both the saved TOML and runtime
 * status are available. Re-seeds when the set of peer IDs changes, but skips
 * re-seed when the config has unsaved edits (isDirty) to avoid overwriting
 * in-progress user work.
 */
export function useConfigHydration({
  dispatch,
  isConfigLoading,
  ownedNodes,
  config,
  isDirty,
  setSavedConfig,
}: UseConfigHydrationParams): void {
  const hasSeededRef = useRef(false);
  const prevPeerIdsRef = useRef<string>('');

  useEffect(() => {
    if (isConfigLoading || ownedNodes.length === 0) return;

    const currentPeerIds = ownedNodes.map((n) => n.id).sort().join(',');
    const peersChanged = currentPeerIds !== prevPeerIdsRef.current;
    prevPeerIdsRef.current = currentPeerIds;

    if (!hasSeededRef.current || (peersChanged && !isDirty)) {
      hasSeededRef.current = true;
      const merged = mergeRuntimeIntoConfig(config, ownedNodes);
      dispatch({ type: "SET_CONFIG", config: merged });
      setSavedConfig(merged);
    }
  }, [isConfigLoading, ownedNodes, config, isDirty, dispatch, setSavedConfig]);
}
