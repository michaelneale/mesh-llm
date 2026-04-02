import { getAssignmentId, getSplitGroupId } from "../pages/config/configReducer";
import { clampResizeBoundaryStart } from "./splitBounds";
import type { MeshConfig, ModelAssignment, ModelSplit } from "../types/config";

export type ResizeSplitBoundaryParams = {
  nodeId: string;
  leftAssignmentId: string;
  rightAssignmentId: string;
  boundaryStart: number;
};

export type ResizeSplitBoundaryResult = {
  config: MeshConfig;
  leftAssignmentId: string;
  rightAssignmentId: string;
};

export type MoveSplitAssignmentParams = {
  sourceNodeId: string;
  targetNodeId: string;
  assignmentId: string;
  advertisedModelsByNode: Map<string, Set<string>>;
  advertisedModelKeysByNodeAndName: Map<string, Map<string, Set<string>>>;
};

export type MoveSplitAssignmentResult =
  | { ok: true; config: MeshConfig; assignmentId: string }
  | { ok: false; config: MeshConfig; error: string };

export type RecombineSplitGroupParams = {
  groupId: string;
  targetNodeId: string;
};

export type RecombineSplitGroupResult =
  | { ok: true; config: MeshConfig; assignmentId: string }
  | { ok: false; config: MeshConfig; error: string };

export type RemoveAssignmentResult = {
  config: MeshConfig;
  replacementAssignmentId: string | null;
  replacementNodeId: string | null;
};

export function splitLayerCount(split: ModelSplit) {
  return split.end - split.start;
}

export function findAssignmentNodeId(
  config: MeshConfig,
  assignmentId: string,
): string | null {
  for (const node of config.nodes) {
    if (node.models.some((model) => getAssignmentId(model) === assignmentId)) {
      return node.node_id;
    }
  }
  return null;
}

export function collectSelectedAssignmentIds(
  config: MeshConfig,
  assignmentId: string | null,
): string[] {
  if (!assignmentId) return [];

  for (const node of config.nodes) {
    const selected = node.models.find(
      (model) => getAssignmentId(model) === assignmentId,
    );
    if (!selected) continue;

    const groupId = getSplitGroupId(selected);
    if (!groupId) return [assignmentId];

    return config.nodes.flatMap((entry) =>
      entry.models
        .filter((model) => getSplitGroupId(model) === groupId)
        .map((model) => getAssignmentId(model)),
    );
  }

  return [];
}

function cloneConfig(config: MeshConfig): MeshConfig {
  return {
    ...config,
    nodes: config.nodes.map((node) => ({
      ...node,
      models: node.models.map((model) => ({
        ...model,
        split: model.split ? { ...model.split } : undefined,
      })),
    })),
  };
}

function findNodeIndex(config: MeshConfig, nodeId: string) {
  return config.nodes.findIndex((node) => node.node_id === nodeId);
}

function validateSplitGroupCoverage(
  config: MeshConfig,
  groupId: string,
): boolean {
  const segments = config.nodes
    .flatMap((node) => node.models)
    .filter((model) => getSplitGroupId(model) === groupId)
    .map((model) => model.split)
    .filter((split): split is ModelSplit => split != null)
    .sort((a, b) => a.start - b.start);

  if (segments.length === 0) return false;

  const total = segments[0].total;
  let expectedStart = 0;

  for (const segment of segments) {
    if (segment.total !== total) return false;
    if (segment.start !== expectedStart) return false;
    if (segment.end <= segment.start) return false;
    expectedStart = segment.end;
  }

  return expectedStart === total;
}

function insertSplitAssignment(
  models: ModelAssignment[],
  assignment: ModelAssignment,
): ModelAssignment[] {
  const groupId = getSplitGroupId(assignment);
  if (!groupId || !assignment.split) {
    return [...models, assignment];
  }

  const firstGreaterIndex = models.findIndex(
    (model) =>
      getSplitGroupId(model) === groupId &&
      model.split &&
      model.split.start > assignment.split!.start,
  );
  if (firstGreaterIndex >= 0) {
    return [
      ...models.slice(0, firstGreaterIndex),
      assignment,
      ...models.slice(firstGreaterIndex),
    ];
  }

  const groupedIndexes = models
    .map((model, index) => ({ model, index }))
    .filter(({ model }) => getSplitGroupId(model) === groupId)
    .map(({ index }) => index);
  const lastGroupIndex = groupedIndexes[groupedIndexes.length - 1];

  if (lastGroupIndex != null) {
    return [
      ...models.slice(0, lastGroupIndex + 1),
      assignment,
      ...models.slice(lastGroupIndex + 1),
    ];
  }

  return [...models, assignment];
}

export function resizeSplitBoundaryInConfig(
  config: MeshConfig,
  params: ResizeSplitBoundaryParams,
): ResizeSplitBoundaryResult | null {
  const nextConfig = cloneConfig(config);
  const nodeIndex = findNodeIndex(nextConfig, params.nodeId);
  if (nodeIndex < 0) return null;

  const node = nextConfig.nodes[nodeIndex];
  const leftIndex = node.models.findIndex(
    (model) => getAssignmentId(model) === params.leftAssignmentId,
  );
  const rightIndex = node.models.findIndex(
    (model) => getAssignmentId(model) === params.rightAssignmentId,
  );
  if (leftIndex < 0 || rightIndex < 0) return null;

  const left = node.models[leftIndex];
  const right = node.models[rightIndex];
  const groupId = getSplitGroupId(left);
  if (
    !left.split ||
    !right.split ||
    !groupId ||
    groupId !== getSplitGroupId(right)
  ) {
    return null;
  }

  const boundaryStart = clampResizeBoundaryStart(left.split, right.split, params.boundaryStart);

  const updatedLeft: ModelAssignment = {
    ...left,
    split: { ...left.split, end: boundaryStart },
  };
  const updatedRight: ModelAssignment = {
    ...right,
    split: { ...right.split, start: boundaryStart },
  };

  node.models[leftIndex] = updatedLeft;
  node.models[rightIndex] = updatedRight;

  if (!validateSplitGroupCoverage(nextConfig, groupId)) {
    return null;
  }

  return {
    config: nextConfig,
    leftAssignmentId: getAssignmentId(updatedLeft),
    rightAssignmentId: getAssignmentId(updatedRight),
  };
}

export function moveSplitAssignmentToNode(
  config: MeshConfig,
  params: MoveSplitAssignmentParams,
): MoveSplitAssignmentResult {
  if (params.sourceNodeId === params.targetNodeId) {
    return { ok: true, config, assignmentId: params.assignmentId };
  }

  const nextConfig = cloneConfig(config);
  const sourceNodeIndex = findNodeIndex(nextConfig, params.sourceNodeId);
  if (sourceNodeIndex < 0) {
    return {
      ok: false,
      config,
      error: "The source node is no longer available.",
    };
  }

  const sourceNode = nextConfig.nodes[sourceNodeIndex];
  const sourceModelIndex = sourceNode.models.findIndex(
    (model) => getAssignmentId(model) === params.assignmentId,
  );
  if (sourceModelIndex < 0) {
    return {
      ok: false,
      config,
      error: "The selected split block could not be found.",
    };
  }

  const assignment = sourceNode.models[sourceModelIndex];
  const groupId = getSplitGroupId(assignment);
  if (!assignment.split || !groupId) {
    return {
      ok: false,
      config,
      error: "Only split blocks can be moved between nodes.",
    };
  }
  const assignmentSplit = assignment.split;

  const advertisedModels = params.advertisedModelsByNode.get(
    params.targetNodeId,
  );
  if (!advertisedModels || !advertisedModels.has(assignment.name)) {
    return {
      ok: false,
      config,
      error: `Destination node does not advertise ${assignment.name}.`,
    };
  }

  const targetNodeModelKeys = params.advertisedModelKeysByNodeAndName.get(
    params.targetNodeId,
  );
  const targetAdvertisedKeys = targetNodeModelKeys?.get(assignment.name);
  if (
    !assignment.model_key ||
    !targetAdvertisedKeys?.has(assignment.model_key)
  ) {
    return {
      ok: false,
      config,
      error: `Destination node does not advertise matching scan metadata for ${assignment.name}.`,
    };
  }

  let targetNodeIndex = findNodeIndex(nextConfig, params.targetNodeId);
  if (targetNodeIndex < 0) {
    nextConfig.nodes.push({ node_id: params.targetNodeId, models: [] });
    targetNodeIndex = nextConfig.nodes.length - 1;
  }

  const targetNode = nextConfig.nodes[targetNodeIndex];
  if (
    targetNode.models.some(
      (model) => model.name === assignment.name && !model.split,
    )
  ) {
    return {
      ok: false,
      config,
      error: `Destination node already has a full-model assignment for ${assignment.name}.`,
    };
  }

  if (
    targetNode.models.some(
      (model) =>
        model.name === assignment.name &&
        model.split &&
        (model.model_key !== assignment.model_key ||
          model.split.total !== assignmentSplit.total),
    )
  ) {
    return {
      ok: false,
      config,
      error: `Destination node already has an incompatible split group for ${assignment.name}.`,
    };
  }

  sourceNode.models.splice(sourceModelIndex, 1);
  targetNode.models = insertSplitAssignment(targetNode.models, assignment);

  if (!validateSplitGroupCoverage(nextConfig, groupId)) {
    return {
      ok: false,
      config,
      error: `Moving ${assignment.name} would break contiguous layer coverage.`,
    };
  }

  return {
    ok: true,
    config: nextConfig,
    assignmentId: getAssignmentId(assignment),
  };
}

export function recombineSplitGroupInConfig(
  config: MeshConfig,
  params: RecombineSplitGroupParams,
): RecombineSplitGroupResult {
  const groupEntries = config.nodes.flatMap((node) =>
    node.models
      .filter((model) => getSplitGroupId(model) === params.groupId)
      .map((model) => ({ nodeId: node.node_id, model })),
  );

  if (groupEntries.length === 0) {
    return {
      ok: false,
      config,
      error: "The selected split group no longer exists.",
    };
  }

  const [{ model: firstModel }] = groupEntries;
  if (groupEntries.some(({ nodeId }) => nodeId !== params.targetNodeId)) {
    return {
      ok: false,
      config,
      error: `Move every split block for ${firstModel.name} onto the same node before recombining.`,
    };
  }

  if (!validateSplitGroupCoverage(config, params.groupId)) {
    return {
      ok: false,
      config,
      error: `${firstModel.name} is missing split segments, so it cannot be recombined yet.`,
    };
  }

  const targetNodeIndex = findNodeIndex(config, params.targetNodeId);
  if (targetNodeIndex < 0) {
    return {
      ok: false,
      config,
      error: "The target node is no longer available.",
    };
  }

  const nextConfig = cloneConfig(config);
  const targetNode = nextConfig.nodes[targetNodeIndex];
  const modelsWithoutGroup = targetNode.models.filter(
    (model) => getSplitGroupId(model) !== params.groupId,
  );
  const mergedAssignment: ModelAssignment = {
    name: firstModel.name,
    model_key: firstModel.model_key,
    ctx_size: firstModel.ctx_size,
    moe_experts: firstModel.moe_experts,
  };

  targetNode.models = [...modelsWithoutGroup, mergedAssignment];

  return {
    ok: true,
    config: nextConfig,
    assignmentId: getAssignmentId(mergedAssignment),
  };
}

export function removeAssignmentFromConfig(
  config: MeshConfig,
  params: { nodeId: string; assignmentId: string },
): RemoveAssignmentResult {
  const nextConfig = cloneConfig(config);
  const sourceNodeIndex = findNodeIndex(nextConfig, params.nodeId);
  if (sourceNodeIndex < 0) {
    return {
      config,
      replacementAssignmentId: null,
      replacementNodeId: null,
    };
  }

  const sourceNode = nextConfig.nodes[sourceNodeIndex];
  const sourceAssignment = sourceNode.models.find(
    (model) => getAssignmentId(model) === params.assignmentId,
  );

  if (!sourceAssignment) {
    return {
      config,
      replacementAssignmentId: null,
      replacementNodeId: null,
    };
  }

  const groupId = getSplitGroupId(sourceAssignment);
  if (groupId) {
    sourceNode.models = sourceNode.models.filter(
      (model) => getAssignmentId(model) !== params.assignmentId,
    );

    const remainingSplitAssignments = nextConfig.nodes.flatMap((node) =>
      node.models
        .filter((model) => getSplitGroupId(model) === groupId)
        .map((model) => ({ nodeId: node.node_id, model })),
    );

    if (remainingSplitAssignments.length === 1) {
      const [{ nodeId, model }] = remainingSplitAssignments;
      const replacementAssignment: ModelAssignment = {
        name: model.name,
        model_key: model.model_key,
        ctx_size: model.ctx_size,
        moe_experts: model.moe_experts,
      };
      const replacementNodeIndex = findNodeIndex(nextConfig, nodeId);
      if (replacementNodeIndex >= 0) {
        const replacementNode = nextConfig.nodes[replacementNodeIndex];
        replacementNode.models = replacementNode.models.map((entry) =>
          getAssignmentId(entry) === getAssignmentId(model)
            ? replacementAssignment
            : entry,
        );

        return {
          config: nextConfig,
          replacementAssignmentId: getAssignmentId(replacementAssignment),
          replacementNodeId: nodeId,
        };
      }
    }

    return {
      config: nextConfig,
      replacementAssignmentId: null,
      replacementNodeId: null,
    };
  }

  sourceNode.models = sourceNode.models.filter(
    (model) => getAssignmentId(model) !== params.assignmentId,
  );
  return {
    config: nextConfig,
    replacementAssignmentId: null,
    replacementNodeId: null,
  };
}
