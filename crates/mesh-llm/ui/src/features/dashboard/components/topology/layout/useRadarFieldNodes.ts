import { useEffect, useMemo, useRef } from "react";

import { formatLatency, formatLiveNodeState, shortName } from "../../../../app-shell/lib/status-helpers";
import type { TopologyNode } from "../../../../app-shell/lib/topology-types";

import { color, hashString, nodeUpdateSignature, TAU } from "../helpers";
import type { RenderNode, RenderVariant } from "../types";
import {
  distributeLatencyBand,
  distributePerimeterClients,
  nodeSize,
  reduceTopologyCrossings,
  resolveNodeOverlap,
} from "./distribution";
import type { TopologyAngularPlacement } from "./distribution";

function nonIdleModels(node: TopologyNode) {
  return node.servingModels.filter((model) => model && model !== "(idle)");
}

function computeLabel(node: TopologyNode) {
  if (node.gpus && node.gpus.length > 0) {
    return `${node.gpus.length} GPU${node.gpus.length === 1 ? "" : "s"}`;
  }
  return node.isSoc ? "SoC" : "GPU unknown";
}

function modelLabel(models: string[]) {
  return models.length > 0 ? models.map(shortName).join(", ") : "idle";
}

function normalizeAngle(angle: number) {
  const wrapped = angle % TAU;
  return wrapped < 0 ? wrapped + TAU : wrapped;
}

export function useRadarFieldNodes(
  nodes: TopologyNode[],
  selectedModel: string,
  fallbackModel: string,
  renderVariant: RenderVariant,
) {
  const previousLayoutRef = useRef<Map<string, RenderNode>>(new Map());
  const previousSignaturesRef = useRef<Map<string, string>>(new Map());
  const previousContextRef = useRef<{
    focusModel: string;
    renderVariant: RenderVariant | null;
  }>({
    focusModel: "",
    renderVariant: null,
  });

  const renderNodes = useMemo<RenderNode[]>(() => {
    if (!nodes.length) return [];

    const palette = renderVariant.scene.nodes;
    const focusModel = selectedModel || fallbackModel || "";
    const currentSignatures = new Map(nodes.map((node) => [node.id, nodeUpdateSignature(node)]));
    const previousLayout = previousLayoutRef.current;
    const previousSignatures = previousSignaturesRef.current;
    const previousContext = previousContextRef.current;
    const selfNode = nodes.find((node) => node.self) ?? nodes[0];
    const others = nodes.filter((node) => node.id !== selfNode.id);

    const selectedServing = others.filter(
      (node) =>
        !node.client &&
        !!focusModel &&
        node.servingModels.some((model) => model === focusModel),
    );
    const selectedIds = new Set(selectedServing.map((node) => node.id));
    const serving = others.filter(
      (node) =>
        !node.client &&
        !selectedIds.has(node.id) &&
        node.servingModels.some((model) => model && model !== "(idle)"),
    );
    const clients = others.filter((node) => node.client);
    const workers = others.filter(
      (node) =>
        !node.client &&
        !selectedIds.has(node.id) &&
        !serving.some((entry) => entry.id === node.id),
    );

    const latencyNodes = others.filter(
      (node) => !node.client && node.latencyMs != null && Number.isFinite(node.latencyMs),
    );
    const minLatency =
      latencyNodes.length > 0
        ? Math.min(...latencyNodes.map((node) => Number(node.latencyMs)))
        : 0;
    const maxLatency =
      latencyNodes.length > 0
        ? Math.max(...latencyNodes.map((node) => Number(node.latencyMs)))
        : 1;
    const addedIds = new Set([...currentSignatures.keys()].filter((id) => !previousSignatures.has(id)));
    const removedIds = new Set([...previousSignatures.keys()].filter((id) => !currentSignatures.has(id)));
    const changedExistingIds = new Set(
      [...currentSignatures.keys()].filter((id) => {
        const previousSignature = previousSignatures.get(id);
        return previousSignature != null && previousSignature !== currentSignatures.get(id);
      }),
    );
    const shouldStabilizeExistingNodes =
      previousSignatures.size > 0 &&
      addedIds.size > 0 &&
      removedIds.size === 0 &&
      changedExistingIds.size === 0 &&
      previousContext.focusModel === focusModel &&
      previousContext.renderVariant === renderVariant;
    const shouldStabilizeForRemoval =
      previousSignatures.size > 0 &&
      removedIds.size > 0 &&
      addedIds.size === 0 &&
      changedExistingIds.size === 0 &&
      previousContext.focusModel === focusModel &&
      previousContext.renderVariant === renderVariant;

    const servingNodes = distributeLatencyBand(
      selectedServing,
      minLatency,
      maxLatency,
      0,
      Math.PI * 2,
      0.08,
      0.18,
      0.07,
      0.15,
      3,
      -0.1,
      0.42,
    );
    const workerNodes = distributeLatencyBand(
      workers,
      minLatency,
      maxLatency,
      0,
      Math.PI * 2,
      0.18,
      0.34,
      0.15,
      0.28,
      3,
      0.08,
      0.7,
    );
    const clientNodes = distributePerimeterClients(clients);
    const activeNodes = distributeLatencyBand(
      serving,
      minLatency,
      maxLatency,
      0,
      Math.PI * 2,
      0.19,
      0.32,
      0.15,
      0.25,
      2,
      0.14,
      0.6,
    );

    const selfSelectedModelMatch =
      !!focusModel && selfNode.servingModels.some((model) => model === focusModel);
    const selfLayoutNode = {
      id: selfNode.id,
      x: 0.5,
      y: 0.52,
      role: selfNode.client ? "Client" : selfNode.host ? "Host" : "Node",
      selectedModelMatch: selfSelectedModelMatch,
    };
    const reusePreviousPlacement = (shouldStabilizeExistingNodes || shouldStabilizeForRemoval)
      ? (nodeId: string) => previousLayout.get(nodeId)
      : () => undefined;
    const angleForPosition = (x: number, y: number) =>
      normalizeAngle(Math.atan2(y - selfLayoutNode.y, x - selfLayoutNode.x));
    const preparedBands: TopologyAngularPlacement<TopologyNode>[][] = [
      clientNodes.map((entry) => {
        const previousNode = reusePreviousPlacement(entry.node.id);
        return {
          ...entry,
          x: previousNode?.x ?? entry.x,
          y: previousNode?.y ?? entry.y,
          angle: previousNode ? angleForPosition(previousNode.x, previousNode.y) : entry.angle,
          role: "Client",
          selectedModelMatch: false,
          locked: previousNode != null,
        };
      }),
      workerNodes.map((entry) => {
        const previousNode = reusePreviousPlacement(entry.node.id);
        return {
          ...entry,
          x: previousNode?.x ?? entry.x,
          y: previousNode?.y ?? entry.y,
          angle: previousNode ? angleForPosition(previousNode.x, previousNode.y) : entry.angle,
          role: entry.node.host ? "Host" : "Worker",
          selectedModelMatch: false,
          locked: previousNode != null,
        };
      }),
      activeNodes.map((entry) => {
        const previousNode = reusePreviousPlacement(entry.node.id);
        return {
          ...entry,
          x: previousNode?.x ?? entry.x,
          y: previousNode?.y ?? entry.y,
          angle: previousNode ? angleForPosition(previousNode.x, previousNode.y) : entry.angle,
          role: entry.node.host ? "Host" : "Serving",
          selectedModelMatch: false,
          locked: previousNode != null,
        };
      }),
      servingNodes.map((entry) => {
        const previousNode = reusePreviousPlacement(entry.node.id);
        return {
          ...entry,
          x: previousNode?.x ?? entry.x,
          y: previousNode?.y ?? entry.y,
          angle: previousNode ? angleForPosition(previousNode.x, previousNode.y) : entry.angle,
          role: entry.node.host ? "Host" : "Serving",
          selectedModelMatch: true,
          locked: previousNode != null,
        };
      }),
    ];
    // When only nodes were removed, all survivors are locked at their previous
    // positions. Crossing reduction would be a no-op but still costs O(N²) for
    // the pair-plan computation — skip it entirely.
    const [optimizedClientNodes, optimizedWorkerNodes, optimizedActiveNodes, optimizedServingNodes] =
      shouldStabilizeForRemoval
        ? preparedBands
        : reduceTopologyCrossings(preparedBands, selfLayoutNode, selfNode.id);

    const output: RenderNode[] = [];

    for (const { node, x, y } of optimizedClientNodes) {
      const selectedModelMatch = false;
      const tuned = renderVariant.tuneNode({
        kind: "client",
        node,
        size: nodeSize(node, 1.5),
        color: color(palette.client.fill, palette.client.fillAlpha),
        lineColor: color(palette.client.line, palette.client.lineAlpha),
        pulse: 0.28 + hashString(`${node.id}:pulse`) * 0.24,
        selectedModelMatch,
        z: 1,
      });
      output.push({
        id: node.id,
        label: node.hostname || node.id,
        subtitle: "Client",
        hostname: node.hostname,
        role: "Client",
        latencyLabel: formatLatency(node.latencyMs),
        vramLabel: "n/a",
        modelLabel: "API-only",
        gpuLabel: "No GPU",
        x,
        y,
        size: tuned.size,
        color: tuned.color,
        lineColor: tuned.lineColor,
        pulse: tuned.pulse,
        selectedModelMatch,
        z: tuned.z,
      });
    }

    for (const { node, x, y, latencyNorm } of optimizedWorkerNodes) {
      const models = nonIdleModels(node);
      const latencyValue = latencyNorm ?? 0.45;
      const glowAlpha = 0.96 - latencyValue * 0.22;
      const selectedModelMatch = false;
      const tuned = renderVariant.tuneNode({
        kind: "worker",
        node,
        size: nodeSize(node, 3),
        color: color(palette.worker.fill, Math.min(glowAlpha, palette.worker.fillAlpha)),
        lineColor: color(palette.worker.line, palette.worker.lineAlpha),
        pulse: 0.42 + (1 - latencyValue) * 0.34 + hashString(`${node.id}:pulse`) * 0.36,
        selectedModelMatch,
        z: 2,
      });
      output.push({
        id: node.id,
        label: node.hostname || node.id,
        subtitle: formatLiveNodeState(node.state),
        hostname: node.hostname,
        role: node.host ? "Host" : "Worker",
        latencyLabel: formatLatency(node.latencyMs),
        vramLabel: `${Math.max(0, node.vram).toFixed(1)} GB`,
        modelLabel: modelLabel(models),
        gpuLabel: computeLabel(node),
        x,
        y,
        size: tuned.size,
        color: tuned.color,
        lineColor: tuned.lineColor,
        pulse: tuned.pulse,
        selectedModelMatch,
        z: tuned.z,
      });
    }

    for (const { node, x, y, latencyNorm } of optimizedActiveNodes) {
      const models = nonIdleModels(node);
      const latencyValue = latencyNorm ?? 0.45;
      const glowAlpha = 0.95 - latencyValue * 0.18;
      const selectedModelMatch = false;
      const tuned = renderVariant.tuneNode({
        kind: "active",
        node,
        size: nodeSize(node, 5),
        color: color(palette.active.fill, Math.min(glowAlpha, palette.active.fillAlpha)),
        lineColor: color(palette.active.line, palette.active.lineAlpha),
        pulse: 0.68 + (1 - latencyValue) * 0.4 + hashString(`${node.id}:pulse`) * 0.34,
        selectedModelMatch,
        z: 3,
      });
      output.push({
        id: node.id,
        label: node.hostname || node.id,
        subtitle: formatLiveNodeState(node.state),
        hostname: node.hostname,
        role: node.host ? "Host" : "Serving",
        latencyLabel: formatLatency(node.latencyMs),
        vramLabel: `${Math.max(0, node.vram).toFixed(1)} GB`,
        modelLabel: modelLabel(models),
        gpuLabel: computeLabel(node),
        x,
        y,
        size: tuned.size,
        color: tuned.color,
        lineColor: tuned.lineColor,
        pulse: tuned.pulse,
        selectedModelMatch,
        z: tuned.z,
      });
    }

    for (const { node, x, y, latencyNorm } of optimizedServingNodes) {
      const models = nonIdleModels(node);
      const latencyValue = latencyNorm ?? 0.45;
      const glowAlpha = 1 - latencyValue * 0.12;
      const selectedModelMatch = true;
      const tuned = renderVariant.tuneNode({
        kind: "serving",
        node,
        size: nodeSize(node, 9),
        color: color(palette.serving.fill, Math.min(glowAlpha, palette.serving.fillAlpha)),
        lineColor: color(palette.serving.line, palette.serving.lineAlpha),
        pulse: 1.02 + (1 - latencyValue) * 0.46 + hashString(`${node.id}:pulse`) * 0.28,
        selectedModelMatch,
        z: 4,
      });
      output.push({
        id: node.id,
        label: node.hostname || node.id,
        subtitle: focusModel ? shortName(focusModel) : formatLiveNodeState(node.state),
        hostname: node.hostname,
        role: node.host ? "Host" : "Serving",
        latencyLabel: formatLatency(node.latencyMs),
        vramLabel: `${Math.max(0, node.vram).toFixed(1)} GB`,
        modelLabel: modelLabel(models),
        gpuLabel: computeLabel(node),
        x,
        y,
        size: tuned.size,
        color: tuned.color,
        lineColor: tuned.lineColor,
        pulse: tuned.pulse,
        selectedModelMatch,
        z: tuned.z,
      });
    }

    const selfTuned = renderVariant.tuneNode({
      kind: "self",
      node: selfNode,
      size: nodeSize(selfNode, 15),
      color: color(palette.self.fill, palette.self.fillAlpha),
      lineColor:
        focusModel && selfNode.servingModels.includes(focusModel)
          ? color(palette.self.selectedLine, palette.self.selectedLineAlpha)
          : color(palette.self.line, palette.self.lineAlpha),
      pulse: 1.4,
      selectedModelMatch: selfSelectedModelMatch,
      z: 5,
    });
    const selfModels = nonIdleModels(selfNode);
    const selfRenderNode: RenderNode = {
      id: selfNode.id,
      label: selfNode.hostname || (selfNode.client ? "this client" : "this node"),
      subtitle: selfNode.client ? "This client" : "This node",
      hostname: selfNode.hostname,
      role: selfNode.client ? "Client" : selfNode.host ? "Host" : "Node",
      latencyLabel: "local",
      vramLabel: selfNode.client ? "n/a" : `${Math.max(0, selfNode.vram).toFixed(1)} GB`,
      modelLabel: selfNode.client ? "API-only" : modelLabel(selfModels),
      gpuLabel: selfNode.client ? "No GPU" : computeLabel(selfNode),
      x: 0.5,
      y: 0.52,
      size: selfTuned.size,
      color: selfTuned.color,
      lineColor: selfTuned.lineColor,
      pulse: selfTuned.pulse,
      selectedModelMatch: selfSelectedModelMatch,
      z: selfTuned.z,
    };

    let resolvedOutput: RenderNode[];
    if (shouldStabilizeForRemoval) {
      resolvedOutput = output;
    } else if (shouldStabilizeExistingNodes) {
      const stableExistingNodes = output.filter((node) => previousLayout.has(node.id));
      const newNodes = output.filter((node) => !previousLayout.has(node.id));
      const resolvedNewNodes = resolveNodeOverlap(newNodes, [...stableExistingNodes, selfRenderNode]);
      resolvedOutput = [...stableExistingNodes, ...resolvedNewNodes];
    } else {
      resolvedOutput = resolveNodeOverlap(output, [selfRenderNode]);
    }

    resolvedOutput.push(selfRenderNode);

    return resolvedOutput.sort(
      (left, right) => left.z - right.z || left.size - right.size,
    );
  }, [fallbackModel, nodes, renderVariant, selectedModel]);

  useEffect(() => {
    previousLayoutRef.current = new Map(renderNodes.map((node) => [node.id, node]));
    previousSignaturesRef.current = new Map(nodes.map((node) => [node.id, nodeUpdateSignature(node)]));
    previousContextRef.current = {
      focusModel: selectedModel || fallbackModel || "",
      renderVariant,
    };
  }, [fallbackModel, nodes, renderNodes, renderVariant, selectedModel]);

  return renderNodes;
}
