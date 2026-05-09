import { useMemo, useState } from "react";

type DetailPanelEntry =
  | { kind: "node"; nodeId: string }
  | { kind: "model"; modelName: string };

type MeshModelLookup = {
  name: string;
};

function detailEntriesEqual(left: DetailPanelEntry | undefined, right: DetailPanelEntry) {
  if (!left) return false;
  if (left.kind !== right.kind) return false;
  if (left.kind === "node" && right.kind === "node") {
    return left.nodeId === right.nodeId;
  }
  if (left.kind === "model" && right.kind === "model") {
    return left.modelName === right.modelName;
  }
  return false;
}

export function useDashboardDetailStack({
  isMeshOverviewFullscreen,
  meshModelByName,
}: {
  isMeshOverviewFullscreen: boolean;
  meshModelByName: Record<string, MeshModelLookup>;
}) {
  const [detailPanelStack, setDetailPanelStack] = useState<DetailPanelEntry[]>([]);

  const activeDetail = useMemo(
    () => detailPanelStack[detailPanelStack.length - 1] ?? null,
    [detailPanelStack],
  );

  function pushDetail(entry: DetailPanelEntry) {
    if (isMeshOverviewFullscreen) return;
    setDetailPanelStack((prev) => {
      const current = prev[prev.length - 1];
      if (detailEntriesEqual(current, entry)) return prev;
      return [...prev, entry];
    });
  }

  function openNodeDetail(nodeId: string) {
    pushDetail({ kind: "node", nodeId });
  }

  function openModelDetail(modelName: string) {
    if (!meshModelByName[modelName]) return;
    pushDetail({ kind: "model", modelName });
  }

  function closeDetailPanel() {
    setDetailPanelStack([]);
  }

  function goBackDetailPanel() {
    setDetailPanelStack((prev) => prev.slice(0, -1));
  }

  return {
    activeDetail,
    closeDetailPanel,
    detailPanelStack,
    goBackDetailPanel,
    openModelDetail,
    openNodeDetail,
  };
}
