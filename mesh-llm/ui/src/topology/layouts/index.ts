import { classicTopologyLayout } from "./classic";
import { elkTopologyLayout } from "./elk";
import type { TopologyLayoutDefinition, TopologyLayoutMode } from "./types";

export const TOPOLOGY_LAYOUTS: Record<
  TopologyLayoutMode,
  TopologyLayoutDefinition
> = {
  elk: elkTopologyLayout,
  classic: classicTopologyLayout,
};

export const TOPOLOGY_LAYOUT_OPTIONS: Array<{
  value: TopologyLayoutMode;
  label: string;
}> = [
  { value: "elk", label: elkTopologyLayout.label },
  { value: "classic", label: classicTopologyLayout.label },
];

export function isTopologyLayoutMode(value: string): value is TopologyLayoutMode {
  return value === "elk" || value === "classic";
}
