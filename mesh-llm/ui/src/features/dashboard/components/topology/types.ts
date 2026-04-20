import type { CSSProperties, ReactElement } from "react";

import type { ThemeMode } from "../../../../lib/resolved-theme";
import type { TopologyNode } from "../../../app-shell/lib/topology-types";

export type TopologyStatusPayload = {
  model_name?: string | null;
};

export type TopologyThemeMode = ThemeMode;

export type ColorTuple = [number, number, number, number];

export type NodePalette = {
  fill: string;
  fillAlpha: number;
  line: string;
  lineAlpha: number;
};

export type ScenePalette = {
  shellClassName: string;
  chromeClassName: string;
  selectedModelClassName: string;
  tooltipClassName: string;
  tooltipSubtleTextClassName: string;
  tooltipMutedTextClassName: string;
  tooltipDividerClassName: string;
  selfLabelPrimaryClassName: string;
  selfLabelSecondaryClassName: string;
  selfLabelTextShadow: string;
  surfaceGradient: (focalPoint: string) => string;
  nebulaBlendClassName: string;
  nebulaOpacity: number;
  nebulaGradient: string;
  gridOpacity: number;
  gridLineColor: string;
  nodes: {
    client: NodePalette;
    worker: NodePalette;
    active: NodePalette;
    serving: NodePalette;
    self: NodePalette & {
      selectedLine: string;
      selectedLineAlpha: number;
    };
  };
};

export type SelfNodeAccentProps = {
  style: CSSProperties;
  selectedModelMatch: boolean;
};

export type RenderNodeKind = "client" | "worker" | "active" | "serving" | "self";

export type NodeVariantInput = {
  kind: RenderNodeKind;
  node: TopologyNode;
  size: number;
  color: ColorTuple;
  lineColor: ColorTuple;
  pulse: number;
  selectedModelMatch: boolean;
  z: number;
};

export type LineBuilderInput = {
  screenNodes: ScreenNode[];
  centerNodeId?: string;
  highlightedNodeIds: Set<string>;
  devicePixelRatio: number;
  lineTailAlpha: number;
  visiblePairKeys?: Set<string>;
  pairAlphaOverrides?: Map<string, number>;
};

export type LineBuilderOutput = {
  positions: Float32Array;
  colors: Float32Array;
  pairKeys?: string[];
  pairRouteSignatures?: Map<string, string>;
};

export type RenderNode = {
  id: string;
  label: string;
  subtitle: string;
  hostname?: string;
  role: string;
  latencyLabel: string;
  vramLabel: string;
  modelLabel: string;
  gpuLabel: string;
  statusLabel?: string;
  ageSeconds?: number | null;
  x: number;
  y: number;
  size: number;
  color: ColorTuple;
  lineColor: ColorTuple;
  pulse: number;
  selectedModelMatch: boolean;
  z: number;
};

export type RenderVariant = {
  scene: ScenePalette;
  lineFragmentShader: string;
  lineWidthPx: number;
  pointFragmentShader: string;
  lineTailAlpha: number;
  applyLineBlendMode: (gl: WebGLRenderingContext) => void;
  applyPointBlendMode: (gl: WebGLRenderingContext) => void;
  tuneNode: (
    input: NodeVariantInput,
  ) => Pick<RenderNode, "size" | "color" | "lineColor" | "pulse" | "z">;
  buildLines: (input: LineBuilderInput) => LineBuilderOutput;
  SelfNodeAccent: (props: SelfNodeAccentProps) => ReactElement;
};

export type ScreenNode = RenderNode & {
  px: number;
  py: number;
  hitSize: number;
  lineRevealProgress: number;
};

export type EntryAnimation = {
  fromX: number;
  fromY: number;
  control1X: number;
  control1Y: number;
  control2X: number;
  control2Y: number;
  toX: number;
  toY: number;
  normalX: number;
  normalY: number;
  meanderAmplitude: number;
  meanderCycles: number;
  meanderPhase: number;
  startedAt: number;
  hideLinksUntilSettled: boolean;
  isReposition: boolean;
};

export type ExitAnimation = {
  node: RenderNode;
  startedAt: number;
};

export type LineRevealAnimation = {
  startedAt: number;
};

export type UpdateTwinkle = {
  startedAt: number;
};

export type PendingLineTransition = {
  addedNodeIds: Set<string>;
  removedNodeIds: Set<string>;
};

export type LineTransition = {
  snapshotScreenNodes: ScreenNode[];
  stableVisiblePairKeys: Set<string>;
  outgoingPairKeys: Set<string>;
  incomingPairKeys: Set<string>;
  addedNodeIds: Set<string>;
  startedAt: number;
  entryStartedAt: number;
  revealStartedAt: number;
  /** Per-pair starting alpha for outgoing pairs carried over from an interrupted transition. */
  outgoingPairStartAlphas?: Map<string, number>;
  /** Per-pair hop delay in ms for staggered reveal (undefined = no stagger). */
  pairHopDelays?: Map<string, number>;
  /** Maximum hop delay across all pairs, for completion-check bookkeeping. */
  maxHopDelay?: number;
};

export type MeshTopologyDiagramProps = {
  status: TopologyStatusPayload | null;
  nodes: TopologyNode[];
  selectedModel: string;
  themeMode: TopologyThemeMode;
  onOpenNode?: (nodeId: string) => void;
  highlightedNodeId?: string;
  fullscreen?: boolean;
  heightClass?: string;
  containerStyle?: CSSProperties;
};

export type MeshRadarFieldProps = Omit<MeshTopologyDiagramProps, "status"> & {
  status: TopologyStatusPayload;
  fullscreen: boolean;
};
