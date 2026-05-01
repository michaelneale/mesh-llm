import type { ResolvedTheme } from "../../../../../lib/resolved-theme";

import type { ScenePalette } from "../types";

export const SCENE_PALETTES: Record<ResolvedTheme, ScenePalette> = {
  dark: {
    shellClassName:
      "border-white/10 bg-[#06111f] shadow-[inset_0_1px_0_rgba(255,255,255,0.06),0_20px_60px_rgba(0,0,0,0.28)]",
    chromeClassName:
      "border-white/10 bg-slate-950/72 text-slate-100 hover:bg-slate-900/80",
    selectedModelClassName:
      "border-amber-300/20 bg-amber-300/10 text-amber-100/90",
    tooltipClassName:
      "border-white/10 bg-slate-950/88 text-white shadow-2xl shadow-black/50",
    tooltipSubtleTextClassName: "text-slate-300",
    tooltipMutedTextClassName: "text-slate-500",
    tooltipDividerClassName: "border-white/10 text-slate-400",
    selfLabelPrimaryClassName: "text-white",
    selfLabelSecondaryClassName: "text-slate-200",
    selfLabelTextShadow: "0 1px 10px rgba(0,0,0,0.72)",
    surfaceGradient: (focalPoint) => `
      radial-gradient(circle at ${focalPoint}, rgba(59,130,246,0.12), transparent 22%),
      radial-gradient(circle at 72% 34%, rgba(250,204,21,0.10), transparent 26%),
      radial-gradient(circle at 30% 58%, rgba(56,189,248,0.08), transparent 32%),
      radial-gradient(circle at ${focalPoint}, rgba(255,255,255,0.035) 0, transparent 38%),
      linear-gradient(180deg, rgba(12,24,40,0.98), rgba(6,17,31,1))
    `,
    nebulaBlendClassName: "mix-blend-screen",
    nebulaOpacity: 0.48,
    nebulaGradient: `
      radial-gradient(40% 28% at 28% 60%, rgba(56,189,248,0.11), transparent 72%),
      radial-gradient(34% 26% at 72% 32%, rgba(250,204,21,0.1), transparent 74%),
      radial-gradient(46% 32% at 50% 54%, rgba(99,102,241,0.08), transparent 78%)
    `,
    gridOpacity: 0.1,
    gridLineColor: "rgba(255,255,255,0.04)",
    nodes: {
      client: {
        fill: "#94a3b8",
        fillAlpha: 0.84,
        line: "#7c8ba1",
        lineAlpha: 0,
      },
      worker: {
        fill: "#7dd3fc",
        fillAlpha: 0.96,
        line: "#38bdf8",
        lineAlpha: 0,
      },
      active: {
        fill: "#4ade80",
        fillAlpha: 0.95,
        line: "#4ade80",
        lineAlpha: 0,
      },
      serving: {
        fill: "#f8fafc",
        fillAlpha: 1,
        line: "#facc15",
        lineAlpha: 0.22,
      },
      self: {
        fill: "#60a5fa",
        fillAlpha: 1,
        line: "#93c5fd",
        lineAlpha: 0.16,
        selectedLine: "#facc15",
        selectedLineAlpha: 0.34,
      },
    },
  },
  light: {
    shellClassName:
      "border-slate-300/80 bg-slate-50 shadow-[inset_0_1px_0_rgba(255,255,255,0.94),inset_0_0_0_1px_rgba(148,163,184,0.10),0_18px_42px_rgba(15,23,42,0.08)]",
    chromeClassName:
      "border-slate-300/70 bg-white/84 text-slate-800 shadow-sm shadow-slate-200/70 hover:bg-white/96",
    selectedModelClassName:
      "border-amber-400/80 bg-amber-100 text-amber-950 shadow-sm shadow-amber-200/70",
    tooltipClassName:
      "border-slate-400/90 bg-white/98 text-slate-950 shadow-2xl shadow-slate-400/45",
    tooltipSubtleTextClassName: "text-slate-800",
    tooltipMutedTextClassName: "text-slate-600",
    tooltipDividerClassName: "border-slate-300/90 text-slate-600",
    selfLabelPrimaryClassName: "text-slate-950",
    selfLabelSecondaryClassName: "text-slate-800",
    selfLabelTextShadow:
      "0 1px 0 rgba(255,255,255,0.9), 0 0 10px rgba(255,255,255,0.7)",
    surfaceGradient: (focalPoint) => `
      linear-gradient(180deg, rgba(255,255,255,0.96), rgba(241,245,249,0.94)),
      linear-gradient(135deg, rgba(30,41,59,0.045), transparent 28%, transparent 72%, rgba(30,41,59,0.03)),
      radial-gradient(circle at ${focalPoint}, rgba(255,255,255,0.92) 0, rgba(255,255,255,0.4) 16%, rgba(148,163,184,0.08) 36%, transparent 58%),
      radial-gradient(circle at 20% 16%, rgba(148,163,184,0.16), transparent 36%),
      radial-gradient(circle at 82% 22%, rgba(125,211,252,0.12), transparent 30%)
    `,
    nebulaBlendClassName: "mix-blend-multiply",
    nebulaOpacity: 0.24,
    nebulaGradient: `
      radial-gradient(46% 34% at 28% 60%, rgba(51,65,85,0.12), transparent 72%),
      radial-gradient(36% 28% at 72% 32%, rgba(14,116,144,0.1), transparent 74%),
      radial-gradient(50% 34% at 50% 54%, rgba(245,158,11,0.06), transparent 80%)
    `,
    gridOpacity: 0.34,
    gridLineColor: "rgba(51,65,85,0.105)",
    nodes: {
      client: {
        fill: "#38495d",
        fillAlpha: 1,
        line: "#475569",
        lineAlpha: 0.2,
      },
      worker: {
        fill: "#0f7184",
        fillAlpha: 1,
        line: "#155e75",
        lineAlpha: 0.28,
      },
      active: {
        fill: "#0f7a63",
        fillAlpha: 1,
        line: "#115e59",
        lineAlpha: 0.3,
      },
      serving: {
        fill: "#0f172a",
        fillAlpha: 1,
        line: "#a16207",
        lineAlpha: 0.38,
      },
      self: {
        fill: "#2e89ea",
        fillAlpha: 1,
        line: "#256ec9",
        lineAlpha: 0.34,
        selectedLine: "#a16207",
        selectedLineAlpha: 0.42,
      },
    },
  },
};
