import { clamp, mixColor, withAlpha } from "../helpers";
import type { NodeVariantInput } from "../types";

import {
  DARK_HOST_VIOLET,
  DARK_SELF_GLEAM,
  LIGHT_GRAPHITE,
  LIGHT_HOST_VIOLET,
  LIGHT_INK,
  LIGHT_SELF_GLEAM,
} from "./color-tokens";

export function tuneDarkNodeVariant({
  kind,
  node,
  size,
  color: baseColor,
  lineColor,
  pulse,
  z,
}: NodeVariantInput) {
  if (kind === "client") {
    return {
      size: clamp(size + 0.4, 10, 21.5),
      color: withAlpha(baseColor, Math.min(baseColor[3], 0.75)),
      lineColor,
      pulse: pulse * 0.18,
      z,
    };
  }

  if (kind === "self") {
    return {
      size: clamp(size + 0.9, 15, 38),
      color: withAlpha(mixColor(baseColor, DARK_SELF_GLEAM, 0.18), 0.84),
      lineColor,
      pulse: pulse * 0.24,
      z,
    };
  }

  if (node.host) {
    const hostColor = mixColor(
      DARK_HOST_VIOLET,
      baseColor,
      kind === "serving" ? 0.12 : kind === "active" ? 0.18 : 0.24,
    );
    return {
      size: clamp(size + 5.6, 16, 43),
      color: withAlpha(hostColor, kind === "serving" ? 0.8 : 0.74),
      lineColor,
      pulse: pulse * 0.24,
      z: z + 1,
    };
  }

  return {
    size: clamp(
      size + (kind === "serving" ? 0.55 : kind === "active" ? 0.3 : 0.1),
      11,
      33,
    ),
    color: withAlpha(
      mixColor(baseColor, DARK_SELF_GLEAM, kind === "serving" ? 0.06 : 0.03),
      kind === "serving" ? 0.76 : kind === "active" ? 0.72 : 0.68,
    ),
    lineColor,
    pulse: pulse * 0.22,
    z,
  };
}

export function tuneLightNodeVariant({
  kind,
  node,
  size,
  color: baseColor,
  lineColor: baseLineColor,
  pulse,
  selectedModelMatch,
  z,
}: NodeVariantInput) {
  if (kind === "client") {
    return {
      size: clamp(size + 1.2, 10, 21.5),
      color: withAlpha(mixColor(baseColor, LIGHT_INK, 0.1), 0.97),
      lineColor: withAlpha(mixColor(baseLineColor, LIGHT_GRAPHITE, 0.22), 0.2),
      pulse: pulse * 0.14,
      z,
    };
  }

  if (kind === "self") {
    return {
      size: clamp(size + 1.1, 15, 38),
      color: withAlpha(mixColor(baseColor, LIGHT_SELF_GLEAM, 0.14), 0.99),
      lineColor: withAlpha(
        mixColor(
          selectedModelMatch ? baseLineColor : baseColor,
          LIGHT_INK,
          0.42,
        ),
        selectedModelMatch ? 0.44 : 0.3,
      ),
      pulse: pulse * 0.2,
      z,
    };
  }

  if (node.host) {
    const hostColor = mixColor(
      LIGHT_HOST_VIOLET,
      baseColor,
      kind === "serving" ? 0.14 : kind === "active" ? 0.2 : 0.26,
    );
    return {
      size: clamp(size + 6.8, 16, 43),
      color: withAlpha(
        mixColor(hostColor, LIGHT_INK, 0.08),
        kind === "serving" ? 0.95 : 0.88,
      ),
      lineColor: withAlpha(
        mixColor(
          selectedModelMatch ? baseLineColor : baseColor,
          LIGHT_INK,
          0.44,
        ),
        selectedModelMatch ? 0.38 : 0.3,
      ),
      pulse: pulse * 0.22,
      z: z + 1,
    };
  }

  const servingBoost = kind === "serving" ? 1.1 : kind === "active" ? 0.65 : 0.2;
  return {
    size: clamp(
      size + servingBoost - (kind === "serving" ? 0.15 : 0.25),
      11,
      33,
    ),
    color: withAlpha(
      mixColor(
        baseColor,
        kind === "serving" ? LIGHT_INK : LIGHT_GRAPHITE,
        kind === "serving" ? 0.16 : 0.12,
      ),
      kind === "serving" ? 0.9 : kind === "active" ? 0.86 : 0.81,
    ),
    lineColor: withAlpha(
      mixColor(
        selectedModelMatch ? baseLineColor : baseColor,
        LIGHT_GRAPHITE,
        0.42,
      ),
      selectedModelMatch ? 0.36 : 0.28,
    ),
    pulse: pulse * 0.24,
    z,
  };
}
