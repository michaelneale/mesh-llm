import type { ResolvedTheme } from "../../../../../lib/resolved-theme";

import { buildProximityLines } from "../render/line-builders";
import {
  DARK_LINE_FRAGMENT_SHADER,
  DARK_POINT_FRAGMENT_SHADER,
  LIGHT_LINE_FRAGMENT_SHADER,
  LIGHT_POINT_FRAGMENT_SHADER,
} from "../render/shaders";
import type { RenderVariant } from "../types";
import { LightSelfNodeAccent, DarkSelfNodeAccent } from "../ui/self-node-accents";
import { tuneDarkNodeVariant, tuneLightNodeVariant } from "./node-variants";
import { SCENE_PALETTES } from "./scene";

export const RENDER_VARIANTS: Record<ResolvedTheme, RenderVariant> = {
  dark: {
    scene: SCENE_PALETTES.dark,
    lineFragmentShader: DARK_LINE_FRAGMENT_SHADER,
    lineWidthPx: 1.6,
    pointFragmentShader: DARK_POINT_FRAGMENT_SHADER,
    lineTailAlpha: 0.02,
    applyLineBlendMode: (gl) => {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    },
    applyPointBlendMode: (gl) => {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    },
    tuneNode: tuneDarkNodeVariant,
    buildLines: buildProximityLines,
    SelfNodeAccent: DarkSelfNodeAccent,
  },
  light: {
    scene: SCENE_PALETTES.light,
    lineFragmentShader: LIGHT_LINE_FRAGMENT_SHADER,
    lineWidthPx: 1.2,
    pointFragmentShader: LIGHT_POINT_FRAGMENT_SHADER,
    lineTailAlpha: 0.08,
    applyLineBlendMode: (gl) => {
      gl.blendFuncSeparate(
        gl.SRC_ALPHA,
        gl.ONE_MINUS_SRC_ALPHA,
        gl.ONE,
        gl.ONE_MINUS_SRC_ALPHA,
      );
    },
    applyPointBlendMode: (gl) => {
      gl.blendFuncSeparate(
        gl.SRC_ALPHA,
        gl.ONE_MINUS_SRC_ALPHA,
        gl.ONE,
        gl.ONE_MINUS_SRC_ALPHA,
      );
    },
    tuneNode: tuneLightNodeVariant,
    buildLines: buildProximityLines,
    SelfNodeAccent: LightSelfNodeAccent,
  },
};
