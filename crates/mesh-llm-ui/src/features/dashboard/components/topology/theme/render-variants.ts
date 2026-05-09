import type { ResolvedTheme } from '@/lib/resolved-theme'

import { buildProximityLines } from '@/features/dashboard/components/topology/render/line-builders'
import {
  DARK_LINE_FRAGMENT_SHADER,
  DARK_POINT_FRAGMENT_SHADER,
  LIGHT_LINE_FRAGMENT_SHADER,
  LIGHT_POINT_FRAGMENT_SHADER
} from '@/features/dashboard/components/topology/render/shaders'
import type { RenderVariant } from '@/features/dashboard/components/topology/types'
import { LightSelfNodeAccent, DarkSelfNodeAccent } from '@/features/dashboard/components/topology/ui/self-node-accents'
import { tuneDarkNodeVariant, tuneLightNodeVariant } from '@/features/dashboard/components/topology/theme/node-variants'
import { SCENE_PALETTES } from '@/features/dashboard/components/topology/theme/scene'

export const RENDER_VARIANTS: Record<ResolvedTheme, RenderVariant> = {
  dark: {
    scene: SCENE_PALETTES.dark,
    lineFragmentShader: DARK_LINE_FRAGMENT_SHADER,
    lineWidthPx: 1.6,
    pointFragmentShader: DARK_POINT_FRAGMENT_SHADER,
    lineTailAlpha: 0.02,
    applyLineBlendMode: (gl) => {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE)
    },
    applyPointBlendMode: (gl) => {
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE)
    },
    tuneNode: tuneDarkNodeVariant,
    buildLines: buildProximityLines,
    SelfNodeAccent: DarkSelfNodeAccent
  },
  light: {
    scene: SCENE_PALETTES.light,
    lineFragmentShader: LIGHT_LINE_FRAGMENT_SHADER,
    lineWidthPx: 1.2,
    pointFragmentShader: LIGHT_POINT_FRAGMENT_SHADER,
    lineTailAlpha: 0.08,
    applyLineBlendMode: (gl) => {
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
    },
    applyPointBlendMode: (gl) => {
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
    },
    tuneNode: tuneLightNodeVariant,
    buildLines: buildProximityLines,
    SelfNodeAccent: LightSelfNodeAccent
  }
}
