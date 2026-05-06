import type { ResolvedTheme } from '@/features/app-tabs/types'

export type MeshVizGridColors = readonly [string, string, string]
export type MeshVizNodeColors = readonly [string, string, string, string]

export type MeshVizDotColorScheme = {
  label: string
  colors: MeshVizGridColors
  nodeColors: MeshVizNodeColors
}

export const MESH_VIZ_DOT_COLOR_SCHEME_COUNT = 3

export const MESH_VIZ_DOT_COLOR_SCHEMES: Record<
  ResolvedTheme,
  readonly [MeshVizDotColorScheme, MeshVizDotColorScheme, MeshVizDotColorScheme]
> = {
  dark: [
    {
      label: 'Ash signal',
      colors: ['oklch(0.64 0.025 252 / 9%)', 'oklch(0.72 0.115 220 / 13%)', 'oklch(0.76 0.105 72 / 7%)'],
      nodeColors: ['oklch(0.64 0.025 252)', 'oklch(0.72 0.115 220)', 'oklch(0.76 0.105 72)', 'oklch(0.66 0.22 28)']
    },
    {
      label: 'Cool trace',
      colors: ['oklch(0.62 0.024 252 / 8%)', 'oklch(0.74 0.12 190 / 12%)', 'oklch(0.72 0.13 275 / 10%)'],
      nodeColors: ['oklch(0.62 0.024 252)', 'oklch(0.74 0.12 190)', 'oklch(0.72 0.13 275)', 'oklch(0.80 0.12 28)']
    },
    {
      label: 'Warm trace',
      colors: ['oklch(0.62 0.024 252 / 8%)', 'oklch(0.78 0.125 74 / 10%)', 'oklch(0.70 0.12 260 / 12%)'],
      nodeColors: ['oklch(0.62 0.024 252)', 'oklch(0.78 0.125 74)', 'oklch(0.70 0.12 260)', 'oklch(0.76 0.12 155)']
    }
  ],
  light: [
    {
      label: 'Paper signal',
      colors: ['oklch(0.52 0.022 252 / 12%)', 'oklch(0.54 0.12 220 / 12%)', 'oklch(0.58 0.18 28 / 9%)'],
      nodeColors: ['oklch(0.68 0.018 252)', 'oklch(0.54 0.12 220)', 'oklch(0.58 0.18 28)', 'oklch(0.48 0.01 252)']
    },
    {
      label: 'Field trace',
      colors: ['oklch(0.55 0.02 252 / 11%)', 'oklch(0.53 0.13 145 / 13%)', 'oklch(0.50 0.12 265 / 10%)'],
      nodeColors: ['oklch(0.68 0.018 252)', 'oklch(0.53 0.13 145)', 'oklch(0.50 0.12 265)', 'oklch(0.48 0.01 252)']
    },
    {
      label: 'Amber trace',
      colors: ['oklch(0.55 0.02 252 / 11%)', 'oklch(0.56 0.13 74 / 11%)', 'oklch(0.50 0.12 245 / 9%)'],
      nodeColors: ['oklch(0.68 0.018 252)', 'oklch(0.56 0.13 74)', 'oklch(0.50 0.12 245)', 'oklch(0.48 0.01 252)']
    }
  ]
}

export function meshVizDotColorSchemeAtIndex(theme: ResolvedTheme, index: number): MeshVizDotColorScheme {
  const schemes = MESH_VIZ_DOT_COLOR_SCHEMES[theme]
  const normalizedIndex = ((index % schemes.length) + schemes.length) % schemes.length

  return schemes[normalizedIndex]
}

export function nextMeshVizDotColorSchemeIndex(index: number): number {
  return (index + 1) % MESH_VIZ_DOT_COLOR_SCHEME_COUNT
}

export function themeFromDocument(): ResolvedTheme {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'
}
