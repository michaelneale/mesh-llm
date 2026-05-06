export type PlaygroundArea =
  | 'shell-controls'
  | 'data-display'
  | 'chat-components'
  | 'configuration-controls'
  | 'tokens-foundations'
  | 'feature-flags'
  | 'meshviz-perf'

export type PlaygroundAreaDefinition = {
  value: PlaygroundArea
  label: string
  description: string
}

export const PLAYGROUND_AREAS: PlaygroundAreaDefinition[] = [
  { value: 'shell-controls', label: 'Shell controls', description: 'connect, picks, compact actions' },
  { value: 'data-display', label: 'Data display', description: 'status, peers, model catalog' },
  { value: 'chat-components', label: 'Chat components', description: 'sidebar, rows, composer' },
  { value: 'configuration-controls', label: 'Configuration controls', description: 'placement, VRAM, cards, TOML' },
  { value: 'tokens-foundations', label: 'Tokens and foundations', description: 'type, controls, machine strings' },
  { value: 'feature-flags', label: 'Feature flags', description: 'rollout switches, local overrides' },
  { value: 'meshviz-perf', label: 'MeshViz 200', description: 'full-width topology stress scene' }
]
