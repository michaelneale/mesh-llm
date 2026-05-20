import type { MeshProvider } from '@/features/reserves/lib/reserve-types'

export const MESH_PROVIDERS: readonly MeshProvider[] = [
  {
    id: 'bare-metal',
    name: 'Bare metal',
    kind: 'Co-located · always on',
    icon: 'server',
    description: 'Register owned hosts, racks, or office machines that can join the mesh without a cloud provider API.',
    availability: 'supported',
    defaultName: 'Bare metal reserve',
    defaultRegion: 'rack-01',
    billing: 'owned capacity',
    summary: 'Manually managed bare-metal reserve capacity. No cloud provider API is attached yet.',
    optionFields: [
      {
        id: 'name',
        label: 'Provider name',
        placeholder: 'Example: Lab rack west',
        helper: 'Shown as the reserve vendor row title.'
      },
      {
        id: 'region',
        label: 'Location or rack',
        placeholder: 'Example: rack-01 · sf-site',
        helper: 'Used for placement tags and node location copy.'
      }
    ]
  },
  {
    id: 'digitalocean',
    name: 'Digital Ocean',
    kind: 'Cloud GPU',
    icon: 'cloud',
    description: 'Reserve GPU droplets from DigitalOcean once provider automation is wired in.',
    availability: 'coming-soon',
    disabledReason: 'DigitalOcean provisioning is not supported in this preview yet.',
    defaultName: 'DigitalOcean reserve',
    defaultRegion: 'nyc1',
    billing: 'provider managed',
    summary: 'DigitalOcean reserve provider placeholder.',
    optionFields: []
  },
  {
    id: 'gcp',
    name: 'GCP',
    kind: 'Cloud GPU',
    icon: 'cloud',
    description: 'Attach Google Cloud accelerator pools after GCP support lands.',
    availability: 'coming-soon',
    disabledReason: 'GCP provider support is not enabled yet.',
    defaultName: 'GCP reserve',
    defaultRegion: 'us-central1',
    billing: 'provider managed',
    summary: 'GCP reserve provider placeholder.',
    optionFields: []
  },
  {
    id: 'aws',
    name: 'AWS',
    kind: 'Cloud GPU',
    icon: 'cloud',
    description: 'Use EC2 GPU capacity once AWS reserve provisioning is available.',
    availability: 'coming-soon',
    disabledReason: 'AWS provider support is not enabled yet.',
    defaultName: 'AWS reserve',
    defaultRegion: 'us-east-1',
    billing: 'provider managed',
    summary: 'AWS reserve provider placeholder.',
    optionFields: []
  }
]

export const DEFAULT_MESH_PROVIDER = MESH_PROVIDERS[0]

export function getMeshProvider(providerId: string): MeshProvider {
  return MESH_PROVIDERS.find((provider) => provider.id === providerId) ?? DEFAULT_MESH_PROVIDER
}
