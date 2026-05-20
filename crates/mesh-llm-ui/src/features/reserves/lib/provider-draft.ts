import { DEFAULT_MESH_PROVIDER } from '@/features/reserves/lib/mesh-providers'
import type { MeshProviderId } from '@/features/reserves/lib/reserve-types'

export type ProviderDraft = {
  name: string
  providerId: MeshProviderId
  region: string
}

export const DEFAULT_PROVIDER_DRAFT: ProviderDraft = {
  providerId: DEFAULT_MESH_PROVIDER.id,
  name: '',
  region: ''
}
