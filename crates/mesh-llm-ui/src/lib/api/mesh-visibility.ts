import type { StatusPayload } from '@/lib/api/types'

type MeshVisibilityFields = Pick<StatusPayload, 'nostr_discovery' | 'publication_state'>

export function isPublicMesh(payload: MeshVisibilityFields): boolean {
  if (payload.publication_state === 'public') return true
  if (payload.publication_state === 'publish_failed') return false
  return payload.nostr_discovery === true
}
