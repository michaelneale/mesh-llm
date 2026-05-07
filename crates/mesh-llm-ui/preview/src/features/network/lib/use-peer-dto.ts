import { useMemo } from 'react'
import type { ModelSummary, Peer, PeerDTO, PeerHostedModel } from '@/features/app-tabs/types'

function sortHostedModels(names: string[], catalog: ModelSummary[]): PeerHostedModel[] {
  const catalogMap = new Map(catalog.map((m) => [m.name, m]))
  return names
    .map((name): PeerHostedModel => {
      const entry = catalogMap.get(name)
      return { name, paramsB: entry?.paramsB, sizeGB: entry?.sizeGB }
    })
    .sort((a, b) => {
      if ((b.paramsB ?? -1) !== (a.paramsB ?? -1)) {
        return (b.paramsB ?? -1) - (a.paramsB ?? -1)
      }
      if ((b.sizeGB ?? -1) !== (a.sizeGB ?? -1)) {
        return (b.sizeGB ?? -1) - (a.sizeGB ?? -1)
      }
      return a.name.localeCompare(b.name)
    })
}

function toPeerDTO(peer: Peer, catalog: ModelSummary[]): PeerDTO {
  return {
    ...peer,
    hostedModels: sortHostedModels(peer.hostedModels, catalog)
  }
}

export function usePeerDtos(peers: Peer[], catalog: ModelSummary[]): PeerDTO[] {
  return useMemo(() => peers.map((peer) => toPeerDTO(peer, catalog)), [peers, catalog])
}
