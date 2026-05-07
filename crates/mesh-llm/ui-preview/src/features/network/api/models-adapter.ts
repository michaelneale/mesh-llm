import type { MeshModelRaw } from '@/lib/api/types'
import type { ModelSummary } from '@/features/app-tabs/types'

function formatSize(sizeGB: number | undefined): string {
  if (sizeGB == null) return 'Unknown'

  if (sizeGB >= 1) {
    return `${sizeGB.toFixed(1)}B`
  }
  return `${(sizeGB * 1000).toFixed(0)}M`
}

function formatContext(contextLength: number | undefined): string {
  if (contextLength == null) return 'Unknown'

  const k = Math.round(contextLength / 1000)
  return `${k}K`
}

function mapModelStatus(status: 'warm' | 'cold'): ModelSummary['status'] {
  if (status === 'warm') return 'warm'
  return 'offline'
}

export function adaptModelsToSummary(models: MeshModelRaw[]): ModelSummary[] {
  return models.map((model) => ({
    name: model.name,
    family: model.family ?? model.name.split('/')[0] ?? 'unknown',
    size: formatSize(model.size_gb),
    context: formatContext(model.context_length),
    status: mapModelStatus(model.status),
    tags: model.tags ?? [],
    nodeCount: model.node_count,
    fullId: model.name,
    paramsB: model.params_b,
    paramsLabel: model.params_b != null ? `${model.params_b}B` : undefined,
    quant: model.quantization,
    sizeGB: model.size_gb,
    diskGB: model.disk_gb,
    ctxMaxK: model.context_length == null ? undefined : Math.round(model.context_length / 1000),
    moe: model.capabilities?.moe ?? model.moe ?? false,
    vision: model.capabilities?.vision ?? model.vision ?? model.tags?.includes('vision') ?? false,
    capabilities: model.capabilities,
    license: model.license
  }))
}
