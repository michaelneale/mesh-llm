import type { MeshModelRaw } from '@/lib/api/types'
import type { ModelSummary } from '@/features/app-tabs/types'

function formatSize(sizeGB: number): string {
  if (sizeGB >= 1) {
    return `${sizeGB.toFixed(1)}B`
  }
  return `${(sizeGB * 1000).toFixed(0)}M`
}

function formatContext(contextLength: number): string {
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
    ctxMaxK: Math.round(model.context_length / 1000),
    moe: model.capabilities.moe,
    vision: model.capabilities.vision
  }))
}
