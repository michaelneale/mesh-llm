import { CFG_CATALOG } from '@/features/app-tabs/data'
import type { ConfigAssign, ConfigModel, ConfigNode, ModelFamilyColorKey } from '@/features/app-tabs/types'

const MODEL_FAMILY_COLOR_FALLBACK: ModelFamilyColorKey = 'family-0'
const MODEL_FAMILY_COLOR_KEYS: readonly ModelFamilyColorKey[] = ['family-0', 'family-1', 'family-2', 'family-3', 'family-4', 'family-5', 'family-6', 'family-7']

export function findModel(modelId: string, models: ConfigModel[] = CFG_CATALOG): ConfigModel | undefined { return models.find((model) => model.id === modelId) }
export function nodeTotalGB(node: ConfigNode): number { return node.gpus.reduce((total, gpu) => total + gpu.totalGB, 0) }
export function nodeReservedGB(node: ConfigNode): number { return node.gpus.reduce((total, gpu) => total + (gpu.reservedGB ?? 0), 0) }
export function nodeUsableGB(node: ConfigNode): number { return Math.max(0, nodeTotalGB(node) - nodeReservedGB(node)) }
export function contextGBPerK(model: ConfigModel): number { return model.ctxPerGB ?? model.paramsB / 120 }
export function contextGB(model: ConfigModel, ctx: number): number { return contextGBPerK(model) * (ctx / 1024) }
export function kvGB(model: ConfigModel, ctx: number): number { return Number(contextGB(model, ctx).toFixed(1)) }
export function containerAssigns(assigns: ConfigAssign[], nodeId: string, containerIdx: number): ConfigAssign[] { return assigns.filter((assign) => assign.nodeId === nodeId && assign.containerIdx === containerIdx) }
export function containerUsedGB(assigns: ConfigAssign[], nodeId: string, containerIdx: number, models: ConfigModel[] = CFG_CATALOG): number {
  return containerAssigns(assigns, nodeId, containerIdx).reduce((total, assign) => {
    const model = findModel(assign.modelId, models)
    return model ? total + model.sizeGB + contextGB(model, assign.ctx) : total
  }, 0)
}
export function containerTotalGB(node: ConfigNode, containerIdx: number): number {
  if (node.placement === 'pooled') return nodeTotalGB(node)
  return node.gpus.find((gpu) => gpu.idx === containerIdx)?.totalGB ?? 0
}
export function containerReservedGB(node: ConfigNode, containerIdx: number): number {
  if (node.placement === 'pooled') return node.gpus.reduce((sum, gpu) => sum + (gpu.reservedGB ?? 0), 0)
  return node.gpus.find((gpu) => gpu.idx === containerIdx)?.reservedGB ?? 0
}
export function modelNeedGB(model: ConfigModel, ctx = 4096): number { return model.sizeGB + contextGB(model, ctx) }
export function containerAvailableGB(assigns: ConfigAssign[], node: ConfigNode, containerIdx: number, ignoredAssignId?: string, models: ConfigModel[] = CFG_CATALOG): number {
  const scopedAssigns = ignoredAssignId ? assigns.filter((assign) => assign.id !== ignoredAssignId) : assigns
  return containerTotalGB(node, containerIdx) - containerReservedGB(node, containerIdx) - containerUsedGB(scopedAssigns, node.id, containerIdx, models)
}
export function canFitModelInContainer(model: ConfigModel, node: ConfigNode, assigns: ConfigAssign[], containerIdx: number, ctx = 4096, ignoredAssignId?: string, models: ConfigModel[] = CFG_CATALOG): boolean {
  return modelNeedGB(model, ctx) <= containerAvailableGB(assigns, node, containerIdx, ignoredAssignId, models)
}
export function findModelFitContainerIdx(model: ConfigModel, node: ConfigNode, assigns: ConfigAssign[], ctx = 4096, models: ConfigModel[] = CFG_CATALOG): number | null {
  if (node.placement === 'pooled') return canFitModelInContainer(model, node, assigns, 0, ctx, undefined, models) ? 0 : null
  return node.gpus.find((gpu) => canFitModelInContainer(model, node, assigns, gpu.idx, ctx, undefined, models))?.idx ?? null
}
export function findPreferredModelFitContainerIdx(model: ConfigModel, node: ConfigNode, assigns: ConfigAssign[], preferredContainerIdx?: number | null, ctx = 4096, models: ConfigModel[] = CFG_CATALOG): number | null {
  if (node.placement === 'pooled') return findModelFitContainerIdx(model, node, assigns, ctx, models)
  if (typeof preferredContainerIdx === 'number' && Number.isFinite(preferredContainerIdx) && canFitModelInContainer(model, node, assigns, preferredContainerIdx, ctx, undefined, models)) return preferredContainerIdx
  return findModelFitContainerIdx(model, node, assigns, ctx, models)
}
export function isUnifiedMemoryNode(node: ConfigNode): boolean {
  return node.memoryTopology === 'unified' || node.region.toLowerCase() === 'unified' || node.gpus.some((gpu) => gpu.name.toLowerCase().includes('unified memory'))
}
export function hasConfigurablePlacement(node: ConfigNode): boolean { return !isUnifiedMemoryNode(node) && node.gpus.length > 1 }
export function modelFamilyColorKey(model: Pick<ConfigModel, 'family' | 'familyColor'> | undefined): ModelFamilyColorKey {
  if (model?.familyColor) return model.familyColor

  const family = model?.family.trim().toLowerCase() ?? ''
  if (!family) return MODEL_FAMILY_COLOR_FALLBACK

  let hash = 0
  for (const character of family) hash = (hash * 31 + character.charCodeAt(0)) % MODEL_FAMILY_COLOR_KEYS.length
  return MODEL_FAMILY_COLOR_KEYS[hash] ?? MODEL_FAMILY_COLOR_FALLBACK
}
export function modelColor(modelId: string, models: ConfigModel[] = CFG_CATALOG): string {
  const colorKey = modelFamilyColorKey(findModel(modelId, models))
  return `var(--model-family-color-${colorKey}, var(--model-family-color-fallback))`
}
export function modelTextColor(): string { return 'var(--model-family-text)' }
