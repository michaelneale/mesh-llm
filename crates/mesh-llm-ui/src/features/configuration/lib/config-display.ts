import { containerUsedGB } from '@/features/configuration/lib/config-math'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'

export function formatGB(value: number, options: { fixedFractionDigits?: number } = {}) {
  if (options.fixedFractionDigits !== undefined) return value.toFixed(options.fixedFractionDigits)

  return Number.isInteger(value) ? value.toString() : value.toFixed(1)
}

export function nodeUsedGB(node: ConfigNode, assigns: ConfigAssign[], models?: ConfigModel[]) {
  return node.gpus.reduce((sum, gpu) => sum + containerUsedGB(assigns, node.id, gpu.idx, models), 0)
}

export function nodeGpuCountLabel(node: ConfigNode) {
  return `${node.gpus.length} ${node.gpus.length === 1 ? 'device' : 'devices'}`
}
