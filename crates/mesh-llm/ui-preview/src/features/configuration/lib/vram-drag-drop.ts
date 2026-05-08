import type { DragEvent } from 'react'
import { canFitModelInContainer, findModel } from '@/features/configuration/lib/config-math'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'

export const SOURCE_CONTAINER_MIME_PREFIX = 'application/x-mesh-source-container-'
export const MODEL_MIME_PREFIX = 'application/x-mesh-model-'
export const ASSIGN_MIME_PREFIX = 'application/x-mesh-assign-'

type VramDropIntent = {
  supported: boolean
  fits: boolean
  effect: 'copy' | 'move' | 'none'
}

type VramDropIntentOptions = {
  event: DragEvent<HTMLElement>
  readOnly: boolean
  sourceContainerType: string
  node: ConfigNode
  assigns: ConfigAssign[]
  containerIdx: number
  models?: ConfigModel[]
}

export function isWithinVramBar(event: DragEvent<HTMLElement>) {
  const rect = event.currentTarget.getBoundingClientRect()
  return (
    event.clientX >= rect.left &&
    event.clientX <= rect.right &&
    event.clientY >= rect.top &&
    event.clientY <= rect.bottom
  )
}

export function isReservedLaneEvent(event: DragEvent<HTMLElement>) {
  const target = event.target
  return target instanceof Element && Boolean(target.closest('[data-vram-reserved-lane="true"]'))
}

export function getTypedDataId(types: string[], prefix: string) {
  return types.find((type) => type.startsWith(prefix))?.slice(prefix.length) ?? ''
}

export function getVramDropIntent({
  event,
  readOnly,
  sourceContainerType,
  node,
  assigns,
  containerIdx,
  models
}: VramDropIntentOptions): VramDropIntent {
  if (readOnly) return { supported: false, fits: false, effect: 'none' }

  const types = Array.from(event.dataTransfer.types)
  const modelId = event.dataTransfer.getData('text/model') || getTypedDataId(types, MODEL_MIME_PREFIX)
  if (modelId) {
    const model = findModel(modelId, models)
    return {
      supported: true,
      fits: Boolean(model && canFitModelInContainer(model, node, assigns, containerIdx, 4096, undefined, models)),
      effect: 'copy'
    }
  }

  if (types.includes('text/model')) return { supported: true, fits: true, effect: 'copy' }

  if (!types.includes('text/assign-id') && !types.some((type) => type.startsWith(ASSIGN_MIME_PREFIX))) {
    return { supported: false, fits: false, effect: 'none' }
  }

  const sourceType = types.find((type) => type.startsWith(SOURCE_CONTAINER_MIME_PREFIX))
  if (sourceType === sourceContainerType) return { supported: false, fits: false, effect: 'none' }

  const assignId = event.dataTransfer.getData('text/assign-id') || getTypedDataId(types, ASSIGN_MIME_PREFIX)
  if (!assignId) return { supported: true, fits: true, effect: 'move' }

  const sourceNodeId = event.dataTransfer.getData('text/source-node')
  const sourceContainerIdx = Number(event.dataTransfer.getData('text/source-container'))
  if (
    sourceNodeId &&
    Number.isFinite(sourceContainerIdx) &&
    sourceNodeId === node.id &&
    sourceContainerIdx === containerIdx
  ) {
    return { supported: false, fits: false, effect: 'none' }
  }

  const existing = assigns.find((assign) => assign.id === assignId)
  if (!existing || (existing.nodeId === node.id && existing.containerIdx === containerIdx)) {
    return { supported: false, fits: false, effect: 'none' }
  }

  const model = findModel(existing.modelId, models)
  return {
    supported: true,
    fits: Boolean(
      model && canFitModelInContainer(model, node, assigns, containerIdx, existing.ctx, existing.id, models)
    ),
    effect: 'move'
  }
}
