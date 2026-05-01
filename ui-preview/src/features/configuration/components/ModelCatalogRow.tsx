import type { DragEvent } from 'react'
import { findModelFitContainerIdx, modelFamilyColorKey } from '@/features/configuration/lib/config-math'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'

type ModelCatalogRowProps = {
  model: ConfigModel
  node: ConfigNode
  assigns: ConfigAssign[]
  models?: ConfigModel[]
  selected?: boolean
  onDragEnd?: () => void
  onDragStart?: () => void
}

function setModelDragImage(event: DragEvent<HTMLButtonElement>) {
  const source = event.currentTarget
  const rect = source.getBoundingClientRect()
  const preview = source.cloneNode(true)

  if (!(preview instanceof HTMLElement)) return

  preview.style.position = 'fixed'
  preview.style.top = '-1000px'
  preview.style.left = '-1000px'
  preview.style.width = `${rect.width}px`
  preview.style.pointerEvents = 'none'
  preview.style.opacity = '1'
  preview.style.transform = 'none'
  preview.style.zIndex = '2147483647'
  preview.style.boxShadow = 'var(--shadow-surface-drag)'

  document.body.append(preview)
  event.dataTransfer.setDragImage(preview, event.clientX - rect.left, event.clientY - rect.top)
  window.setTimeout(() => preview.remove(), 0)
}

export function ModelCatalogRow({ model, node, assigns, models, selected = false, onDragEnd, onDragStart }: ModelCatalogRowProps) {
  const fits = findModelFitContainerIdx(model, node, assigns, 4096, models) !== null
  const verdict = fits ? 'Fits' : 'No fit'
  return <button draggable aria-label={`${model.name}, ${model.sizeGB} GB, ${model.ctxMaxK}k context, ${verdict}`} onDragEnd={onDragEnd} onDragStart={(event) => { event.dataTransfer.setData('text/model', model.id); event.dataTransfer.setData(`application/x-mesh-model-${model.id}`, model.id); event.dataTransfer.effectAllowed = 'copy'; setModelDragImage(event); onDragStart?.() }} className={`pointer-events-auto grid w-full cursor-grab grid-cols-[0.25rem_1fr_auto] gap-3 rounded-[var(--radius-lg)] border bg-background p-3 text-left active:cursor-grabbing ${selected ? 'border-accent shadow-[var(--shadow-surface-selected)]' : 'border-border'}`} type="button"><span className="rounded-full" data-model-family-color={modelFamilyColorKey(model)} /><div><div className="font-medium">{model.name}</div><div className="font-mono text-[length:var(--density-type-annotation)] text-muted-foreground">{model.family} · {model.quant} · {model.sizeGB} GB · {model.ctxMaxK}k ctx</div></div><span className={fits ? 'text-good' : 'text-bad'}>{verdict}</span></button>
}
