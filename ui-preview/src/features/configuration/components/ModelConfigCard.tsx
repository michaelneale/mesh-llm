import { animate, createScope } from 'animejs'
import { Trash2 } from 'lucide-react'
import { useLayoutEffect, useRef } from 'react'
import { CtxSlider } from '@/features/configuration/components/CtxSlider'
import { CTX_MAX, CTX_MIN, fmtCtx, snapCtx } from '@/features/configuration/components/ctx-slider-utils'
import { DetailPill } from '@/features/configuration/components/DetailPill'
import {
  contextGB,
  contextGBPerK,
  findModel,
  kvGB,
  modelFamilyColorKey
} from '@/features/configuration/lib/config-math'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'

type ModelConfigCardProps = {
  assign: ConfigAssign
  node: ConfigNode
  models?: ConfigModel[]
  containerFreeGB: number
  controlTabIndex?: number
  onCtxChange: (ctx: number) => void
  onRemove: () => void
}

function formatGB(value: number): string {
  return `${value.toFixed(1)} GB`
}
function formatShortfallGB(value: number): string {
  if (value < 1) return `${Math.max(1, Math.round(value * 1024)).toLocaleString()} MB`
  return formatGB(value)
}
function formatOptional(value: number | string | undefined): string {
  return value === undefined ? 'auto' : `${value}`
}

export function ModelConfigCard({
  assign,
  node,
  models,
  containerFreeGB,
  controlTabIndex,
  onCtxChange,
  onRemove
}: ModelConfigCardProps) {
  const cardRef = useRef<HTMLElement>(null)
  const model = findModel(assign.modelId, models)

  useLayoutEffect(() => {
    if (!cardRef.current) return

    const reduceMotion =
      typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

    const scope = createScope({ root: cardRef }).add(() => {
      animate(cardRef.current!, {
        opacity: [0, 1],
        y: reduceMotion ? 0 : [-6, 0],
        scaleY: reduceMotion ? 1 : [0.98, 1],
        duration: reduceMotion ? 0 : 180,
        ease: 'out(4)'
      })
    })

    return () => scope.revert()
  }, [])

  if (!model) return null

  const kv = contextGB(model, assign.ctx)
  const displayKV = kvGB(model, assign.ctx)
  const total = model.sizeGB + kv
  const modelCtxMax = Math.min(CTX_MAX, model.ctxMaxK * 1024)
  const maxAllowedCtx = Math.max(
    CTX_MIN,
    containerFreeGB > 0 ? (containerFreeGB / contextGBPerK(model)) * 1024 : CTX_MIN
  )
  const safeCtx = Math.min(CTX_MAX, maxAllowedCtx)
  const selectedGpu = node.gpus.find((gpu) => gpu.idx === assign.containerIdx)
  const locationLabel =
    node.placement === 'pooled'
      ? `${node.hostname} pool`
      : `GPU ${assign.containerIdx} · ${selectedGpu?.name ?? 'unknown'}`
  const paramsLabel = model.paramsLabel ?? `${model.paramsB}B`
  const modelShortfallGB = Math.max(0, -containerFreeGB)
  const ctxShortfallGB = Math.max(0, kv - Math.max(0, containerFreeGB))
  const hasError = modelShortfallGB > 0 || ctxShortfallGB > 0
  const errorText =
    modelShortfallGB > 0
      ? `ERROR: model allocation exceeds this container by ${formatGB(modelShortfallGB)}.`
      : `ERROR: context allocation needs ${formatShortfallGB(ctxShortfallGB)} more KV cache.`
  const details = [
    ['Ctx len', fmtCtx(modelCtxMax)],
    ['Weights', formatGB(model.sizeGB)],
    ['KV cache', formatGB(displayKV)],
    ['Total', formatGB(total)],
    ['Embed', formatOptional(model.embed)],
    ['Layers', formatOptional(model.layers)],
    ['Heads', formatOptional(model.heads)],
    ['Offload', model.layers === undefined ? 'auto' : Math.round(model.layers * 0.82).toString()],
    ['Tokenizer', formatOptional(model.tokenizer)],
    ['Dense', model.moe ? 'MoE' : 'Capable']
  ]

  return (
    <article
      aria-invalid={hasError}
      className={`mt-2 select-none rounded-[var(--radius-lg)] border bg-panel px-5 py-4 ${hasError ? 'border-bad shadow-[var(--shadow-surface-error-inset)]' : 'border-border-soft'}`}
      data-model-selection-area="true"
      ref={cardRef}
      style={{
        transformOrigin: 'top center'
      }}
    >
      <div className="flex flex-wrap items-center gap-2.5">
        <span aria-hidden="true" className="size-2 rounded-full" data-model-family-color={modelFamilyColorKey(model)} />
        <h3 className="min-w-0 flex-1 truncate text-[length:var(--density-type-body)] font-semibold">{model.name}</h3>
        <span className="font-mono text-[length:var(--density-type-label)] text-fg-faint">{model.family}</span>
        <span className="rounded-full border border-border-soft bg-background px-2 py-0.5 font-mono text-[length:var(--density-type-annotation)] text-fg-dim">
          {paramsLabel}
        </span>
        <span className="ml-auto text-[length:var(--density-type-caption)] text-fg-dim">
          on <span className="font-mono text-fg">{locationLabel}</span>
        </span>
        <button
          className="ui-control-destructive inline-flex size-[30px] shrink-0 items-center justify-center rounded-[var(--radius)] border"
          onClick={onRemove}
          tabIndex={controlTabIndex}
          title="Remove"
          type="button"
          aria-label={`Remove ${model.name} from ${locationLabel}`}
        >
          <Trash2 aria-hidden="true" className="size-[15px]" strokeWidth={1.8} />
        </button>
      </div>

      <div className="mt-3.5 flex flex-wrap gap-[5px]">
        {details.map(([label, value]) => (
          <DetailPill key={label} label={label} value={value} />
        ))}
      </div>

      {hasError ? (
        <div className="mt-2.5 rounded-[var(--radius)] border border-bad/70 bg-bad/10 px-2.5 py-2 text-[length:var(--density-type-caption)] font-medium text-bad">
          {errorText} Reduce context, remove another allocation, or move the model to a larger{' '}
          {node.placement === 'pooled' ? 'pool' : 'GPU'}.
        </div>
      ) : null}

      <div className="mt-2.5">
        <CtxSlider
          value={assign.ctx}
          onChange={onCtxChange}
          maxCtx={safeCtx}
          invalid={hasError}
          controlTabIndex={controlTabIndex}
        />
        <p className="mt-1.5 text-right text-[length:var(--density-type-label)] text-fg-faint">
          max ≈ {fmtCtx(snapCtx(maxAllowedCtx))} on this {node.placement === 'pooled' ? 'pool' : 'GPU'}
        </p>
      </div>
    </article>
  )
}
