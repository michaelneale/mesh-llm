/**
 * Browser-side image description via Transformers.js + Florence-2.
 *
 * The model is loaded only when an image attachment needs description. Its
 * output is injected as text so regular text models can still reason about
 * attached images, matching the legacy console behavior.
 */

import type { Tensor } from '@huggingface/transformers'

let pipelineCache: DescriptionPipeline | null = null
let loadingPromise: Promise<DescriptionPipeline> | null = null

const MODEL_ID = 'onnx-community/Florence-2-base-ft'

type DescriptionPipeline = Awaited<ReturnType<typeof createDescriptionPipeline>>
type SliceableTensor = { slice: (start: null, end: [number, null]) => Tensor }

async function createDescriptionPipeline() {
  const { Florence2ForConditionalGeneration, AutoProcessor, AutoTokenizer, RawImage } =
    await import('@huggingface/transformers')

  const [model, processor, tokenizer] = await Promise.all([
    Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype: 'fp32',
      device: 'wasm'
    }),
    AutoProcessor.from_pretrained(MODEL_ID),
    AutoTokenizer.from_pretrained(MODEL_ID)
  ])

  return { model, processor, tokenizer, RawImage }
}

async function getDescriptionPipeline(): Promise<DescriptionPipeline> {
  if (pipelineCache) return pipelineCache
  if (loadingPromise) return loadingPromise

  loadingPromise = (async () => {
    try {
      pipelineCache = await createDescriptionPipeline()
      return pipelineCache
    } catch (error) {
      loadingPromise = null
      throw error
    }
  })()

  return loadingPromise
}

export type ImageDescriptionResult = {
  description: string
  ocrText: string | null
  objects: string[]
  combinedText: string
}

let pipelineQueue: Promise<unknown> = Promise.resolve()
function enqueue<T>(task: () => Promise<T>): Promise<T> {
  const next = pipelineQueue.then(task, task)
  pipelineQueue = next.catch(() => undefined)
  return next
}

export async function describeImage(
  imageSource: string,
  onProgress?: (message: string) => void
): Promise<ImageDescriptionResult> {
  return enqueue(() => describeImageInternal(imageSource, onProgress))
}

async function describeImageInternal(
  imageSource: string,
  onProgress?: (message: string) => void
): Promise<ImageDescriptionResult> {
  const modelAlreadyLoaded = pipelineCache != null
  if (!modelAlreadyLoaded) onProgress?.('Downloading vision model...')
  const { model, processor, tokenizer, RawImage } = await getDescriptionPipeline()
  if (!modelAlreadyLoaded) onProgress?.('Starting local vision model...')

  const image = await RawImage.fromURL(imageSource)
  onProgress?.('Processing image...')

  const captionPrompt = '<MORE_DETAILED_CAPTION>'
  const captionInputs = await processor(image, captionPrompt)
  const captionIds = await model.generate({
    ...captionInputs,
    max_new_tokens: 256
  })
  const captionGenerated = sliceGeneratedIds(captionIds, captionInputs.input_ids.dims.at(-1))
  const description = tokenizer.batch_decode(captionGenerated, { skip_special_tokens: true })[0]?.trim() ?? ''

  let objects: string[] = []
  try {
    const regionPrompt = '<DENSE_REGION_CAPTION>'
    const regionInputs = await processor(image, regionPrompt)
    const regionIds = await model.generate({
      ...regionInputs,
      max_new_tokens: 256
    })
    const regionGenerated = sliceGeneratedIds(regionIds, regionInputs.input_ids.dims.at(-1))
    const regionText = tokenizer.batch_decode(regionGenerated, { skip_special_tokens: true })[0]?.trim() ?? ''
    objects = extractObjectLabels(regionText, description)
  } catch {
    // Region captioning is best-effort; caption + OCR still land.
  }

  let ocrText: string | null = null
  try {
    const ocrPrompt = '<OCR>'
    const ocrInputs = await processor(image, ocrPrompt)
    const ocrIds = await model.generate({
      ...ocrInputs,
      max_new_tokens: 256
    })
    const ocrGenerated = sliceGeneratedIds(ocrIds, ocrInputs.input_ids.dims.at(-1))
    const raw = tokenizer.batch_decode(ocrGenerated, { skip_special_tokens: true })[0]?.trim() ?? ''
    if (raw.length > 3) ocrText = raw
  } catch {
    // OCR is best-effort.
  }

  const parts: string[] = []
  if (description) parts.push(`[Image description: ${description}]`)
  if (objects.length) parts.push(`[Visible objects: ${objects.join('; ')}]`)
  if (ocrText) parts.push(`[Text visible in image: ${ocrText}]`)

  return {
    description,
    ocrText,
    objects,
    combinedText: parts.join('\n') || '[Unable to describe image]'
  }
}

function isSliceableTensor(value: unknown): value is SliceableTensor {
  return typeof value === 'object' && value !== null && 'slice' in value && typeof value.slice === 'function'
}

function sliceGeneratedIds(value: unknown, promptTokenCount: number | undefined): Tensor {
  if (!isSliceableTensor(value)) {
    throw new Error('Unexpected vision model output')
  }

  return value.slice(null, [promptTokenCount ?? 0, null])
}

export function extractObjectLabels(raw: string, caption: string): string[] {
  if (!raw) return []

  const stripped = raw.replace(/<loc_\d+>/g, '\n')
  const captionLower = caption.toLowerCase()
  const seen = new Set<string>()
  const labels: string[] = []

  for (const rawLabel of stripped.split(/[\n,;]+/)) {
    const label = rawLabel.replace(/\s+/g, ' ').trim()
    if (!label) continue

    const key = label.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    if (captionLower.includes(key)) continue

    labels.push(label)
    if (labels.length >= 12) break
  }

  return labels
}

export function canRunBrowserVision(): boolean {
  return typeof WebAssembly !== 'undefined'
}

export function isModelLoaded(): boolean {
  return pipelineCache != null
}
