import type { ContentPart } from '@tanstack/ai'
import type { MultimodalContent } from '@tanstack/ai-client'

const IMAGE_FALLBACK_TEXT = '[Image attached but could not be described]'

export type AttachmentProcessingStage = 'downloading' | 'starting' | 'processing'

type PdfExtractionResult = {
  text: string
  pageCount: number
  pagesWithText: number
  wordCount: number
}

type LegacyAttachmentDeps = {
  describeImage?: (
    dataUrl: string,
    onStage?: (stage: AttachmentProcessingStage) => void
  ) => Promise<{ imageDescription?: string }>
  extractPdfText?: (file: File) => Promise<PdfExtractionResult>
  describeScannedPdf?: (file: File, onStage?: (stage: AttachmentProcessingStage) => void) => Promise<string>
  onProcessingStage?: (stage: AttachmentProcessingStage, file: File) => void
}

type SourceMetadata = {
  fileName?: string
}

function isImageFile(file: File): boolean {
  return file.type.startsWith('image/')
}

function isAudioFile(file: File): boolean {
  return file.type.startsWith('audio/')
}

function isPdfFile(file: File): boolean {
  return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
}

function getPdfLabel(file: File): string {
  return file.name ? `[Content from ${file.name}]` : '[Extracted PDF content]'
}

function getPdfFallbackText(file: File): string {
  return `${getPdfLabel(file)}\n\n[Extracted PDF content unavailable in preview]`
}

async function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error(`Failed to read ${file.name}`))
    reader.onload = () => resolve(String(reader.result ?? ''))
    reader.readAsDataURL(file)
  })
}

async function resizeImageDataUrl(dataUrl: string, maxSide = 512): Promise<string> {
  return new Promise((resolve) => {
    const image = new Image()
    image.onload = () => {
      const largestSide = Math.max(image.width, image.height)
      if (largestSide <= maxSide) {
        resolve(dataUrl)
        return
      }

      const scale = maxSide / largestSide
      const canvas = document.createElement('canvas')
      canvas.width = Math.max(1, Math.round(image.width * scale))
      canvas.height = Math.max(1, Math.round(image.height * scale))
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        resolve(dataUrl)
        return
      }

      ctx.drawImage(image, 0, 0, canvas.width, canvas.height)
      resolve(canvas.toDataURL('image/jpeg', 0.85))
    }
    image.onerror = () => resolve(dataUrl)
    image.src = dataUrl
  })
}

function dataUrlToSource(dataUrl: string, mimeType: string) {
  const [, base64 = ''] = dataUrl.split(',', 2)
  return { type: 'data' as const, value: base64, mimeType }
}

async function buildImagePart(file: File, deps?: LegacyAttachmentDeps): Promise<ContentPart> {
  if (!deps?.describeImage) {
    return { type: 'text', content: IMAGE_FALLBACK_TEXT }
  }

  deps.onProcessingStage?.('downloading', file)
  const dataUrl = await readFileAsDataUrl(file)
  const resizedDataUrl = await resizeImageDataUrl(dataUrl)
  const result = await deps.describeImage(resizedDataUrl, (stage) => deps.onProcessingStage?.(stage, file))
  return {
    type: 'text',
    content: result.imageDescription?.trim() || IMAGE_FALLBACK_TEXT
  }
}

async function buildPdfPart(file: File, deps?: LegacyAttachmentDeps): Promise<ContentPart> {
  if (deps?.extractPdfText) {
    const extracted = await deps.extractPdfText(file)
    if (extracted.pagesWithText > 0 && extracted.wordCount > 20) {
      return {
        type: 'text',
        content: `${getPdfLabel(file)}\n\n${extracted.text}`
      }
    }
  }

  if (deps?.describeScannedPdf) {
    deps.onProcessingStage?.('downloading', file)
    const described = (await deps.describeScannedPdf(file, (stage) => deps.onProcessingStage?.(stage, file))).trim()
    if (described) {
      return {
        type: 'text',
        content: `${getPdfLabel(file)}\n\n${described}`
      }
    }
  }

  return {
    type: 'text',
    content: getPdfFallbackText(file)
  }
}

async function buildUploadPart(file: File, type: 'audio' | 'document'): Promise<ContentPart> {
  const dataUrl = await readFileAsDataUrl(file)
  const source = dataUrlToSource(dataUrl, file.type || (type === 'audio' ? 'audio/wav' : 'application/octet-stream'))
  const metadata: SourceMetadata = { fileName: file.name || undefined }

  if (type === 'audio') {
    return { type: 'audio', source, metadata }
  }

  return { type: 'document', source, metadata }
}

export async function buildComposerMessageContent(
  prompt: string,
  attachments: File[],
  deps?: LegacyAttachmentDeps
): Promise<string | MultimodalContent> {
  const trimmedPrompt = prompt.trim()
  if (attachments.length === 0) {
    return trimmedPrompt
  }

  const content: ContentPart[] = []
  if (trimmedPrompt) {
    content.push({ type: 'text', content: trimmedPrompt })
  }

  for (const attachment of attachments) {
    if (isImageFile(attachment)) {
      content.push(await buildImagePart(attachment, deps))
      continue
    }

    if (isPdfFile(attachment)) {
      content.push(await buildPdfPart(attachment, deps))
      continue
    }

    if (isAudioFile(attachment)) {
      content.push(await buildUploadPart(attachment, 'audio'))
      continue
    }

    content.push(await buildUploadPart(attachment, 'document'))
  }

  return { content }
}
