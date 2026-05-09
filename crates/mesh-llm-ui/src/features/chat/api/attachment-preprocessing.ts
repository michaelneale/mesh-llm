import { describeImage, isModelLoaded } from '@/lib/image-describe'
import { extractPdfText, renderPdfPagesToImages } from '@/lib/pdf'
import type { AttachmentProcessingStage } from '@/features/chat/api/legacy-attachments'

export type ImagePromptDescription = { imageDescription?: string }

function imageProgressToStage(message: string): AttachmentProcessingStage {
  const normalized = message.toLowerCase()
  if (normalized.includes('starting')) return 'starting'
  if (normalized.includes('processing') || normalized.includes('analyzing')) return 'processing'
  return 'downloading'
}

export async function describeImageForPrompt(
  dataUrl: string,
  onStage?: (stage: AttachmentProcessingStage) => void
): Promise<ImagePromptDescription> {
  const result = await describeImage(dataUrl, (message) => onStage?.(imageProgressToStage(message)))
  const imageDescription = result.combinedText.trim()
  return imageDescription ? { imageDescription } : {}
}

export function isBrowserVisionModelLoaded(): boolean {
  return isModelLoaded()
}

export async function extractPdfTextFromFile(file: File) {
  const buffer = await file.arrayBuffer()
  return extractPdfText(buffer)
}

export async function describeScannedPdf(
  file: File,
  onStage?: (stage: AttachmentProcessingStage) => void
): Promise<string> {
  const buffer = await file.arrayBuffer()
  onStage?.('processing')
  const renderedPageImages = await renderPdfPagesToImages(buffer, { maxPages: 3, scale: 1, quality: 0.7 })
  if (renderedPageImages.length === 0) return ''

  const descriptions: string[] = []
  for (let index = 0; index < renderedPageImages.length; index += 1) {
    const { imageDescription } = await describeImageForPrompt(renderedPageImages[index] ?? '', onStage)
    if (imageDescription) {
      descriptions.push(`[Page ${index + 1}]\n${imageDescription}`)
    }
  }

  return descriptions.join('\n\n')
}
