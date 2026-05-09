/**
 * PDF text extraction and scanned-page rendering via pdf.js.
 *
 * Loaded lazily so the preview console does not pay for PDF support until a
 * user attaches a PDF.
 */

import pdfjsWorkerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import type { PDFDocumentProxy, TextItem, TextMarkedContent } from 'pdfjs-dist/types/src/display/api'

type PdfJs = typeof import('pdfjs-dist')

let pdfjsLib: PdfJs | null = null

async function loadPdfJs(): Promise<PdfJs> {
  if (pdfjsLib) return pdfjsLib

  try {
    pdfjsLib = await import('pdfjs-dist')
    pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorkerSrc
    return pdfjsLib
  } catch {
    throw new Error('Could not load PDF.js. Check that the application assets are available.')
  }
}

export type PdfExtractionResult = {
  text: string
  pageCount: number
  pagesWithText: number
  wordCount: number
}

export async function extractPdfText(buffer: ArrayBuffer): Promise<PdfExtractionResult> {
  const lib = await loadPdfJs()
  let doc: PDFDocumentProxy | null = null

  try {
    doc = await lib.getDocument({ data: new Uint8Array(buffer) }).promise
    const pageCount = doc.numPages
    const pages: string[] = []
    let pagesWithText = 0

    for (let pageNumber = 1; pageNumber <= pageCount; pageNumber += 1) {
      const page = await doc.getPage(pageNumber)
      try {
        const content = await page.getTextContent()
        let pageText = ''

        for (const item of content.items) {
          if (isTextItem(item)) {
            pageText += item.str
            if (item.hasEOL) pageText += '\n'
          }
        }

        const trimmed = pageText.trim()
        if (trimmed.length > 0) {
          pagesWithText += 1
          pages.push(`--- Page ${pageNumber} ---\n${trimmed}`)
        }
      } finally {
        page.cleanup()
      }
    }

    const text = pages.join('\n\n')
    const wordCount = text.split(/\s+/).filter(Boolean).length

    return { text, pageCount, pagesWithText, wordCount }
  } finally {
    if (doc) {
      await doc.destroy()
    }
  }
}

export async function renderPdfPagesToImages(
  buffer: ArrayBuffer,
  opts?: { maxPages?: number; scale?: number; quality?: number }
): Promise<string[]> {
  const lib = await loadPdfJs()
  const doc = await lib.getDocument({ data: new Uint8Array(buffer) }).promise

  try {
    const pageCount = doc.numPages
    const maxPages = opts?.maxPages ?? 4
    const scale = opts?.scale ?? 1
    const quality = opts?.quality ?? 0.72
    const images: string[] = []

    for (let pageNumber = 1; pageNumber <= Math.min(pageCount, maxPages); pageNumber += 1) {
      const page = await doc.getPage(pageNumber)
      let canvas: HTMLCanvasElement | undefined
      try {
        const viewport = page.getViewport({ scale })
        canvas = document.createElement('canvas')
        canvas.width = viewport.width
        canvas.height = viewport.height
        const ctx = canvas.getContext('2d')
        if (!ctx) continue

        await page.render({ canvasContext: ctx, viewport }).promise
        images.push(canvas.toDataURL('image/jpeg', quality))
      } finally {
        if (canvas) {
          canvas.width = 0
          canvas.height = 0
        }
        page.cleanup()
      }
    }

    return images
  } finally {
    await doc.destroy()
  }
}

function isTextItem(item: TextItem | TextMarkedContent): item is TextItem {
  return 'str' in item && typeof item.str === 'string'
}

export function isPdfMimeType(mimeType: string): boolean {
  return mimeType === 'application/pdf' || mimeType === 'application/x-pdf'
}

export function dataUrlToArrayBuffer(dataUrl: string): ArrayBuffer {
  const base64 = dataUrl.split(',')[1]
  if (!base64) throw new Error('Invalid data URL')

  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes.buffer
}
