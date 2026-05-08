import { describeImage } from "../../../lib/image-describe";

type ImageDescriptionResult = Awaited<ReturnType<typeof describeImage>>;

export async function describeImageAttachmentForPrompt(
  dataUrl: string,
  options?: {
    describe?: typeof describeImage;
    onProgress?: (message: string) => void;
  },
): Promise<{
  imageDescription?: string;
  extractionSummary: string;
  error?: string;
}> {
  const describe = options?.describe ?? describeImage;
  try {
    const result = await describe(dataUrl, options?.onProgress);
    const imageDescription = result.combinedText.trim();
    return {
      imageDescription: imageDescription || undefined,
      extractionSummary: result.ocrText
        ? "Described + OCR extracted"
        : "Described by local vision",
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return {
      imageDescription: undefined,
      extractionSummary: "Image description failed — retry or send placeholder text",
      error: `Image description failed: ${message}`,
    };
  }
}

export async function describeRenderedPagesAsText(
  renderedPageImages: string[],
  options?: {
    describe?: (dataUrl: string) => Promise<ImageDescriptionResult>;
    onProgress?: (message: string) => void;
  },
): Promise<string> {
  const describe = options?.describe ?? ((dataUrl: string) => describeImage(dataUrl));
  const pageSections: string[] = [];

  for (const [index, pageDataUrl] of renderedPageImages.entries()) {
    options?.onProgress?.(
      `Describing scanned PDF page ${index + 1}/${renderedPageImages.length}...`,
    );
    try {
      const result = await describe(pageDataUrl);
      const combinedText = result.combinedText.trim();
      pageSections.push(
        `[Page ${index + 1}]\n${combinedText || "[Unable to describe page]"}`,
      );
    } catch {
      pageSections.push(`[Page ${index + 1}]\n[Unable to describe page]`);
    }
  }

  return pageSections.join("\n\n");
}
