/**
 * Browser-side image description via Transformers.js + Florence-2.
 *
 * Lazy-loads the library from the app bundle on first use — zero bundle cost
 * until the feature is actually used. The model (~230MB quantized) is fetched
 * from Hugging Face and cached in the browser's Cache API / IndexedDB for
 * instant subsequent loads.
 *
 * Used as a fallback when the user attaches an image but no vision
 * model is warm in the mesh — we describe the image locally and inject
 * the description as text so any text model can reason about it.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

let pipelineCache: any | null = null;
let loadingPromise: Promise<any> | null = null;

const MODEL_ID = "onnx-community/Florence-2-base-ft";

/**
 * Load transformers.js and initialize the Florence-2 pipeline.
 * Subsequent calls return the cached pipeline instantly.
 */
async function getDescriptionPipeline(): Promise<any> {
  if (pipelineCache) return pipelineCache;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    try {
      const { Florence2ForConditionalGeneration, AutoProcessor, AutoTokenizer, RawImage } =
        await import("@huggingface/transformers");

      const [model, processor, tokenizer] = await Promise.all([
        Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, {
          dtype: "fp32",
          device: "wasm",
        }),
        AutoProcessor.from_pretrained(MODEL_ID),
        AutoTokenizer.from_pretrained(MODEL_ID),
      ]);

      pipelineCache = { model, processor, tokenizer, RawImage };
      return pipelineCache;
    } catch (err) {
      loadingPromise = null;
      throw err;
    }
  })();

  return loadingPromise;
}

export type ImageDescriptionResult = {
  /** Detailed description of the image. */
  description: string;
  /** Any text/OCR content found in the image. */
  ocrText: string | null;
  /** Combined text for injection into the LLM context. */
  combinedText: string;
};

/**
 * Describe an image using Florence-2 running locally in the browser.
 *
 * @param imageSource - A data URL, blob URL, or HTMLImageElement.
 * @param onProgress - Optional callback for loading progress messages.
 * @returns Description and OCR text extracted from the image.
 */
export async function describeImage(
  imageSource: string,
  onProgress?: (message: string) => void,
): Promise<ImageDescriptionResult> {
  onProgress?.("Loading vision model...");
  const { model, processor, tokenizer, RawImage } = await getDescriptionPipeline();
  onProgress?.("Analyzing image...");

  const image = await RawImage.fromURL(imageSource);

  // Run detailed captioning.
  const captionPrompt = "<MORE_DETAILED_CAPTION>";
  const captionInputs = await processor(image, captionPrompt);
  const captionIds = await model.generate({
    ...captionInputs,
    max_new_tokens: 256,
  });
  // Slice off the prompt tokens from the generated output.
  const captionGenerated = captionIds.slice(
    null,
    [captionInputs.input_ids.dims.at(-1), null],
  );
  const description = tokenizer
    .batch_decode(captionGenerated, { skip_special_tokens: true })[0]
    ?.trim() ?? "";

  // Run OCR to extract any text in the image.
  let ocrText: string | null = null;
  try {
    const ocrPrompt = "<OCR>";
    const ocrInputs = await processor(image, ocrPrompt);
    const ocrIds = await model.generate({
      ...ocrInputs,
      max_new_tokens: 256,
    });
    const ocrGenerated = ocrIds.slice(
      null,
      [ocrInputs.input_ids.dims.at(-1), null],
    );
    const raw = tokenizer
      .batch_decode(ocrGenerated, { skip_special_tokens: true })[0]
      ?.trim() ?? "";
    if (raw.length > 3) ocrText = raw;
  } catch {
    // OCR is best-effort.
  }

  // Build the combined text for LLM injection.
  const parts: string[] = [];
  if (description) {
    parts.push(`[Image description: ${description}]`);
  }
  if (ocrText) {
    parts.push(`[Text visible in image: ${ocrText}]`);
  }
  const combinedText = parts.join("\n") || "[Unable to describe image]";

  return { description, ocrText, combinedText };
}

/**
 * Check if the browser can run the vision model.
 * Requires WebAssembly at minimum.
 */
export function canRunBrowserVision(): boolean {
  return typeof WebAssembly !== "undefined";
}

/**
 * Check if the model has been loaded (cached) previously.
 */
export function isModelLoaded(): boolean {
  return pipelineCache != null;
}
