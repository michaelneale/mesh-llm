/**
 * Helpers for SSE streaming that avoid per-token React re-renders.
 *
 * The core idea: tokens arrive faster than frames paint. We accumulate
 * them and flush to React at most once per requestAnimationFrame so the
 * browser only does one reconciliation + layout per vsync.
 */

/**
 * Returns true when the Responses API input array contains at least one
 * content block whose URL starts with "mesh://blob/".  Blob tokens are
 * single-use and request-scoped, so retrying with the same tokens always
 * fails with "Unknown or expired blob token".
 */
export function hasBlobContent(input: unknown): boolean {
  if (!Array.isArray(input)) return false;
  return input.some(
    (msg: unknown) =>
      typeof msg === "object" &&
      msg !== null &&
      Array.isArray((msg as Record<string, unknown>).content) &&
      ((msg as Record<string, unknown>).content as Array<unknown>).some(
        (block) =>
          typeof block === "object" &&
          block !== null &&
          Object.values(block as Record<string, unknown>).some(
            (v) => typeof v === "string" && v.startsWith("mesh://blob/"),
          ),
      ),
  );
}

/**
 * Reads the response body and returns the most useful error string:
 * - `parsed.error.message` if the body is `{"error":{"message":"..."}}`
 * - `parsed.error` if the body is `{"error":"..."}`
 * - The raw body text for short (<500 char) non-JSON responses
 * - `"HTTP <status>"` as a final fallback
 */
export async function parseApiErrorBody(response: Response): Promise<string> {
  const fallback = `HTTP ${response.status}`;
  try {
    const errorBody = await response.text();
    if (errorBody.length === 0) return fallback;
    try {
      const parsed = JSON.parse(errorBody) as unknown;
      if (
        typeof parsed === "object" &&
        parsed !== null &&
        "error" in parsed
      ) {
        const err = (parsed as Record<string, unknown>).error;
        if (
          typeof err === "object" &&
          err !== null &&
          typeof (err as Record<string, unknown>).message === "string"
        ) {
          return (err as Record<string, unknown>).message as string;
        }
        if (typeof err === "string") {
          return err;
        }
      }
      if (errorBody.length < 500) return errorBody;
    } catch {
      if (errorBody.length < 500) return errorBody;
    }
  } catch {
    // Body couldn't be read — keep the status code message.
  }
  return fallback;
}

/** Schedule a callback at most once per animation frame. */
export function createRafBatcher(callback: (text: string) => void) {
  let raf = 0;
  let latest = "";

  return {
    /** Call on every stream update — stores the latest text snapshot + raf check. */
    push(text: string) {
      latest = text;
      if (!raf) {
        raf = window.requestAnimationFrame(() => {
          raf = 0;
          callback(latest);
        });
      }
    },
    /** Flush any pending update synchronously (call when stream ends). */
    flush() {
      if (raf) {
        window.cancelAnimationFrame(raf);
        raf = 0;
      }
      callback(latest);
    },
    cancel() {
      if (raf) {
        window.cancelAnimationFrame(raf);
        raf = 0;
      }
    },
  };
}
