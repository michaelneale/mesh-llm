/**
 * Helpers for SSE streaming that avoid per-token React re-renders.
 *
 * The core idea: tokens arrive faster than frames paint. We accumulate
 * them and flush to React at most once per requestAnimationFrame so the
 * browser only does one reconciliation + layout per vsync.
 */

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
