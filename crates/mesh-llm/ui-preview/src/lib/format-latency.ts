export function formatLatency(latencyMs: number): string {
  if (latencyMs > 0 && latencyMs < 1) return '<1'
  return Math.max(0, Math.round(latencyMs)).toString()
}
