export function formatLatency(latencyMs: number): string {
  return latencyMs < 2 ? '<1' : latencyMs.toFixed(1)
}
