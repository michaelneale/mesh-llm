const TOKENS_PER_K = 1024

export const CTX_TICKS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144] as const
export const CTX_MIN = CTX_TICKS[0]
export const CTX_MAX = CTX_TICKS[CTX_TICKS.length - 1]

function clampPct(pct: number): number { return Math.min(100, Math.max(0, pct)) }

export function normalizeCtx(ctx: number): number {
  if (!Number.isFinite(ctx)) return CTX_MIN
  return Math.round(boundCtx(ctx))
}

export function boundCtx(ctx: number): number {
  if (!Number.isFinite(ctx)) return CTX_MIN
  return Math.min(CTX_MAX, Math.max(CTX_MIN, ctx))
}

export function stepCtx(ctx: number, direction: -1 | 1): number {
  if (direction > 0) return ctx < TOKENS_PER_K ? TOKENS_PER_K : normalizeCtx(ctx + TOKENS_PER_K)
  return ctx <= TOKENS_PER_K ? CTX_MIN : normalizeCtx(ctx - TOKENS_PER_K)
}

export function jumpCtxPower(ctx: number, direction: -1 | 1): number {
  const bounded = boundCtx(ctx)

  if (direction > 0) return CTX_TICKS.find((tick) => tick > bounded) ?? CTX_MAX

  return [...CTX_TICKS].reverse().find((tick) => tick < bounded) ?? CTX_MIN
}

function nearestTickIndex(ctx: number): number {
  return CTX_TICKS.reduce((bestIndex, tick, index) => {
    const best = CTX_TICKS[bestIndex]
    return Math.abs(Math.log2(tick) - Math.log2(ctx)) < Math.abs(Math.log2(best) - Math.log2(ctx)) ? index : bestIndex
  }, 0)
}

export function ctxToPct(ctx: number): number {
  const bounded = Math.min(CTX_MAX, Math.max(CTX_MIN, ctx))
  return clampPct(((Math.log2(bounded) - Math.log2(CTX_MIN)) / (Math.log2(CTX_MAX) - Math.log2(CTX_MIN))) * 100)
}

export function pctToCtx(pct: number): number {
  const bounded = clampPct(pct)
  return 2 ** (Math.log2(CTX_MIN) + (bounded / 100) * (Math.log2(CTX_MAX) - Math.log2(CTX_MIN)))
}

export function snapCtx(ctx: number): number { return CTX_TICKS[nearestTickIndex(ctx)] }

export function fmtCtx(ctx: number): string {
  const tokens = Math.round(boundCtx(ctx))
  if (tokens < TOKENS_PER_K) return tokens.toLocaleString()
  if (tokens % TOKENS_PER_K === 0) return `${tokens / TOKENS_PER_K}K`
  if (tokens < TOKENS_PER_K * 16) return tokens.toLocaleString()
  return `${Math.round(tokens / TOKENS_PER_K)}K`
}
