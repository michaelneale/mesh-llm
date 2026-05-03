import * as SliderPrimitive from '@radix-ui/react-slider'
import { animate } from 'animejs'
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { CTX_MAX, CTX_MIN, CTX_TICKS, boundCtx, ctxToPct, fmtCtx, normalizeCtx, pctToCtx, stepCtx } from '@/features/configuration/components/ctx-slider-utils'

type CtxSliderProps = { value: number; onChange: (value: number) => void; maxCtx: number; invalid?: boolean; controlTabIndex?: number }

export function CtxSlider({ value, onChange, maxCtx, invalid = false, controlTabIndex }: CtxSliderProps) {
  const trackAlertRef = useRef<HTMLSpanElement>(null)
  const fillAlertRef = useRef<HTMLSpanElement>(null)
  const knobAlertRef = useRef<HTMLSpanElement>(null)
  const alertAnimationsRef = useRef<Array<ReturnType<typeof animate>>>([])
  const pointerInteractingRef = useRef(false)
  const latestRawCtx = useRef(value)
  const latestEmittedCtx = useRef(value)
  const [dragging, setDragging] = useState(false)
  const [draftCtx, setDraftCtx] = useState(value)
  const [hoveredTick, setHoveredTick] = useState<number | null>(null)

  useEffect(() => () => {
    alertAnimationsRef.current.forEach((animation) => { animation.pause() })
  }, [])

  useEffect(() => {
    if (dragging) return
    latestRawCtx.current = value
    latestEmittedCtx.current = value
  }, [dragging, value])

  const commitCtx = useCallback((nextCtx: number) => {
    if (!Number.isFinite(nextCtx)) return
    const raw = boundCtx(nextCtx)
    const next = normalizeCtx(raw)
    latestRawCtx.current = raw
    setDraftCtx(raw)
    if (next === latestEmittedCtx.current) return
    latestEmittedCtx.current = next
    onChange(next)
  }, [onChange])

  const commitPct = useCallback((nextPct: number) => {
    if (!Number.isFinite(nextPct)) return
    commitCtx(pctToCtx(nextPct))
  }, [commitCtx])

  const handleValueChange = useCallback((nextValues: number[]) => {
    const nextPct = nextValues[0]
    if (!Number.isFinite(nextPct)) return

    if (pointerInteractingRef.current) {
      commitPct(nextPct)
      return
    }

    if (nextPct <= 0) {
      commitCtx(CTX_MIN)
      return
    }

    if (nextPct >= 100) {
      commitCtx(CTX_MAX)
      return
    }

    const currentPct = ctxToPct(latestEmittedCtx.current)
    if (nextPct === currentPct) return

    commitCtx(stepCtx(latestEmittedCtx.current, nextPct > currentPct ? 1 : -1))
  }, [commitCtx, commitPct])

  const handleValueCommit = useCallback((nextValues: number[]) => {
    const nextPct = nextValues[0]
    if (!Number.isFinite(nextPct)) return
    if (!pointerInteractingRef.current) return
    commitPct(nextPct)
    pointerInteractingRef.current = false
    setDragging(false)
  }, [commitPct])

  const displayCtx = dragging ? draftCtx : value

  const valuePct = ctxToPct(displayCtx)
  const dangerStartPct = ctxToPct(maxCtx)
  const showDanger = maxCtx < CTX_MAX
  const overAllocated = invalid || displayCtx > maxCtx
  const valueText = displayCtx > maxCtx ? `${fmtCtx(displayCtx)} context exceeds ${fmtCtx(maxCtx)} safe limit` : `${fmtCtx(displayCtx)} context`

  useLayoutEffect(() => {
    const alertTargets = [trackAlertRef.current, fillAlertRef.current, knobAlertRef.current].filter((target): target is HTMLElement => target !== null)
    if (alertTargets.length === 0) return

    alertAnimationsRef.current.forEach((animation) => { animation.pause() })
    const reduceMotion = typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

    alertAnimationsRef.current = [animate(alertTargets, {
      opacity: overAllocated ? 1 : 0,
      duration: reduceMotion ? 0 : 180,
      ease: 'out(4)',
    })]
  }, [overAllocated])

  return (
    <div className="select-none">
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <span className="text-[length:var(--density-type-caption)] font-medium text-fg-dim">Context</span>
        <span className={`font-mono text-[length:var(--density-type-caption)] ${invalid ? 'text-bad' : 'text-fg'}`}>{fmtCtx(displayCtx)} ctx</span>
      </div>
      <SliderPrimitive.Root
        aria-invalid={overAllocated}
        className="relative flex h-6 cursor-pointer touch-none select-none items-center rounded-[var(--radius)] outline-none transition-[box-shadow] duration-150 focus-visible:shadow-[var(--shadow-focus-accent)] focus-visible:[&_[data-ctx-slider-track]]:border-accent"
        max={100}
        min={0}
        onPointerDown={() => {
          pointerInteractingRef.current = true
          setDragging(true)
        }}
        onPointerCancel={() => {
          pointerInteractingRef.current = false
          setDragging(false)
        }}
        onValueChange={handleValueChange}
        onValueCommit={handleValueCommit}
        step={0.001}
        value={[valuePct]}
      >
        <SliderPrimitive.Track data-ctx-slider-track className="relative h-full grow overflow-hidden rounded-[var(--radius)] border border-border-soft bg-muted transition-[border-color] duration-150">
          <SliderPrimitive.Range className="absolute inset-y-0 left-0 bg-accent" />
          <span ref={fillAlertRef} aria-hidden="true" className="pointer-events-none absolute inset-y-0 left-0 bg-bad opacity-0" style={{ width: `${valuePct}%` }} />
          {showDanger ? (
            <span
              aria-hidden="true"
              className="absolute inset-y-0 right-0 rounded-r-[var(--radius)] border-l border-dashed border-bad/80 opacity-75"
              style={{ left: `${dangerStartPct}%`, backgroundImage: 'repeating-linear-gradient(135deg, color-mix(in oklch, var(--color-bad) 42%, transparent) 0 3px, transparent 3px 7px)' }}
            />
          ) : null}
          {CTX_TICKS.map((tick) => (
            <span
              aria-hidden="true"
              className="absolute top-1/2 h-3 -translate-x-1/2 -translate-y-1/2 border-l border-background/60"
              key={tick}
              style={{ left: `${ctxToPct(tick)}%` }}
            />
          ))}
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          aria-invalid={overAllocated}
          aria-label="Context"
          aria-valuemax={CTX_MAX}
          aria-valuemin={CTX_MIN}
          aria-valuenow={normalizeCtx(displayCtx)}
          aria-valuetext={valueText}
          className={`block size-[13px] overflow-hidden rounded-full border border-panel bg-accent shadow-[var(--shadow-slider-thumb)] outline-none transition-transform duration-150 ${dragging ? 'scale-110' : ''}`}
          tabIndex={controlTabIndex}
        >
          <span ref={knobAlertRef} aria-hidden="true" className="absolute inset-0 rounded-full bg-bad opacity-0" />
        </SliderPrimitive.Thumb>
        <span ref={trackAlertRef} aria-hidden="true" className="pointer-events-none absolute inset-0 rounded-[var(--radius)] border border-bad opacity-0 shadow-[var(--shadow-slider-alert)]" />
      </SliderPrimitive.Root>
      <div className="relative mt-1 h-4">
        {CTX_TICKS.map((tick) => {
          const active = hoveredTick === tick || normalizeCtx(displayCtx) === tick
          const unsafe = tick > maxCtx
          return (
            <button
              className={`absolute -translate-x-1/2 rounded-[3px] px-1 py-0.5 font-mono text-[length:var(--density-type-micro)] transition-[background,color] duration-150 ${active ? '' : unsafe ? 'text-bad' : 'text-fg-faint hover:bg-muted hover:text-fg'}`}
              key={tick}
              onClick={() => commitCtx(tick)}
              onMouseEnter={() => setHoveredTick(tick)}
              onMouseLeave={() => setHoveredTick(null)}
              style={{
                left: `${ctxToPct(tick)}%`,
                ...(active ? { background: 'var(--color-accent)', color: 'var(--color-accent-ink)' } : undefined),
              }}
              tabIndex={controlTabIndex}
              title={`Set context to ${fmtCtx(tick)}`}
              type="button"
            >
              {fmtCtx(tick)}
            </button>
          )
        })}
      </div>
    </div>
  )
}
