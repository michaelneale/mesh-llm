import { useEffect, useRef } from 'react'
import { animate, createScope, createTimeline, type Scope } from 'animejs'

export function MeshVizTopologyGhost() {
  const rootRef = useRef<HTMLDivElement | null>(null)
  const scopeRef = useRef<Scope | null>(null)

  useEffect(() => {
    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (prefersReducedMotion) return undefined

    scopeRef.current = createScope({ root: rootRef }).add(() => {
      const clientPaths = [
        {
          from: [-146, -102],
          points: [
            [-154, -78],
            [-124, -108],
            [-97, -56],
            [-88, -49],
            [-106, -64],
            [-122, -84]
          ],
          opacity: [0, 0.34, 0.46, 0.58, 0.72, 0.62, 0.48, 0.34, 0],
          scale: [0.52, 0.72, 0.84, 0.76, 0.96, 0.82, 0.9, 0.72, 0.52],
          duration: 11_800,
          delay: 220,
          loopDelay: 1800
        },
        {
          from: [150, -94],
          points: [
            [158, -118],
            [126, -90],
            [92, -64],
            [82, -56],
            [102, -74],
            [124, -94]
          ],
          opacity: [0, 0.3, 0.42, 0.56, 0.7, 0.6, 0.46, 0.3, 0],
          scale: [0.5, 0.68, 0.82, 0.74, 0.92, 0.8, 0.88, 0.68, 0.5],
          duration: 12_900,
          delay: 1180,
          loopDelay: 2600
        },
        {
          from: [-150, 110],
          points: [
            [-164, 136],
            [-126, 104],
            [-92, 64],
            [-82, 58],
            [-104, 72],
            [-124, 98]
          ],
          opacity: [0, 0.32, 0.44, 0.54, 0.68, 0.56, 0.44, 0.32, 0],
          scale: [0.5, 0.7, 0.84, 0.76, 0.94, 0.8, 0.88, 0.7, 0.5],
          duration: 12_200,
          delay: 2160,
          loopDelay: 2200
        },
        {
          from: [146, 116],
          points: [
            [164, 140],
            [126, 118],
            [88, 70],
            [78, 62],
            [100, 80],
            [122, 104]
          ],
          opacity: [0, 0.28, 0.4, 0.52, 0.66, 0.55, 0.42, 0.28, 0],
          scale: [0.48, 0.66, 0.78, 0.72, 0.9, 0.76, 0.84, 0.66, 0.48],
          duration: 13_600,
          delay: 3120,
          loopDelay: 3200
        }
      ]

      const meshTimeline = createTimeline({ loop: true, loopDelay: 320, defaults: { ease: 'outQuart' } })
        .set('[data-dashboard-mesh-link]', { opacity: 0, strokeDashoffset: 1 })
        .set('[data-dashboard-mesh-node]', {
          opacity: 0,
          scale: 0.54,
          boxShadow: '0 0 0 0 color-mix(in oklab, var(--color-accent) 0%, transparent)'
        })
        .set('[data-dashboard-mesh-core]', { scale: 0.94, opacity: 0.76 })

      ;[0, 1, 2].forEach((index) => {
        const start = index * 720
        const nodeSelector = `[data-dashboard-mesh-node="${index}"]`
        const linkSelector = `[data-dashboard-mesh-link="${index}"]`

        meshTimeline
          .add(
            nodeSelector,
            {
              opacity: [0, 1, 0.98],
              scale: [0.6, 1.34, 1.08],
              boxShadow: [
                '0 0 0 0 color-mix(in oklab, var(--color-accent) 0%, transparent)',
                '0 0 30px 3px color-mix(in oklab, var(--color-accent) 34%, transparent)',
                '0 0 14px 1px color-mix(in oklab, var(--color-accent) 20%, transparent)'
              ],
              duration: 380
            },
            start
          )
          .add(linkSelector, { opacity: [0, 0.62], strokeDashoffset: [1, 0], duration: 420 }, start + 380)
          .add(
            '[data-dashboard-mesh-core]',
            { opacity: [0.76, 1, 0.84], scale: [0.94, 1.12, 0.98], duration: 360 },
            start + 500
          )
      })

      const disconnectStart = 2720

      ;[0, 1, 2].forEach((index) => {
        const start = disconnectStart + index * 260
        const nodeSelector = `[data-dashboard-mesh-node="${index}"]`
        const linkSelector = `[data-dashboard-mesh-link="${index}"]`

        meshTimeline
          .add(linkSelector, { opacity: [0.62, 0], strokeDashoffset: [0, -1], duration: 260, ease: 'inQuart' }, start)
          .add(
            nodeSelector,
            {
              opacity: [0.98, 0],
              scale: [1.08, 0.72],
              boxShadow: [
                '0 0 14px 1px color-mix(in oklab, var(--color-accent) 20%, transparent)',
                '0 0 0 0 color-mix(in oklab, var(--color-accent) 0%, transparent)'
              ],
              duration: 280,
              ease: 'inQuart'
            },
            start + 90
          )
      })

      meshTimeline.add(
        '[data-dashboard-mesh-core]',
        { opacity: [0.84, 0.76], scale: [0.98, 0.94], duration: 260, ease: 'inQuart' },
        disconnectStart + 860
      )

      clientPaths.forEach((path, index) => {
        const selector = `[data-dashboard-client-node="${index}"]`
        const node = rootRef.current?.querySelector<HTMLElement>(selector)
        if (node) {
          node.style.opacity = `${path.opacity[0] ?? 0}`
          node.style.transform = `translateX(${path.from[0]}px) translateY(${path.from[1]}px) scale(${path.scale[0] ?? 0.7})`
        }

        animate(selector, {
          opacity: path.opacity,
          translateX: [path.from[0], path.from[0], ...path.points.map((point) => point[0]), path.from[0]],
          translateY: [path.from[1], path.from[1], ...path.points.map((point) => point[1]), path.from[1]],
          scale: path.scale,
          duration: path.duration,
          delay: path.delay,
          loop: true,
          loopDelay: path.loopDelay,
          ease: 'inOutSine'
        })
      })
    })

    return () => {
      scopeRef.current?.revert()
      scopeRef.current = null
    }
  }, [])

  return (
    <div ref={rootRef} className="h-full min-h-0">
      <section className="panel-shell flex h-full min-h-0 flex-col rounded-[var(--radius-lg)] border border-border bg-panel p-3.5">
        <div className="flex shrink-0 items-center justify-between">
          <div className="h-4 w-32 rounded bg-[color:color-mix(in_oklab,var(--color-foreground)_8%,transparent)]" />
          <div className="h-6 w-24 rounded-full bg-[color:color-mix(in_oklab,var(--color-foreground)_8%,transparent)]" />
        </div>
        <div className="relative mt-4 grid min-h-0 flex-1 place-items-center overflow-hidden rounded-[var(--radius)] border border-border-soft bg-[radial-gradient(ellipse_at_center,color-mix(in_oklab,var(--color-panel)_70%,var(--color-background)),var(--color-background)_76%)]">
          <span
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,color-mix(in_oklab,var(--color-accent)_8%,transparent),transparent_58%)] opacity-70"
          />
          <span
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 opacity-[0.34] [background-image:radial-gradient(circle,color-mix(in_oklab,var(--color-foreground)_12%,transparent)_0.75px,transparent_1.45px),radial-gradient(circle,color-mix(in_oklab,var(--color-accent)_10%,transparent)_0.6px,transparent_1.35px)] [background-position:0_0,10px_10px] [background-size:20px_20px,20px_20px] [mask-image:radial-gradient(ellipse_at_center,black_22%,transparent_90%)]"
          />
          <span
            aria-hidden="true"
            className="pointer-events-none absolute left-1/2 top-1/2 size-[28rem] -translate-x-1/2 -translate-y-1/2 rounded-full border border-[color:color-mix(in_oklab,var(--color-border)_62%,transparent)] opacity-[0.24]"
          />
          <span
            aria-hidden="true"
            className="pointer-events-none absolute left-1/2 top-1/2 size-[38rem] -translate-x-1/2 -translate-y-1/2 rounded-full border border-[color:color-mix(in_oklab,var(--color-border)_46%,transparent)] opacity-[0.18]"
          />
          <span
            aria-hidden="true"
            className="pointer-events-none absolute left-1/2 top-1/2 h-px w-[32rem] -translate-x-1/2 -translate-y-1/2 rotate-[-18deg] bg-[linear-gradient(90deg,transparent,color-mix(in_oklab,var(--color-accent)_14%,transparent),transparent)] opacity-45"
          />
          <span
            aria-hidden="true"
            className="pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-[linear-gradient(180deg,transparent,color-mix(in_oklab,var(--color-panel)_42%,transparent))] opacity-50"
          />
          <div className="relative z-10 size-56">
            <span className="absolute inset-0 rounded-full border border-border-soft" />
            <svg aria-hidden="true" className="absolute inset-0 size-full" viewBox="0 0 224 224">
              <line
                className="opacity-30"
                data-dashboard-mesh-link="0"
                pathLength={1}
                stroke="color-mix(in oklab, var(--color-accent) 48%, var(--color-border))"
                strokeDasharray="0.055 0.055"
                strokeDashoffset={1}
                strokeLinecap="round"
                strokeWidth="1"
                x1="112"
                x2="40"
                y1="112"
                y2="56"
              />
              <line
                className="opacity-30"
                data-dashboard-mesh-link="1"
                pathLength={1}
                stroke="color-mix(in oklab, var(--color-accent) 48%, var(--color-border))"
                strokeDasharray="0.055 0.055"
                strokeDashoffset={1}
                strokeLinecap="round"
                strokeWidth="1"
                x1="112"
                x2="72"
                y1="112"
                y2="180"
              />
              <line
                className="opacity-30"
                data-dashboard-mesh-link="2"
                pathLength={1}
                stroke="color-mix(in oklab, var(--color-accent) 48%, var(--color-border))"
                strokeDasharray="0.055 0.055"
                strokeDashoffset={1}
                strokeLinecap="round"
                strokeWidth="1"
                x1="112"
                x2="176"
                y1="112"
                y2="88"
              />
            </svg>
            <span
              className="absolute left-8 top-12 size-4 rounded-full border border-[color:color-mix(in_oklab,var(--color-accent)_34%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_16%,var(--color-panel-strong))] opacity-60 shadow-[0_0_10px_color-mix(in_oklab,var(--color-accent)_12%,transparent)] will-change-transform"
              data-dashboard-mesh-node="0"
            />
            <span
              className="absolute bottom-9 left-16 size-4 rounded-full border border-[color:color-mix(in_oklab,var(--color-accent)_34%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_16%,var(--color-panel-strong))] opacity-60 shadow-[0_0_10px_color-mix(in_oklab,var(--color-accent)_12%,transparent)] will-change-transform"
              data-dashboard-mesh-node="1"
            />
            <span
              className="absolute right-10 top-20 size-4 rounded-full border border-[color:color-mix(in_oklab,var(--color-accent)_34%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_16%,var(--color-panel-strong))] opacity-60 shadow-[0_0_10px_color-mix(in_oklab,var(--color-accent)_12%,transparent)] will-change-transform"
              data-dashboard-mesh-node="2"
            />
            <span
              className="absolute left-1/2 top-1/2 size-2 rounded-full border border-[color:color-mix(in_oklab,var(--color-foreground)_64%,var(--color-accent))] bg-[color:color-mix(in_oklab,var(--color-foreground)_76%,var(--color-accent))] shadow-[0_0_12px_color-mix(in_oklab,var(--color-foreground)_20%,transparent)] will-change-transform"
              data-dashboard-client-node="0"
            />
            <span
              className="absolute left-1/2 top-1/2 size-1.5 rounded-full border border-[color:color-mix(in_oklab,var(--color-foreground)_64%,var(--color-accent))] bg-[color:color-mix(in_oklab,var(--color-foreground)_76%,var(--color-accent))] shadow-[0_0_12px_color-mix(in_oklab,var(--color-foreground)_20%,transparent)] will-change-transform"
              data-dashboard-client-node="1"
            />
            <span
              className="absolute left-1/2 top-1/2 size-2 rounded-full border border-[color:color-mix(in_oklab,var(--color-foreground)_64%,var(--color-accent))] bg-[color:color-mix(in_oklab,var(--color-foreground)_76%,var(--color-accent))] shadow-[0_0_12px_color-mix(in_oklab,var(--color-foreground)_20%,transparent)] will-change-transform"
              data-dashboard-client-node="2"
            />
            <span
              className="absolute left-1/2 top-1/2 size-1.5 rounded-full border border-[color:color-mix(in_oklab,var(--color-foreground)_64%,var(--color-accent))] bg-[color:color-mix(in_oklab,var(--color-foreground)_76%,var(--color-accent))] shadow-[0_0_12px_color-mix(in_oklab,var(--color-foreground)_20%,transparent)] will-change-transform"
              data-dashboard-client-node="3"
            />
            <span
              className="absolute left-1/2 top-1/2 size-5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-accent-contrast bg-[color:color-mix(in_oklab,var(--color-accent-contrast)_22%,var(--color-panel))] text-accent-contrast shadow-[0_0_18px_color-mix(in_oklab,var(--color-accent-contrast)_24%,transparent)] will-change-transform"
              data-dashboard-mesh-core
            />
          </div>
        </div>
      </section>
    </div>
  )
}
