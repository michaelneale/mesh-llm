import * as DialogPrimitive from '@radix-ui/react-dialog'
import { useBlocker } from '@tanstack/react-router'
import { AlertTriangle, LogOut, RotateCcw } from 'lucide-react'
import { useRef } from 'react'

function isConfigurationPath(pathname: string) {
  return pathname === '/configuration' || pathname.startsWith('/configuration/')
}

export function UnsavedConfigurationNavigationBlocker({ hasUnsavedChanges }: { hasUnsavedChanges: boolean }) {
  const contentRef = useRef<HTMLDivElement | null>(null)
  const blocker = useBlocker({
    shouldBlockFn: ({ current, next }) =>
      hasUnsavedChanges && !(isConfigurationPath(current.pathname) && isConfigurationPath(next.pathname)),
    enableBeforeUnload: hasUnsavedChanges,
    withResolver: true
  })

  if (blocker.status !== 'blocked') return null

  return (
    <DialogPrimitive.Root
      open
      onOpenChange={(open) => {
        if (!open) blocker.reset()
      }}
    >
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="surface-overlay fixed inset-0 z-50" />
        <DialogPrimitive.Content
          ref={contentRef}
          className="shadow-surface-modal fixed left-1/2 top-1/2 z-50 w-[min(430px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 rounded-[var(--radius-lg)] border border-border bg-panel p-5 text-foreground outline-none"
          onOpenAutoFocus={(event) => {
            event.preventDefault()
            contentRef.current?.focus()
          }}
          tabIndex={-1}
        >
          <div className="flex gap-3.5">
            <div
              className="grid size-9 shrink-0 place-items-center rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-warn)_42%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-warn)_12%,var(--color-panel))] text-warn"
              aria-hidden="true"
            >
              <AlertTriangle className="size-4" strokeWidth={2.2} />
            </div>
            <div className="min-w-0 flex-1 pt-0.5">
              <DialogPrimitive.Title className="text-[length:var(--density-type-headline)] font-semibold tracking-[-0.02em]">
                Unsaved configuration
              </DialogPrimitive.Title>
              <DialogPrimitive.Description className="mt-1.5 text-[length:var(--density-type-control)] leading-relaxed text-fg-dim">
                Save your configuration before leaving, or discard the changes and continue.
              </DialogPrimitive.Description>
            </div>
          </div>
          <div className="mt-5 flex justify-end gap-2 border-t border-border-soft pt-3.5">
            <button
              className="ui-control inline-flex h-8 min-w-[88px] items-center justify-center gap-1.5 rounded-[var(--radius)] border px-3 text-[length:var(--density-type-control)] font-medium leading-none"
              onClick={blocker.reset}
              type="button"
            >
              <RotateCcw className="size-3.5" aria-hidden="true" strokeWidth={2.2} />
              Stay
            </button>
            <button
              className="ui-control-primary inline-flex h-8 min-w-[122px] items-center justify-center gap-1.5 rounded-[var(--radius)] px-3 text-[length:var(--density-type-control)] font-semibold leading-none"
              onClick={blocker.proceed}
              type="button"
            >
              Leave page
              <LogOut className="size-3.5" aria-hidden="true" strokeWidth={2.2} />
            </button>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
