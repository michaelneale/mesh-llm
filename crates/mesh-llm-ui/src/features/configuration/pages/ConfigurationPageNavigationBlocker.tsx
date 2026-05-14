import { useBlocker } from '@tanstack/react-router'
import { AlertTriangle, LogOut, RotateCcw } from 'lucide-react'
import { useRef } from 'react'
import {
  SharedModal,
  SharedModalActionStrip,
  SharedModalBody,
  SharedModalContent,
  SharedModalDescription,
  SharedModalHeader,
  SharedModalTitle
} from '@/components/ui/SharedModal'

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
    <SharedModal
      open
      onOpenChange={(open) => {
        if (!open) blocker.reset()
      }}
    >
      <SharedModalContent
        ref={contentRef}
        onOpenAutoFocus={(event) => {
          event.preventDefault()
          contentRef.current?.focus()
        }}
        tabIndex={-1}
      >
        <SharedModalHeader className="relative pr-14">
          <SharedModalTitle>Unsaved configuration</SharedModalTitle>
          <div
            className="absolute right-5 top-1/2 -translate-y-1/2 grid size-9 shrink-0 place-items-center rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-warn)_42%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-warn)_12%,var(--color-panel))] text-warn"
            aria-hidden="true"
          >
            <AlertTriangle className="size-4" strokeWidth={2.2} />
          </div>
        </SharedModalHeader>
        <SharedModalBody>
          <SharedModalDescription className="leading-relaxed">
            Save your configuration before leaving, or discard the changes and continue.
          </SharedModalDescription>
        </SharedModalBody>
        <SharedModalActionStrip>
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
        </SharedModalActionStrip>
      </SharedModalContent>
    </SharedModal>
  )
}
