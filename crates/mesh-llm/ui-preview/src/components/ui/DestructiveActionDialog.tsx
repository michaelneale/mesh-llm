import * as AlertDialogPrimitive from '@radix-ui/react-alert-dialog'
import { AlertTriangle } from 'lucide-react'
import { useRef, type ReactNode, type RefObject } from 'react'
import { cn } from '@/lib/cn'

type DestructiveActionDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: ReactNode
  description: ReactNode
  destructiveLabel: ReactNode
  onConfirm: () => void
  cancelLabel?: ReactNode
  returnFocusRef?: RefObject<HTMLElement | null>
}

export function DestructiveActionDialog({
  open,
  onOpenChange,
  title,
  description,
  destructiveLabel,
  onConfirm,
  cancelLabel = 'Cancel',
  returnFocusRef
}: DestructiveActionDialogProps) {
  const cancelButtonRef = useRef<HTMLButtonElement | null>(null)
  const actionButtonRef = useRef<HTMLButtonElement | null>(null)

  return (
    <AlertDialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <AlertDialogPrimitive.Portal>
        <AlertDialogPrimitive.Overlay className="surface-scrim fixed inset-0 z-50 data-[state=closed]:animate-out data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" />
        <AlertDialogPrimitive.Content
          className="shadow-surface-modal fixed left-1/2 top-1/2 z-50 w-[min(480px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel text-foreground outline-none data-[state=closed]:animate-out data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95"
          onCloseAutoFocus={(event) => {
            const returnFocusElement = returnFocusRef?.current
            if (!returnFocusElement || !document.contains(returnFocusElement)) return

            event.preventDefault()
            returnFocusElement.focus()
          }}
          onOpenAutoFocus={(event) => {
            event.preventDefault()
            actionButtonRef.current?.focus()
          }}
          onKeyDown={(event) => {
            event.stopPropagation()

            if (event.key !== 'Tab') return

            const cancelButton = cancelButtonRef.current
            const actionButton = actionButtonRef.current
            if (!cancelButton || !actionButton) return

            if (event.shiftKey) {
              if (document.activeElement === cancelButton) {
                event.preventDefault()
                actionButton.focus()
              }
              return
            }

            if (document.activeElement === actionButton) {
              event.preventDefault()
              cancelButton.focus()
            }
          }}
        >
          <div className="grid grid-cols-[36px_minmax(0,1fr)] gap-x-3.5 px-5 pb-4 pt-4.5">
            <div className="pt-0.5">
              <span className="grid size-9 place-items-center rounded-[var(--radius)] border border-[color:color-mix(in_oklch,var(--color-destructive)_38%,var(--color-border))] bg-[color:color-mix(in_oklch,var(--color-destructive)_11%,var(--color-panel))] text-destructive shadow-[inset_0_1px_0_color-mix(in_oklch,var(--color-destructive)_16%,transparent)]">
                <AlertTriangle aria-hidden="true" className="size-4" strokeWidth={2.2} />
              </span>
            </div>
            <div className="min-w-0 pt-0.5">
              <AlertDialogPrimitive.Title className="text-[length:var(--density-type-headline)] font-semibold leading-5 tracking-[-0.02em] text-fg">
                {title}
              </AlertDialogPrimitive.Title>
              <AlertDialogPrimitive.Description className="mt-2.5 max-w-[48ch] text-[length:var(--density-type-control)] leading-[1.5] text-fg-dim">
                {description}
              </AlertDialogPrimitive.Description>
            </div>
          </div>

          <div className="flex flex-col-reverse gap-2.5 border-t border-border-soft bg-panel-strong/70 px-5 py-3 sm:flex-row sm:justify-end">
            <AlertDialogPrimitive.Cancel
              ref={cancelButtonRef}
              className={cn(
                'ui-control inline-flex h-8 min-w-[96px] items-center justify-center rounded-[var(--radius)] border px-3.5 text-[length:var(--density-type-control)] font-medium leading-none outline-none transition-[background,color,box-shadow,transform]',
                'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent'
              )}
            >
              {cancelLabel}
            </AlertDialogPrimitive.Cancel>
            <AlertDialogPrimitive.Action
              ref={actionButtonRef}
              className={cn(
                'inline-flex h-8 min-w-[116px] items-center justify-center rounded-[var(--radius)] border border-[color:color-mix(in_oklch,var(--color-destructive)_62%,var(--color-border))] bg-destructive px-3.5 text-[length:var(--density-type-control)] font-semibold leading-none text-destructive-foreground outline-none transition-[background,box-shadow,filter,transform]',
                'hover:brightness-110 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-destructive'
              )}
              onClick={onConfirm}
            >
              {destructiveLabel}
            </AlertDialogPrimitive.Action>
          </div>
        </AlertDialogPrimitive.Content>
      </AlertDialogPrimitive.Portal>
    </AlertDialogPrimitive.Root>
  )
}
