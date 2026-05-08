import * as DialogPrimitive from '@radix-ui/react-dialog'
import { useMemo, useRef, type ReactNode, type RefObject } from 'react'
import { cn } from '@/lib/cn'

type TextInputDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: ReactNode
  description: ReactNode
  label: string
  value: string
  onValueChange: (value: string) => void
  onSave: (value: string) => void
  placeholder?: string
  saveLabel?: ReactNode
  cancelLabel?: ReactNode
  returnFocusRef?: RefObject<HTMLElement | null>
}

export function TextInputDialog({
  open,
  onOpenChange,
  title,
  description,
  label,
  value,
  onValueChange,
  onSave,
  placeholder,
  saveLabel = 'Save',
  cancelLabel = 'Cancel',
  returnFocusRef
}: TextInputDialogProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const characterCount = useMemo(() => new Intl.NumberFormat().format(value.length), [value.length])
  const characterLabel = `${characterCount} character${value.length === 1 ? '' : 's'}`

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="surface-scrim fixed inset-0 z-50 data-[state=closed]:animate-out data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" />
        <DialogPrimitive.Content
          className="shadow-surface-modal fixed left-1/2 top-1/2 z-50 w-[min(680px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel text-foreground outline-none data-[state=closed]:animate-out data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95"
          onCloseAutoFocus={(event) => {
            const returnFocusElement = returnFocusRef?.current
            if (!returnFocusElement || !document.contains(returnFocusElement)) return

            event.preventDefault()
            returnFocusElement.focus()
          }}
          onOpenAutoFocus={(event) => {
            event.preventDefault()
            textareaRef.current?.focus()
          }}
          onKeyDown={(event) => event.stopPropagation()}
        >
          <div className="px-5 pb-4 pt-4.5">
            <DialogPrimitive.Title className="text-[length:var(--density-type-headline)] font-semibold leading-5 tracking-[-0.02em] text-fg">
              {title}
            </DialogPrimitive.Title>
            <DialogPrimitive.Description className="mt-2 max-w-none text-wrap text-[length:var(--density-type-control)] leading-[1.5] text-fg-dim">
              {description}
            </DialogPrimitive.Description>
            <div className="mt-4">
              <label className="sr-only" htmlFor="text-input-dialog-textarea">
                {label}
              </label>
              <textarea
                ref={textareaRef}
                id="text-input-dialog-textarea"
                className="block w-full resize-y rounded-[var(--radius)] border border-border bg-panel-strong px-3.5 py-3 text-[length:var(--density-type-body)] leading-[1.5] text-fg outline-none transition-[border-color,box-shadow] placeholder:text-fg-faint focus-visible:border-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                placeholder={placeholder}
                value={value}
                style={{ minHeight: 180, fontFamily: 'var(--font-sans)' }}
                onChange={(event) => onValueChange(event.target.value)}
              />
              <div className="mt-2 text-right text-[length:var(--density-type-caption)] text-fg-faint">
                {characterLabel}
              </div>
            </div>
          </div>

          <div className="flex flex-col-reverse gap-2.5 border-t border-border-soft bg-panel-strong/70 px-5 py-3 sm:flex-row sm:justify-end">
            <DialogPrimitive.Close asChild>
              <button
                type="button"
                className={cn(
                  'ui-control inline-flex h-8 min-w-[96px] items-center justify-center rounded-[var(--radius)] border px-3.5 text-[length:var(--density-type-control)] font-medium leading-none outline-none transition-[background,color,box-shadow,transform]',
                  'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent'
                )}
              >
                {cancelLabel}
              </button>
            </DialogPrimitive.Close>
            <button
              type="button"
              className={cn(
                'ui-control-primary inline-flex h-8 min-w-[112px] items-center justify-center rounded-[var(--radius)] px-3.5 text-[length:var(--density-type-control)] font-semibold leading-none outline-none transition-[background,box-shadow,filter,transform]',
                'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent'
              )}
              onClick={() => {
                onSave(value)
                onOpenChange(false)
              }}
            >
              {saveLabel}
            </button>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
