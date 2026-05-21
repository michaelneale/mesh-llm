import * as DialogPrimitive from '@radix-ui/react-dialog'
import { useMemo, useRef, type ReactNode, type RefObject } from 'react'
import {
  SharedModal,
  SharedModalActionStrip,
  SharedModalBody,
  SharedModalContent,
  SharedModalDescription,
  SharedModalHeader,
  SharedModalTitle
} from '@/components/ui/SharedModal'
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
    <SharedModal open={open} onOpenChange={onOpenChange}>
      <SharedModalContent
        className="w-[min(680px,calc(100vw-2rem))]"
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
        <SharedModalHeader>
          <SharedModalTitle>{title}</SharedModalTitle>
          <SharedModalDescription className="max-w-none text-wrap">{description}</SharedModalDescription>
        </SharedModalHeader>
        <SharedModalBody>
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
        </SharedModalBody>

        <SharedModalActionStrip>
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
        </SharedModalActionStrip>
      </SharedModalContent>
    </SharedModal>
  )
}
