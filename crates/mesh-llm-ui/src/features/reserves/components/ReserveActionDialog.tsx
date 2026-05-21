import type { ReactNode } from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  SharedModal,
  SharedModalActionStrip,
  SharedModalBody,
  SharedModalContent,
  SharedModalDescription,
  SharedModalHeader,
  SharedModalTitle
} from '@/components/ui/SharedModal'

type ReserveActionDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description: ReactNode
  confirmLabel?: string
  cancelLabel?: string
  confirmTone?: 'default' | 'destructive'
  onConfirm?: () => void
  previewOnly?: boolean
  showCancel?: boolean
  children?: ReactNode
}

export function ReserveActionDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel = 'Done',
  cancelLabel = 'Cancel',
  confirmTone = 'default',
  onConfirm,
  previewOnly = true,
  showCancel = true,
  children
}: ReserveActionDialogProps) {
  return (
    <SharedModal open={open} onOpenChange={onOpenChange}>
      <SharedModalContent
        className="w-[min(720px,calc(100vw-1.5rem))] max-w-[720px]"
        onKeyDown={(event) => event.stopPropagation()}
      >
        <SharedModalHeader>
          <div className="flex flex-wrap items-center gap-2">
            <SharedModalTitle>{title}</SharedModalTitle>
            {previewOnly ? (
              <Badge className="h-5 rounded-full px-2 text-[10px] uppercase tracking-[0.08em] text-fg-dim">
                UI only
              </Badge>
            ) : null}
          </div>
          <SharedModalDescription className="max-w-[62ch] leading-[1.55]">{description}</SharedModalDescription>
        </SharedModalHeader>

        {children ? <SharedModalBody className="space-y-4">{children}</SharedModalBody> : null}

        <SharedModalActionStrip>
          {showCancel ? (
            <Button
              className="ui-control h-8 min-w-[96px] rounded-[var(--radius)] border px-3.5 text-[length:var(--density-type-control)]"
              onClick={() => onOpenChange(false)}
              size="sm"
              type="button"
              variant="outline"
            >
              {cancelLabel}
            </Button>
          ) : null}
          <Button
            className={
              confirmTone === 'destructive'
                ? 'ui-control-destructive h-8 min-w-[116px] rounded-[var(--radius)] border px-3.5 text-[length:var(--density-type-control)]'
                : 'ui-control-primary h-8 min-w-[116px] rounded-[var(--radius)] px-3.5 text-[length:var(--density-type-control)]'
            }
            onClick={() => {
              onConfirm?.()
              onOpenChange(false)
            }}
            size="sm"
            type="button"
            variant={confirmTone === 'destructive' ? 'destructive' : 'default'}
          >
            {confirmLabel}
          </Button>
        </SharedModalActionStrip>
      </SharedModalContent>
    </SharedModal>
  )
}
