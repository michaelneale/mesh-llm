import * as DialogPrimitive from '@radix-ui/react-dialog'
import type { ReactNode } from 'react'

type DrawerProps = {
  open: boolean
  onClose: () => void
  width?: string | number
  labelledBy?: string
  ariaLabel?: string
  children: ReactNode
}

export function Drawer({ open, onClose, width = 'min(480px, 92vw)', labelledBy, ariaLabel, children }: DrawerProps) {
  return (
    <DialogPrimitive.Root
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) onClose()
      }}
    >
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay aria-hidden="true" className="drawer-backdrop surface-scrim fixed inset-0 z-50" />

        <div className="fixed inset-0 z-50">
          <div className="absolute inset-y-0 right-0 max-w-full">
            <DialogPrimitive.Content
              aria-describedby={undefined}
              aria-label={labelledBy ? undefined : ariaLabel}
              aria-labelledby={labelledBy}
              className="drawer-panel shadow-surface-drawer h-full max-w-[92vw] overflow-y-auto overscroll-contain border-l border-border bg-panel text-foreground outline-none"
              style={{
                width
              }}
              tabIndex={-1}
            >
              <DialogPrimitive.Title className="sr-only">{ariaLabel ?? 'Drawer'}</DialogPrimitive.Title>
              {children}
            </DialogPrimitive.Content>
          </div>
        </div>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
