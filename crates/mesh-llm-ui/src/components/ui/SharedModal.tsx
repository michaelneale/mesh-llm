import * as DialogPrimitive from '@radix-ui/react-dialog'
import * as React from 'react'
import { cn } from '@/lib/cn'

const SharedModal = DialogPrimitive.Root

const SharedModalOverlay = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={cn(
      'surface-scrim fixed inset-0 z-50 data-[state=closed]:animate-out data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
      className
    )}
    {...props}
  />
))
SharedModalOverlay.displayName = DialogPrimitive.Overlay.displayName

const SharedModalContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <DialogPrimitive.Portal>
    <SharedModalOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        'shadow-surface-modal fixed left-1/2 top-1/2 z-50 w-[min(430px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel text-foreground outline-none data-[state=closed]:animate-out data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95',
        className
      )}
      {...props}
    >
      {children}
    </DialogPrimitive.Content>
  </DialogPrimitive.Portal>
))
SharedModalContent.displayName = DialogPrimitive.Content.displayName

function SharedModalHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('border-b border-border-soft px-5 pb-4 pt-4.5', className)} {...props} />
}

const SharedModalTitle = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title
    ref={ref}
    className={cn(
      'text-[length:var(--density-type-headline)] font-semibold leading-5 tracking-[-0.02em] text-fg',
      className
    )}
    {...props}
  />
))
SharedModalTitle.displayName = DialogPrimitive.Title.displayName

const SharedModalDescription = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Description>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Description
    ref={ref}
    className={cn('mt-2 text-[length:var(--density-type-control)] leading-[1.5] text-fg-dim', className)}
    {...props}
  />
))
SharedModalDescription.displayName = DialogPrimitive.Description.displayName

function SharedModalBody({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('px-5 py-4', className)} {...props} />
}

function SharedModalActionStrip({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'flex flex-col-reverse gap-2.5 border-t border-border-soft bg-panel-strong/70 px-5 py-3 sm:flex-row sm:justify-end',
        className
      )}
      {...props}
    />
  )
}

export {
  SharedModal,
  SharedModalActionStrip,
  SharedModalBody,
  SharedModalContent,
  SharedModalDescription,
  SharedModalHeader,
  SharedModalTitle
}
