import * as DialogPrimitive from '@radix-ui/react-dialog'
import { cva, type VariantProps } from 'class-variance-authority'
import * as React from 'react'
import { cn } from '@/lib/cn'

const Dialog = DialogPrimitive.Root

const CommandBarDialogOverlay = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={cn(
      'surface-scrim fixed inset-0 z-50 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
      className
    )}
    {...props}
  />
))
CommandBarDialogOverlay.displayName = DialogPrimitive.Overlay.displayName

type CommandBarDialogContentProps = React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content> & {
  overlayClassName?: string
}

const DialogContent = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Content>,
  CommandBarDialogContentProps
>(({ className, children, overlayClassName, ...props }, ref) => (
  <DialogPrimitive.Portal>
    <CommandBarDialogOverlay className={overlayClassName} />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        'shadow-surface-modal fixed left-1/2 top-1/2 z-50 grid w-[min(720px,calc(100vw-1.5rem))] max-h-[78vh] -translate-x-1/2 -translate-y-1/2 gap-4 overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel text-foreground outline-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
        className
      )}
      {...props}
    >
      {children}
    </DialogPrimitive.Content>
  </DialogPrimitive.Portal>
))
DialogContent.displayName = DialogPrimitive.Content.displayName

const DialogTitle = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title
    ref={ref}
    className={cn('text-[length:var(--density-type-body)] font-semibold text-foreground', className)}
    {...props}
  />
))
DialogTitle.displayName = DialogPrimitive.Title.displayName

const DialogDescription = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Description>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Description
    ref={ref}
    className={cn('text-[length:var(--density-type-caption)] text-muted-foreground', className)}
    {...props}
  />
))
DialogDescription.displayName = DialogPrimitive.Description.displayName

const alertVariants = cva(
  'relative w-full rounded-[var(--radius)] border px-3 py-2.5 text-[length:var(--density-type-caption)]',
  {
    variants: {
      variant: {
        default: 'border-border bg-background text-foreground',
        destructive:
          'border-[color:color-mix(in_oklch,var(--color-destructive)_48%,var(--color-border))] bg-[color:color-mix(in_oklch,var(--color-destructive)_10%,var(--color-panel))] text-destructive'
      }
    },
    defaultVariants: {
      variant: 'default'
    }
  }
)

const Alert = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>
>(({ className, variant, ...props }, ref) => (
  <div ref={ref} role="alert" className={cn(alertVariants({ variant }), className)} {...props} />
))
Alert.displayName = 'CommandBarAlert'

const AlertTitle = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h4 ref={ref} className={cn('font-medium leading-none tracking-tight', className)} {...props} />
  )
)
AlertTitle.displayName = 'CommandBarAlertTitle'

const AlertDescription = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <p
      ref={ref}
      className={cn('mt-1 text-[length:var(--density-type-label)] text-current/85 [&_p]:leading-relaxed', className)}
      {...props}
    />
  )
)
AlertDescription.displayName = 'CommandBarAlertDescription'

function Badge({ className, ...props }: React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border border-border bg-background px-2 py-0.5 font-mono text-[length:var(--density-type-label)] font-medium text-muted-foreground',
        className
      )}
      {...props}
    />
  )
}
Badge.displayName = 'CommandBarBadge'

type InputProps = React.InputHTMLAttributes<HTMLInputElement>

const Input = React.forwardRef<HTMLInputElement, InputProps>(({ className, type, ...props }, ref) => (
  <input
    ref={ref}
    type={type}
    className={cn(
      'flex h-10 w-full rounded-[var(--radius)] border border-border bg-background px-3 py-2 text-[length:var(--density-type-control)] text-foreground placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50',
      className
    )}
    {...props}
  />
))
Input.displayName = 'CommandBarInput'

export { Alert, AlertDescription, AlertTitle, Badge, Dialog, DialogContent, DialogDescription, DialogTitle, Input }
