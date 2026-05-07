import * as DropdownMenuPrimitive from '@radix-ui/react-dropdown-menu'
import { forwardRef, type ComponentPropsWithoutRef, type ComponentRef } from 'react'
import { cn } from '@/lib/cn'

export function DropdownMenu(props: ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Root>) {
  return <DropdownMenuPrimitive.Root {...props} />
}

export const DropdownMenuTrigger = forwardRef<
  ComponentRef<typeof DropdownMenuPrimitive.Trigger>,
  ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Trigger>
>((props, ref) => <DropdownMenuPrimitive.Trigger ref={ref} {...props} />)
DropdownMenuTrigger.displayName = 'DropdownMenuTrigger'

export function DropdownMenuPortal(props: ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Portal>) {
  return <DropdownMenuPrimitive.Portal {...props} />
}

export const DropdownMenuSeparator = forwardRef<
  ComponentRef<typeof DropdownMenuPrimitive.Separator>,
  ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Separator>
>((props, ref) => <DropdownMenuPrimitive.Separator ref={ref} {...props} />)
DropdownMenuSeparator.displayName = 'DropdownMenuSeparator'

export const DropdownMenuContent = forwardRef<
  ComponentRef<typeof DropdownMenuPrimitive.Content>,
  ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Content>
>(({ className, align = 'end', collisionPadding = 8, sideOffset = 6, ...props }, ref) => (
  <DropdownMenuPrimitive.Portal>
    <DropdownMenuPrimitive.Content
      ref={ref}
      align={align}
      collisionPadding={collisionPadding}
      sideOffset={sideOffset}
      className={cn(
        'surface-menu-panel z-50 min-w-[150px] overflow-hidden rounded-[var(--radius)] p-1 text-[length:var(--density-type-caption)] text-fg outline-none',
        className
      )}
      {...props}
    />
  </DropdownMenuPrimitive.Portal>
))
DropdownMenuContent.displayName = 'DropdownMenuContent'

type DropdownMenuItemProps = ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Item> & {
  tone?: 'default' | 'destructive'
}

export const DropdownMenuItem = forwardRef<ComponentRef<typeof DropdownMenuPrimitive.Item>, DropdownMenuItemProps>(
  ({ className, tone = 'default', ...props }, ref) => (
    <DropdownMenuPrimitive.Item
      ref={ref}
      className={cn(
        'flex cursor-default select-none items-center gap-2 rounded-[calc(var(--radius)-2px)] px-2 py-1.5 outline-none transition-[background,color] focus:bg-panel focus:text-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
        tone === 'destructive' &&
          'text-destructive focus:bg-[color-mix(in_oklab,var(--color-destructive)_12%,var(--color-panel))] focus:text-destructive',
        className
      )}
      {...props}
    />
  )
)
DropdownMenuItem.displayName = 'DropdownMenuItem'
