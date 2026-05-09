import * as React from 'react'
import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/buttonVariants'

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'secondary' | 'ghost' | 'outline' | 'destructive'
  size?: 'default' | 'sm' | 'lg' | 'icon'
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, ...props }, ref) => {
  return <button className={cn(buttonVariants({ variant, size }), className)} ref={ref} {...props} />
})
Button.displayName = 'Button'

export { Button }
