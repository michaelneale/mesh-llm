import * as React from 'react'

import { cn } from '@/lib/utils'

const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          'ui-field flex min-h-[80px] w-full rounded-md border px-3 py-2 text-sm outline-none transition-[border-color,background,box-shadow,color] duration-150 ease-out',
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Textarea.displayName = 'Textarea'

export { Textarea }
