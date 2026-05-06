import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '../../lib/utils';

const alertVariants = cva(
    'relative w-full rounded-lg border p-4 [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-2px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg]:text-foreground',
    {
      variants: {
        variant: {
          default: 'bg-background text-foreground',
          primary: 'border-primary/20 bg-primary/5 text-foreground [&>svg]:text-primary',
          amber: 'border-amber-500/30 bg-amber-500/5 text-foreground [&>svg]:text-amber-500',
          blue: 'border-blue-500/30 bg-blue-500/5 text-foreground [&>svg]:text-blue-500',
          destructive:
            'border-destructive/50 bg-destructive/10 text-destructive [&>svg]:text-destructive dark:border-destructive/70 dark:bg-destructive/20 dark:text-red-100 dark:[&>svg]:text-red-200',
        },
      },
      defaultVariants: {
        variant: 'default',
      },
    },
  );

const Alert = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>
>(({ className, variant, ...props }, ref) => (
  <div ref={ref} role="alert" className={cn(alertVariants({ variant }), className)} {...props} />
));
Alert.displayName = 'Alert';

const AlertTitle = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h4 ref={ref} className={cn('mb-1 font-medium leading-none tracking-tight', className)} {...props} />
  ),
);
AlertTitle.displayName = 'AlertTitle';

const AlertDescription = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <p ref={ref} className={cn('text-sm [&_p]:leading-relaxed', className)} {...props} />
  ),
);
AlertDescription.displayName = 'AlertDescription';

export { Alert, AlertDescription, AlertTitle };
