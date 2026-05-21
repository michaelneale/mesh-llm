import { Slot } from '@radix-ui/react-slot'
import { forwardRef, useId, type ComponentPropsWithoutRef, type ReactNode } from 'react'
import { cn } from '@/lib/cn'

type TextFieldProps = Omit<ComponentPropsWithoutRef<'input'>, 'size'> & {
  asChild?: boolean
  containerClassName?: string
  errorText?: ReactNode
  helperText?: ReactNode
  inputClassName?: string
  label: ReactNode
  labelClassName?: string
}

export const TextField = forwardRef<HTMLInputElement, TextFieldProps>(
  (
    {
      asChild = false,
      className,
      containerClassName,
      disabled,
      errorText,
      helperText,
      id,
      inputClassName,
      label,
      labelClassName,
      ...props
    },
    ref
  ) => {
    const generatedId = useId()
    const inputId = id ?? generatedId
    const helperId = helperText ? `${inputId}-helper` : undefined
    const errorId = errorText ? `${inputId}-error` : undefined
    const describedBy = [props['aria-describedby'], helperId, errorId].filter(Boolean).join(' ') || undefined
    const Control = asChild ? Slot : 'input'

    return (
      <div className={cn('space-y-1.5', containerClassName)}>
        <label className={cn('type-label text-fg-faint', disabled && 'opacity-60', labelClassName)} htmlFor={inputId}>
          {label}
        </label>
        <Control
          {...props}
          aria-describedby={describedBy}
          aria-invalid={errorText ? true : props['aria-invalid']}
          className={cn(
            'ui-field flex h-8 w-full rounded-[var(--radius)] border px-2 text-[length:var(--density-type-control)] leading-none outline-none transition-[border-color,background,box-shadow,color] duration-150 ease-out active:translate-y-0 active:transform-none',
            className,
            inputClassName
          )}
          disabled={disabled}
          id={inputId}
          ref={ref}
        />
        {helperText ? (
          <span className="block text-[length:var(--density-type-caption)] text-fg-faint" id={helperId}>
            {helperText}
          </span>
        ) : null}
        {errorText ? (
          <span className="block text-[length:var(--density-type-caption)] font-medium text-destructive" id={errorId}>
            {errorText}
          </span>
        ) : null}
      </div>
    )
  }
)

TextField.displayName = 'TextField'
