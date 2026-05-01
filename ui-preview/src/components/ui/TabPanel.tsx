import * as Tabs from '@radix-ui/react-tabs'
import { useState, type CSSProperties, type ElementType, type ReactNode } from 'react'
import { Tooltip } from '@/components/ui/Tooltip'
import { cn } from '@/lib/cn'

type TabPanelIcon = ElementType<{ 'aria-hidden'?: boolean; className?: string; strokeWidth?: number }>

export type TabPanelRenderContext<TValue extends string = string> = {
  active: boolean
  disabled: boolean
  value: TValue
}

export type TabPanelItem<TValue extends string = string> = {
  value: TValue
  label: ReactNode
  content: ReactNode
  accessory?: ReactNode | ((context: TabPanelRenderContext<TValue>) => ReactNode)
  contentClassName?: string
  description?: string
  disabled?: boolean
  icon?: TabPanelIcon
  iconClassName?: string
  renderIcon?: (context: TabPanelRenderContext<TValue>) => ReactNode
  triggerAttributes?: Record<`data-${string}`, string | undefined>
  triggerClassName?: string
}

export type TabPanelProps<TValue extends string = string> = {
  ariaLabel?: string
  ariaLabelledBy?: string
  className?: string
  contentClassName?: string
  defaultValue?: TValue
  iconClassName?: string
  iconStrokeWidth?: number
  listClassName?: string
  tabBarAccessory?: ReactNode
  tabBarClassName?: string
  tabs: readonly TabPanelItem<TValue>[]
  triggerClassName?: string
  value?: TValue
  onValueChange?: (value: TValue) => void
}

function findEnabledValue<TValue extends string>(tabs: readonly TabPanelItem<TValue>[]) {
  return tabs.find((tab) => !tab.disabled)?.value
}

function hasEnabledTabValue<TValue extends string>(tabs: readonly TabPanelItem<TValue>[], value: TValue | undefined) {
  return value !== undefined && tabs.some((tab) => tab.value === value && !tab.disabled)
}

function renderAccessory<TValue extends string>(item: TabPanelItem<TValue>, context: TabPanelRenderContext<TValue>) {
  if (typeof item.accessory === 'function') return item.accessory(context)
  return item.accessory
}

export function TabPanel<TValue extends string = string>({
  ariaLabel,
  ariaLabelledBy,
  className,
  contentClassName,
  defaultValue,
  iconClassName,
  iconStrokeWidth = 1.6,
  listClassName,
  tabBarAccessory,
  tabBarClassName,
  tabs,
  triggerClassName,
  value,
  onValueChange,
}: TabPanelProps<TValue>) {
  const fallbackValue = hasEnabledTabValue(tabs, defaultValue) ? defaultValue : findEnabledValue(tabs)
  const [internalValue, setInternalValue] = useState<TValue | undefined>(fallbackValue)
  const currentValue = value === undefined ? (hasEnabledTabValue(tabs, internalValue) ? internalValue : fallbackValue) : hasEnabledTabValue(tabs, value) ? value : undefined

  return (
    <Tabs.Root
      className={cn('bg-transparent', className)}
      onValueChange={(nextValue) => {
        const nextTab = tabs.find((tab) => tab.value === nextValue)
        if (!nextTab || nextTab.disabled) return

        if (value === undefined) setInternalValue(nextTab.value)
        onValueChange?.(nextTab.value)
      }}
      value={currentValue}
    >
      <div className={cn('panel-divider flex items-stretch border-b border-border bg-transparent px-2', tabBarClassName)}>
        <Tabs.List aria-label={ariaLabel} aria-labelledby={ariaLabelledBy} className={cn('flex h-[37px] items-stretch gap-0 overflow-x-auto', listClassName)}>
          {tabs.map((item) => {
            const active = currentValue === item.value
            const disabled = Boolean(item.disabled)
            const context: TabPanelRenderContext<TValue> = { active, disabled, value: item.value }
            const Icon = item.icon
            const triggerStyle: CSSProperties = {
              borderBottomColor: active ? 'var(--color-accent)' : 'transparent',
              color: active ? 'var(--color-foreground)' : 'var(--color-fg-faint)',
            }

            const trigger = (
              <Tabs.Trigger
                className={cn(
                  '-mb-px inline-flex h-[37px] items-center gap-[7px] whitespace-nowrap border-b-2 px-[14px] text-[12.5px] font-medium leading-none tracking-[-0.05px] outline-none transition-[border-color,color] hover:text-fg-dim focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50',
                  triggerClassName,
                  item.triggerClassName,
                )}
                data-active={active ? 'true' : undefined}
                disabled={disabled}
                key={item.value}
                style={triggerStyle}
                aria-description={item.description}
                value={item.value}
                {...item.triggerAttributes}
              >
                {item.renderIcon ? item.renderIcon(context) : Icon ? <Icon aria-hidden={true} className={cn('size-[13px] shrink-0', iconClassName, item.iconClassName)} strokeWidth={iconStrokeWidth} /> : null}
                {item.label}
                {renderAccessory(item, context)}
              </Tabs.Trigger>
            )

            if (!item.description) return trigger

            return (
              <Tooltip key={item.value} content={item.description} side="bottom">
                {trigger}
              </Tooltip>
            )
          })}
        </Tabs.List>
        {tabBarAccessory ? <div className="ml-auto flex items-center pl-2">{tabBarAccessory}</div> : null}
      </div>
      {tabs.map((item) => (
        <Tabs.Content className={cn('mt-0 px-5 pt-4 outline-none', contentClassName, item.contentClassName)} key={item.value} value={item.value}>
          {item.content}
        </Tabs.Content>
      ))}
    </Tabs.Root>
  )
}
