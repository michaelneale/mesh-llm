import * as Select from '@radix-ui/react-select'
import { ChevronDown, Cpu } from 'lucide-react'
import { memo } from 'react'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import { cn } from '@/lib/cn'
import type { ModelSelectOption } from '@/features/app-tabs/types'

type ModelSelectProps = {
  options: ModelSelectOption[]
  value: string
  onChange: (value: string) => void
  className?: string
}

function statusBadgeTone(tone: NonNullable<ModelSelectOption['status']>['tone'] = 'good'): StatusBadgeTone {
  return tone
}

const ModelSelectImpl = ({ options, value, onChange, className }: ModelSelectProps) => {
  const hasOptions = options.length > 0
  const selectedOption = options.find((option) => option.value === value)
  const selected = selectedOption ?? options[0]
  const selectedLabel = selected?.value === 'auto' ? 'Auto (router picks best)' : (selected?.label ?? 'Select model')
  return (
    <Select.Root value={selectedOption?.value} onValueChange={onChange} disabled={!hasOptions}>
      <Select.Trigger
        aria-label="Select model"
        className={cn(
          'ui-control inline-flex min-w-0 items-center gap-2 rounded-[var(--radius)] border px-2.5 py-[5px] font-mono text-[length:var(--density-type-control)]',
          'w-full sm:w-auto sm:min-w-[220px]',
          className
        )}
      >
        <Cpu className="size-3 shrink-0 text-fg-dim" />
        <span className="min-w-0 flex-1 truncate whitespace-nowrap text-left">{selectedLabel}</span>
        <Select.Icon asChild>
          <ChevronDown className="size-3 shrink-0 text-fg-dim" />
        </Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Content
          align="end"
          className="surface-floating-panel z-10 overflow-hidden rounded-[var(--radius)]"
          position="popper"
          side="bottom"
          sideOffset={4}
          style={{ minWidth: 260 }}
        >
          <Select.Viewport>
            {options.map((option, index) => (
              <Select.Item
                key={option.value}
                value={option.value}
                data-active={option.value === value ? 'true' : undefined}
                className={cn(
                  'ui-row-action flex w-full min-w-0 items-center justify-between gap-3 px-3 py-2 text-left text-[length:var(--density-type-control)] outline-none',
                  index < options.length - 1 ? 'border-b border-border-soft' : '',
                  option.value === value ? 'bg-[color-mix(in_oklab,var(--color-accent)_10%,transparent)]' : '',
                  'data-[highlighted]:bg-[color-mix(in_oklab,var(--color-accent)_8%,transparent)]'
                )}
              >
                <Select.ItemText>
                  <span className="block max-w-[18rem] truncate whitespace-nowrap font-mono">{option.label}</span>
                </Select.ItemText>
                {option.status ? (
                  <StatusBadge className="shrink-0" dot tone={statusBadgeTone(option.status.tone)}>
                    {option.status.label}
                  </StatusBadge>
                ) : null}
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  )
}

export const ModelSelect = memo(ModelSelectImpl)
