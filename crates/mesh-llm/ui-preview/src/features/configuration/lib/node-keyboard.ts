import type { Placement } from '@/features/app-tabs/types'

type NodeKeyboardAttributesOptions = {
  collapsed: boolean
  placement: Placement
  gpuCount: number
  readOnly: boolean
  configurablePlacement: boolean
}

export function nodeKeyboardAttributes({
  collapsed,
  placement,
  gpuCount,
  readOnly,
  configurablePlacement
}: NodeKeyboardAttributesOptions) {
  const hasKeyboardGpuSlots = !collapsed && (placement === 'pooled' || gpuCount > 0)
  const keyShortcuts = [
    ...(hasKeyboardGpuSlots ? ['ArrowUp', 'ArrowDown', 'Shift+ArrowUp', 'Shift+ArrowDown'] : []),
    ...(hasKeyboardGpuSlots
      ? [
          'ArrowLeft',
          'ArrowRight',
          'Shift+ArrowLeft',
          'Shift+ArrowRight',
          'Alt+ArrowLeft',
          'Alt+ArrowRight',
          'Alt+Shift+ArrowLeft',
          'Alt+Shift+ArrowRight'
        ]
      : []),
    ...(!readOnly ? ['A'] : []),
    ...(!readOnly && configurablePlacement ? ['P', 'S'] : [])
  ].join(' ')
  const shortcutHelp = [
    hasKeyboardGpuSlots
      ? 'Use up and down arrows to select GPU slots, or hold Shift to move the selected model between GPU slots.'
      : null,
    hasKeyboardGpuSlots
      ? 'Use left and right arrows to select models in the current GPU slot, or hold Shift to jump to the first or last model in that slot.'
      : null,
    hasKeyboardGpuSlots
      ? 'Hold Alt with left or right to adjust context, or hold Alt and Shift to jump context.'
      : null,
    readOnly ? 'Remote node context is read-only.' : 'Press A to add a model.',
    !readOnly && configurablePlacement ? 'Press P or S to switch placement.' : null
  ]
    .filter((item): item is string => Boolean(item))
    .join(' ')

  return { keyShortcuts, shortcutHelp }
}
