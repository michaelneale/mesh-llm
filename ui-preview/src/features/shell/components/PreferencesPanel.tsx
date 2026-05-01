import * as RadioGroup from '@radix-ui/react-radio-group'
import { Moon, Sun, X } from 'lucide-react'
import { SegmentedControl } from '@/components/ui/SegmentedControl'
import { cn } from '@/lib/cn'
import { useDataMode, type DataMode } from '@/lib/data-mode'
import type { Accent, Density, PanelStyle, Theme } from '@/features/app-tabs/types'

type PreferencesPanelProps = {
  open: boolean
  theme: Theme
  accent: Accent
  density: Density
  panelStyle: PanelStyle
  onThemeChange: (theme: Theme) => void
  onAccentChange: (accent: Accent) => void
  onDensityChange: (density: Density) => void
  onPanelStyleChange: (style: PanelStyle) => void
  onClose: () => void
}

const themes: { value: Theme; label: string; Icon: typeof Moon }[] = [
  { value: 'dark', label: 'Dark', Icon: Moon },
  { value: 'light', label: 'Light', Icon: Sun },
]

const accents: { value: Accent; color: string }[] = [
  { value: 'blue', color: 'oklch(0.68 0.18 258)' },
  { value: 'cyan', color: 'oklch(0.76 0.16 195)' },
  { value: 'violet', color: 'oklch(0.75 0.14 292)' },
  { value: 'green', color: 'oklch(0.76 0.16 132)' },
  { value: 'amber', color: 'oklch(0.78 0.15 76)' },
]

const densities: { value: Density; label: string }[] = [
  { value: 'compact', label: 'Compact' },
  { value: 'normal', label: 'Normal' },
  { value: 'sparse', label: 'Sparse' },
]

const panelStyles: { value: PanelStyle; label: string }[] = [
  { value: 'solid', label: 'Solid' },
  { value: 'soft', label: 'Soft' },
]

const dataModes: { value: DataMode; label: string }[] = [
  { value: 'harness', label: 'Harness' },
  { value: 'live', label: 'Live API' },
]

const themeOptions = themes.map(({ value, label, Icon }) => ({
  value,
  label: (
    <>
      <Icon aria-hidden="true" className="size-[var(--preferences-icon-size)]" strokeWidth={1.8} />
      {label}
    </>
  ),
}))

const densityOptions = densities.map(({ value, label }) => ({ value, label }))
const panelStyleOptions = panelStyles.map(({ value, label }) => ({ value, label }))
const dataModeOptions = dataModes.map(({ value, label }) => ({ value, label }))

function isTheme(value: string): value is Theme {
  return themes.some((option) => option.value === value)
}

function isAccent(value: string): value is Accent {
  return accents.some((option) => option.value === value)
}

function isDensity(value: string): value is Density {
  return densities.some((option) => option.value === value)
}

function isPanelStyle(value: string): value is PanelStyle {
  return panelStyles.some((option) => option.value === value)
}

function isDataMode(value: string): value is DataMode {
  return dataModes.some((option) => option.value === value)
}

export function PreferencesPanel({
  open,
  theme,
  accent,
  density,
  panelStyle,
  onThemeChange,
  onAccentChange,
  onDensityChange,
  onPanelStyleChange,
  onClose,
}: PreferencesPanelProps) {
  const { mode, setMode } = useDataMode()

  if (!open) return null

  return (
    <aside
      aria-label="Interface preferences"
      className="panel-shell shadow-surface-modal fixed right-[var(--preferences-offset-right)] top-[var(--preferences-offset-top)] z-40 w-[var(--preferences-width)] rounded-[var(--radius-lg)] border border-border bg-panel pb-[var(--preferences-bottom-pad)]"
    >
      <div className="flex h-[var(--preferences-header-height)] items-center justify-between border-b border-border-soft px-[var(--preferences-pad-x)]">
        <h2 className="text-[length:var(--preferences-title-size)] font-semibold leading-none text-foreground">Preferences</h2>
        <button
          aria-label="Close preferences"
          className="ui-control grid size-[var(--preferences-close-size)] place-items-center rounded-[var(--radius)] border"
          onClick={onClose}
          type="button"
        >
          <X aria-hidden="true" className="size-[var(--preferences-close-icon-size)]" strokeWidth={1.8} />
        </button>
      </div>

      <div className="px-[var(--preferences-pad-x)] pt-[var(--preferences-body-top)]">
        <section aria-labelledby="theme-heading">
          <h3
            className="mb-[var(--preferences-heading-gap)] text-[length:var(--preferences-label-size)] font-medium uppercase leading-none tracking-[0.08em] text-fg-faint"
            id="theme-heading"
          >
            Theme
          </h3>
          <SegmentedControl
            ariaLabelledBy="theme-heading"
            className="grid grid-cols-2 gap-[var(--preferences-option-gap)]"
            itemClassName="flex h-[var(--preferences-control-height)] items-center justify-center gap-[var(--preferences-option-gap)] rounded-[var(--radius)] text-[length:var(--preferences-control-text-size)]"
            name="preferences-theme"
            onValueChange={(nextTheme) => {
              if (isTheme(nextTheme)) onThemeChange(nextTheme)
            }}
            options={themeOptions}
            value={theme}
          />
        </section>

        <section aria-labelledby="accent-heading" className="mt-[var(--preferences-section-gap)]">
          <h3
            className="mb-[var(--preferences-heading-gap)] text-[length:var(--preferences-label-size)] font-medium uppercase leading-none tracking-[0.08em] text-fg-faint"
            id="accent-heading"
          >
            Accent
          </h3>
          <RadioGroup.Root
            aria-labelledby="accent-heading"
            className="grid grid-cols-6 gap-[var(--preferences-option-gap)]"
            name="preferences-accent"
            onValueChange={(value) => {
              if (isAccent(value)) onAccentChange(value)
            }}
            value={accent}
          >
            {accents.map(({ value, color }) => {
              const selected = accent === value
              return (
                <RadioGroup.Item
                  aria-label={`Use ${value} accent`}
                  className={cn(
                    'h-[var(--preferences-swatch-height)] rounded-[var(--radius)] outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent active:translate-y-px',
                    selected
                      ? 'border border-foreground shadow-[var(--shadow-surface-selected)]'
                      : 'border border-border hover:border-foreground/70 hover:shadow-[var(--shadow-surface-hover)]',
                  )}
                  key={value}
                  style={{ backgroundColor: color }}
                  value={value}
                />
              )
            })}
          </RadioGroup.Root>
        </section>

        <section aria-labelledby="density-heading" className="mt-[var(--preferences-section-gap)]">
          <h3
            className="mb-[var(--preferences-heading-gap)] text-[length:var(--preferences-label-size)] font-medium uppercase leading-none tracking-[0.08em] text-fg-faint"
            id="density-heading"
          >
            Density
          </h3>
          <SegmentedControl
            ariaLabelledBy="density-heading"
            className="grid grid-cols-3 gap-[var(--preferences-option-gap)]"
            itemClassName="h-[var(--preferences-control-height)] rounded-[var(--radius)] text-[length:var(--preferences-control-text-size)]"
            name="preferences-density"
            onValueChange={(nextDensity) => {
              if (isDensity(nextDensity)) onDensityChange(nextDensity)
            }}
            options={densityOptions}
            value={density}
          />
        </section>

        <section aria-labelledby="panel-style-heading" className="mt-[var(--preferences-section-gap)]">
          <h3
            className="mb-[var(--preferences-heading-gap)] text-[length:var(--preferences-label-size)] font-medium uppercase leading-none tracking-[0.08em] text-fg-faint"
            id="panel-style-heading"
          >
            Panels
          </h3>
          <SegmentedControl
            ariaLabelledBy="panel-style-heading"
            className="grid grid-cols-2 gap-[var(--preferences-option-gap)]"
            itemClassName="h-[var(--preferences-control-height)] rounded-[var(--radius)] text-[length:var(--preferences-control-text-size)]"
            name="preferences-panel-style"
            onValueChange={(nextStyle) => {
              if (isPanelStyle(nextStyle)) onPanelStyleChange(nextStyle)
            }}
            options={panelStyleOptions}
            value={panelStyle}
          />
        </section>

        <section aria-labelledby="data-source-heading" className="mt-[var(--preferences-section-gap)]">
          <h3
            className="mb-[var(--preferences-heading-gap)] text-[length:var(--preferences-label-size)] font-medium uppercase leading-none tracking-[0.08em] text-fg-faint"
            id="data-source-heading"
          >
            Data source
          </h3>
          <SegmentedControl
            ariaLabelledBy="data-source-heading"
            className="grid grid-cols-2 gap-[var(--preferences-option-gap)]"
            itemClassName="h-[var(--preferences-control-height)] rounded-[var(--radius)] text-[length:var(--preferences-control-text-size)]"
            name="preferences-data-source"
            onValueChange={(nextMode) => {
              if (isDataMode(nextMode)) setMode(nextMode)
            }}
            options={dataModeOptions}
            value={mode}
          />
        </section>
      </div>
    </aside>
  )
}
