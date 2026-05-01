import { useMemo, type PointerEvent } from 'react'
import * as RadioGroup from '@radix-ui/react-radio-group'
import { NativeSelect } from '@/components/ui/NativeSelect'
import { SegmentedControl, type SegmentedControlOption } from '@/components/ui/SegmentedControl'
import { Slider } from '@/components/ui/Slider'
import { BrainCircuit, Cog, Cpu, MemoryStick, RotateCcw, type LucideIcon } from 'lucide-react'
import { configurationNavigationIconClassName } from '@/features/configuration/components/configuration-navigation-class-names'
import { SettingsCategoryRail, SettingsPreviewRail, SettingsRow, SettingsSection, SettingsSummaryBanner, type SettingsCategoryItem } from '@/features/configuration/components/settings/SettingsScaffold'
import { useDefaultsSettingsState } from '@/features/configuration/hooks/useDefaultsSettingsState'
import type { ConfigurationDefaultsCategory, ConfigurationDefaultsCategoryId, ConfigurationDefaultsControl, ConfigurationDefaultsHarnessData, ConfigurationDefaultsSetting, ConfigurationDefaultsValues } from '@/features/app-tabs/types'
import { cn } from '@/lib/cn'

const categoryIcons: Record<ConfigurationDefaultsCategoryId, LucideIcon> = {
  runtime: Cpu,
  memory: MemoryStick,
  'speculative-decoding': BrainCircuit,
  advanced: Cog,
}

const slotOptions = Array.from({ length: 16 }, (_, index) => index + 1)

const llamaFlavorColors: Record<string, string> = {
  cpu: 'var(--color-accent)',
  cuda: '#76b900',
  metal: '#8b5cf6',
  rocm: '#ed1c24',
  vulkan: '#ac162c',
}

function LlamaFlavorOption({ option, selected }: { option: SegmentedControlOption; selected: boolean }) {
  return (
    <>
      <span
        aria-hidden="true"
        className={cn('size-1.5 rounded-full', selected ? 'opacity-100' : 'opacity-65')}
        data-model-runtime={option.value}
        style={{ backgroundColor: llamaFlavorColors[option.value] ?? 'var(--color-accent)' }}
      />
      {option.label}
    </>
  )
}

const draftModelModeSettingId = 'speculation-mode'
const draftModelModeValue = 'draft_model'
const incompatiblePairingBehaviorSettingId = 'incompatible-pairing-behavior'
const draftModelOnlySettingIds = new Set(['draft-selection-policy', 'draft-max-tokens', 'draft-min-tokens', 'draft-acceptance-threshold', incompatiblePairingBehaviorSettingId])

type DefaultsTabProps = {
  data: ConfigurationDefaultsHarnessData
  values: ConfigurationDefaultsValues
  onSettingValueChange: (settingId: string, value: string) => void
  onResetAll?: () => void
  configFilePath?: string
}

function getSettingValue(setting: ConfigurationDefaultsSetting, values: ConfigurationDefaultsValues) {
  return values[setting.id] ?? setting.control.value
}

function settingKey(control: ConfigurationDefaultsControl, fallback: string) {
  return control.kind === 'metric' ? fallback.replaceAll('-', '_') : control.name
}

function tomlValue(value: string) {
  return /^\d+(\.\d+)?$/.test(value) ? value : JSON.stringify(value)
}

function isBooleanToggleChoice(setting: ConfigurationDefaultsSetting) {
  return setting.control.kind === 'choice'
    && setting.control.presentation === 'toggle'
    && setting.control.options.length === 2
    && setting.control.options.every((option) => option.value === 'on' || option.value === 'off')
}

function toggleTomlValue(value: string) {
  return value === 'on' ? 'true' : 'false'
}

type DefaultsPreviewLine =
  | { kind: 'blank'; id: string }
  | { kind: 'section'; id: string; value: string }
  | { kind: 'pair'; id: string; keyName: string; value: string }

function formatRangeValue(setting: ConfigurationDefaultsSetting, value: string) {
  if (setting.id === 'memory-margin') return Number(value).toFixed(1)
  if (setting.id === 'draft-acceptance-threshold') return Number(value).toFixed(2)
  if (setting.id === 'repeat-penalty') return Number(value).toFixed(2)
  return value
}

function rangeUnit(setting: ConfigurationDefaultsSetting, value: string) {
  if (setting.id === 'parallel-slots') return `slot${value === '1' ? '' : 's'}`
  if (setting.id === 'memory-margin') return 'GB'
  if (setting.id === 'draft-acceptance-threshold' || setting.id === 'repeat-penalty') return undefined
  if (setting.control.kind === 'range') return setting.control.unit
  return undefined
}

function formatMetricValue(control: ConfigurationDefaultsControl) {
  if (control.kind === 'metric' && control.unit) return `${control.value} ${control.unit}`
  if (control.kind === 'metric') return control.value
  return ''
}

function choiceItemClassName(setting: ConfigurationDefaultsSetting) {
  return cn(
    'min-w-[64px] capitalize',
    setting.control.kind === 'choice' && setting.control.presentation === 'toggle' && 'min-w-[38px]',
    setting.id === 'flash-attention' && 'min-w-[38px]',
    setting.id === 'kv-cache' && 'min-w-[58px]',
    setting.id === 'speculation-mode' && 'min-w-[72px]',
    setting.id === 'draft-selection-policy' && 'min-w-[86px]',
    setting.id === 'incompatible-pairing-behavior' && 'min-w-[104px]',
    setting.id === 'reasoning-format' && 'min-w-[58px]',
    setting.id === 'llamacpp-flavor' && 'min-w-[52px] gap-1.5 font-mono normal-case',
  )
}

function tomlPreviewValue(setting: ConfigurationDefaultsSetting, value: string) {
  if (isBooleanToggleChoice(setting)) return toggleTomlValue(value)
  if (setting.id === 'memory-margin') return Number(value).toFixed(1)
  if (setting.id === 'draft-acceptance-threshold') return Number(value).toFixed(2)
  if (setting.id === 'repeat-penalty') return Number(value).toFixed(2)
  return tomlValue(value)
}

function defaultsSectionId(categoryId: ConfigurationDefaultsCategoryId) {
  if (categoryId === 'advanced') return 'runtime'
  if (categoryId === 'speculative-decoding') return 'speculative_decoding'
  return null
}

function buildDefaultsPreviewLines(data: ConfigurationDefaultsHarnessData, values: ConfigurationDefaultsValues): DefaultsPreviewLine[] {
  const mainLines: DefaultsPreviewLine[] = [{ kind: 'section', id: 'defaults-section', value: '[defaults]' }]
  const sectionGroups = new Map<string, DefaultsPreviewLine[]>()
  const speculationModeSetting = data.settings.find((setting) => setting.id === draftModelModeSettingId)
  const speculationMode = speculationModeSetting ? getSettingValue(speculationModeSetting, values) : null

  for (const setting of data.settings) {
    if (setting.id === incompatiblePairingBehaviorSettingId && speculationMode !== draftModelModeValue) continue

    const value = getSettingValue(setting, values)
    if (setting.control.kind === 'text' && value.trim().length === 0) continue

    const line: DefaultsPreviewLine = {
      kind: 'pair',
      id: setting.id,
      keyName: settingKey(setting.control, setting.id),
      value: tomlPreviewValue(setting, value),
    }
    const sectionId = defaultsSectionId(setting.categoryId)
    if (!sectionId) {
      mainLines.push(line)
      continue
    }

    const groupLines = sectionGroups.get(sectionId) ?? [{ kind: 'section', id: `defaults-${sectionId}-section`, value: `[defaults.${sectionId}]` }]
    groupLines.push(line)
    sectionGroups.set(sectionId, groupLines)
  }

  const groupedLines = Array.from(sectionGroups.entries()).flatMap(([sectionId, lines]) => [{ kind: 'blank' as const, id: `defaults-preview-${sectionId}-spacer` }, ...lines])
  return [...mainLines, ...groupedLines]
}

function renderDefaultsPreview(lines: readonly DefaultsPreviewLine[]) {
  return lines.map((line) => {
    if (line.kind === 'blank') return ''
    if (line.kind === 'section') return line.value
    return `${line.keyName} = ${line.value}`
  }).join('\n')
}

function settingDescription(setting: ConfigurationDefaultsSetting) {
  return setting.description
}

function sectionSubtitle(category: ConfigurationDefaultsCategory) {
  if (category.id === 'memory') return 'VRAM accounting and KV cache policy'
  if (category.id === 'speculative-decoding') return 'Speculative draft model and acceptance defaults'
  if (category.id === 'advanced') return 'Reasoning and sampling defaults'
  return category.help
}

function RangeControl({ disabled = false, setting, value, onChange }: { disabled?: boolean; setting: ConfigurationDefaultsSetting; value: string; onChange: (value: string) => void }) {
  if (setting.control.kind !== 'range') return null

  return (
    <Slider
      ariaLabel={setting.label}
      disabled={disabled}
      max={setting.control.max}
      min={setting.control.min}
      name={setting.control.name}
      onValueChange={onChange}
      step={setting.control.step}
      formatValue={(nextValue) => formatRangeValue(setting, nextValue)}
      unit={rangeUnit(setting, value)}
      value={value}
      valueLabelAlign="right"
      valueLabelPlacement="bottom"
    />
  )
}

function TextControl({ disabled = false, setting, value, onChange }: { disabled?: boolean; setting: ConfigurationDefaultsSetting; value: string; onChange: (value: string) => void }) {
  if (setting.control.kind !== 'text') return null

  return (
    <input
      aria-label={setting.label}
      className={cn('ui-control h-[32px] w-full min-w-[220px] rounded-[var(--radius)] border bg-surface px-2.5 font-mono text-[length:var(--density-type-control)] text-foreground outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent', disabled && 'cursor-not-allowed opacity-60')}
      disabled={disabled}
      name={setting.control.name}
      onChange={(event) => onChange(event.currentTarget.value)}
      placeholder={setting.control.placeholder}
      value={value}
    />
  )
}

function SlotCountControl({ setting, value, onChange }: { setting: ConfigurationDefaultsSetting; value: string; onChange: (value: string) => void }) {
  if (setting.control.kind !== 'range') return null

  const control = setting.control
  const selectedSlots = Math.max(control.min, Math.min(control.max, Number(value)))
  const estimatedGb = (selectedSlots * 0.3).toFixed(1)
  const selectSlotFromPointer = (event: PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    if (bounds.width <= 0) return

    const clampedX = Math.max(0, Math.min(bounds.width, event.clientX - bounds.left))
    const slotCount = Math.max(control.min, Math.min(control.max, Math.ceil((clampedX / bounds.width) * control.max)))

    onChange(String(slotCount))
  }

  return (
    <RadioGroup.Root
      aria-label={setting.label}
      className="rounded-[6px] border border-border-soft bg-panel-strong px-2.5 py-2 text-[length:var(--density-type-caption)] text-fg-dim"
      name={control.name}
      onValueChange={onChange}
      value={String(selectedSlots)}
    >
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <span className="text-fg-faint">est. KV @ 16K ctx</span>
        <span className="font-mono text-fg-dim">{estimatedGb} GB · {selectedSlots} × 0.30 GB</span>
      </div>
      <div
        className="inline-grid touch-none select-none grid-cols-[repeat(16,18px)] gap-px"
        data-testid="defaults-slot-meter"
        onPointerDown={(event) => {
          event.preventDefault()
          if (event.currentTarget.setPointerCapture) event.currentTarget.setPointerCapture(event.pointerId)
          selectSlotFromPointer(event)
        }}
        onPointerMove={(event) => {
          if (event.currentTarget.hasPointerCapture && !event.currentTarget.hasPointerCapture(event.pointerId) && event.buttons !== 1) return
          selectSlotFromPointer(event)
        }}
        onPointerUp={(event) => {
          if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
        }}
      >
        {slotOptions.map((slotCount) => (
          <RadioGroup.Item
            aria-label={`${slotCount} slot${slotCount === 1 ? '' : 's'}`}
            className="group flex w-[18px] cursor-pointer appearance-none items-center rounded-[2px] border-0 bg-transparent py-1 outline-none transition-[opacity] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
            key={slotCount}
            value={String(slotCount)}
          >
            <span
              className={cn(
                'h-1.5 w-full rounded-[1px] transition-colors',
                slotCount <= selectedSlots ? 'bg-accent opacity-100' : 'bg-border-soft opacity-50',
              )}
              data-slot-empty={slotCount > selectedSlots ? 'true' : undefined}
            />
          </RadioGroup.Item>
        ))}
      </div>
    </RadioGroup.Root>
  )
}

function KvPolicyMatrix({ policy }: { policy: string }) {
  const rows = [
    { label: '<5GB', detail: 'K F16 · V F16', active: policy === 'auto' || policy === 'quality' },
    { label: '5–50GB', detail: 'K q8_0 · V q4_0', active: policy === 'auto' || policy === 'balanced' },
    { label: '≥50GB', detail: 'K q4_0 · V q4_0', active: policy === 'auto' || policy === 'saver' },
  ]

  return (
    <fieldset className="mt-2 grid max-w-[420px] grid-cols-3 gap-1.5 font-mono text-[length:var(--density-type-annotation)]" aria-label="KV cache memory tiers">
      {rows.map((row) => (
        <span
          className={cn(
            'rounded-[5px] border px-2 py-1.5 text-fg-faint transition-[background-color,border-color,opacity]',
            row.active
              ? 'border-[color:color-mix(in_oklab,var(--color-accent)_35%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_6%,var(--color-panel-strong))] opacity-100'
              : 'border-border-soft bg-panel-strong opacity-50',
          )}
          data-kv-tier-active={row.active ? 'true' : 'false'}
          key={row.label}
        >
          <span className="block text-[9.5px] uppercase tracking-[0.05em]">{row.label}</span>
          <span className="mt-0.5 block text-fg-dim">{row.detail}</span>
        </span>
      ))}
    </fieldset>
  )
}

function SettingControl({ disabled = false, setting, value, onChange }: { disabled?: boolean; setting: ConfigurationDefaultsSetting; value: string; onChange: (value: string) => void }) {
  return (
    <div aria-disabled={disabled ? 'true' : undefined} data-setting-control-disabled={disabled ? 'true' : undefined}>
      {setting.control.kind === 'choice' && setting.control.presentation === 'select' ? (
        <NativeSelect ariaLabel={setting.label} disabled={disabled} name={setting.control.name} onValueChange={onChange} options={setting.control.options} value={value} />
      ) : null}
      {setting.control.kind === 'choice' && setting.control.presentation !== 'select' ? (
        <SegmentedControl
          ariaLabel={setting.label}
          disabled={disabled}
          itemClassName={choiceItemClassName(setting)}
          name={setting.control.name}
          onValueChange={onChange}
          options={setting.control.options}
          renderOption={setting.id === 'llamacpp-flavor' ? (option, selected) => <LlamaFlavorOption option={option} selected={selected} /> : undefined}
          value={value}
          variant="pill"
        />
      ) : null}
      {setting.id === 'parallel-slots' ? <SlotCountControl setting={setting} value={value} onChange={onChange} /> : <RangeControl disabled={disabled} setting={setting} value={value} onChange={onChange} />}
      <TextControl disabled={disabled} setting={setting} value={value} onChange={onChange} />
      {setting.control.kind === 'metric' ? (
        <span className="rounded-[var(--radius)] border border-border-soft bg-surface px-2.5 py-1.5 font-mono text-[length:var(--density-type-control)] text-fg-dim">
          {formatMetricValue(setting.control)}
        </span>
      ) : null}
      {setting.id === 'kv-cache' ? <KvPolicyMatrix policy={value} /> : null}
    </div>
  )
}

function DefaultsSection({ category, settings, values, onSettingValueChange }: { category: ConfigurationDefaultsCategory; settings: readonly ConfigurationDefaultsSetting[]; values: ConfigurationDefaultsValues; onSettingValueChange: (settingId: string, value: string) => void }) {
  const Icon = categoryIcons[category.id]
  const speculationModeSetting = settings.find((setting) => setting.id === draftModelModeSettingId)
  const speculationMode = category.id === 'speculative-decoding' && speculationModeSetting ? getSettingValue(speculationModeSetting, values) : null
  const draftModelControlsDisabled = speculationMode !== null && speculationMode !== draftModelModeValue
  const isSettingDisabled = (setting: ConfigurationDefaultsSetting) => draftModelControlsDisabled && draftModelOnlySettingIds.has(setting.id)

  return (
    <SettingsSection id={`defaults-${category.id}`} icon={<Icon aria-hidden="true" className="size-[18px]" strokeWidth={1.9} />} title={category.label} subtitle={sectionSubtitle(category)}>
      {settings.map((setting, settingIndex) => {
        const value = getSettingValue(setting, values)
        const disabled = isSettingDisabled(setting)

        return (
          <SettingsRow className={cn(settingIndex === 0 && 'border-t-0', disabled && 'opacity-55')} key={setting.id} label={setting.label} hint={settingDescription(setting)}>
            <SettingControl disabled={disabled} setting={setting} value={value} onChange={(nextValue) => onSettingValueChange(setting.id, nextValue)} />
          </SettingsRow>
        )
      })}
    </SettingsSection>
  )
}

export function DefaultsTab({ data, values, onSettingValueChange, onResetAll, configFilePath }: DefaultsTabProps) {
  const { activeCategoryId, setActiveCategoryId } = useDefaultsSettingsState(data)
  const changedCount = data.settings.filter((setting) => getSettingValue(setting, values) !== setting.control.value).length
  const previewLines = useMemo(() => buildDefaultsPreviewLines(data, values), [data, values])
  const categories: SettingsCategoryItem[] = useMemo(
    () => data.categories.map((category) => {
      const Icon = categoryIcons[category.id]

      return {
        ...category,
        count: data.settings.filter((setting) => setting.categoryId === category.id).length,
        icon: <Icon aria-hidden="true" className={configurationNavigationIconClassName} strokeWidth={1.7} />,
      }
    }),
    [data.categories, data.settings],
  )

  const selectCategory = (categoryId: string) => {
    setActiveCategoryId(categoryId)
    const target = document.getElementById(`defaults-${categoryId}`)
    if (target && 'scrollIntoView' in target) target.scrollIntoView({ block: 'start', behavior: 'smooth' })
  }

  return (
    <section aria-labelledby="defaults-summary-heading" className="space-y-3.5 pt-2" data-screen-label="Configuration · defaults">
      <SettingsSummaryBanner
        action={(
          <button className={cn('ui-control inline-flex h-[30px] items-center gap-1.5 rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-control)] font-semibold')} disabled={changedCount === 0} onClick={onResetAll} type="button">
            <RotateCcw aria-hidden="true" className="size-3.5" />
            Reset all
          </button>
        )}
        description={<>These values flow into every <span className="rounded border border-border-soft bg-surface px-1 font-mono text-foreground">[[models]]</span> placement unless that placement explicitly overrides them. Per-placement overrides surface as <span className="rounded border border-border-soft bg-surface px-1 font-mono text-accent">OVERRIDE</span> badges in Model Deployment.</>}
        status={changedCount === 0 ? 'all upstream' : `${changedCount} modified`}
        title="Inherited defaults"
      />

      <div className="grid gap-4 xl:grid-cols-[200px_minmax(0,1fr)_280px]">
        <SettingsCategoryRail
          activeId={activeCategoryId}
          categories={categories}
          footer={(
            <span className="inline-flex flex-col gap-1 whitespace-nowrap leading-none">
              <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-fg-faint">Configuration Path</span>
              <span className="font-mono text-fg-dim">{configFilePath ?? '~/.mesh-llm/config.toml'}</span>
            </span>
          )}
          onSelect={selectCategory}
        />

        <div className="space-y-3.5">
          {data.categories.map((category) => (
            <DefaultsSection
              category={category}
              key={category.id}
              onSettingValueChange={onSettingValueChange}
              settings={data.settings.filter((setting) => setting.categoryId === category.id)}
              values={values}
            />
          ))}
        </div>

        <SettingsPreviewRail title="[defaults]" code={renderDefaultsPreview(previewLines)} tip={<>Adjust placements in <span className="text-foreground">Model Deployment</span> to override these for a single model.</>} />
      </div>
    </section>
  )
}
