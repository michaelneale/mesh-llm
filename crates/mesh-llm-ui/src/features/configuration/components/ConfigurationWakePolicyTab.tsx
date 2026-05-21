import { useCallback, useState } from 'react'
import { ArrowUpDown, Plus, SlidersHorizontal } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { SegmentedControl, type SegmentedControlOption } from '@/components/ui/SegmentedControl'
import { configurationNavigationIconClassName } from '@/features/configuration/components/configuration-navigation-class-names'
import {
  SettingsCategoryRail,
  SettingsRow,
  SettingsSection,
  SettingsSummaryBanner,
  type SettingsCategoryItem
} from '@/features/configuration/components/settings/SettingsScaffold'
import { AddReserveProviderDialog } from '@/features/reserves/components/AddReserveProviderDialog'
import { ProviderOrderList } from '@/features/reserves/components/ReservePolicySettingsForm'
import { getMeshProvider } from '@/features/reserves/lib/mesh-providers'
import { DEFAULT_PROVIDER_DRAFT, type ProviderDraft } from '@/features/reserves/lib/provider-draft'
import { DEFAULT_RESERVE_WAKE_POLICY_SETTINGS } from '@/features/reserves/lib/reserve-policy'
import type { ReserveWakePolicySettings } from '@/features/reserves/lib/reserve-types'

const WAKE_POLICY_CATEGORIES: readonly SettingsCategoryItem[] = [
  {
    id: 'provider-order',
    label: 'Providers',
    summary: 'Reserve provider priority',
    count: 1,
    icon: <ArrowUpDown className={configurationNavigationIconClassName} />
  },
  {
    id: 'wake-behavior',
    label: 'Wake behavior',
    summary: 'When reserves automatically spin up',
    count: 4,
    icon: <SlidersHorizontal className={configurationNavigationIconClassName} />
  }
]

const WAKE_MODE_OPTIONS: readonly SegmentedControlOption[] = [
  { value: 'true', label: 'Enabled' },
  { value: 'false', label: 'Paused' }
]

const UTILIZATION_THRESHOLD_OPTIONS: readonly SegmentedControlOption[] = [65, 75, 85].map((threshold) => ({
  value: String(threshold),
  label: `${threshold}%`
}))

const SUSTAINED_FOR_OPTIONS: readonly SegmentedControlOption[] = [30, 60, 120].map((seconds) => ({
  value: String(seconds),
  label: seconds < 60 ? `${seconds}s` : `${seconds / 60} min`
}))

const SLEEP_IDLE_OPTIONS: readonly SegmentedControlOption[] = [5, 8, 12].map((minutes) => ({
  value: String(minutes),
  label: `${minutes} min`
}))

function selectCategory(categoryId: string) {
  document.getElementById(categoryId)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

export function ConfigurationWakePolicyTab() {
  const [settings, setSettings] = useState<ReserveWakePolicySettings>(DEFAULT_RESERVE_WAKE_POLICY_SETTINGS)
  const [activeCategoryId, setActiveCategoryId] = useState(WAKE_POLICY_CATEGORIES[0].id)
  const [addProviderOpen, setAddProviderOpen] = useState(false)
  const [providerDraft, setProviderDraft] = useState<ProviderDraft>(DEFAULT_PROVIDER_DRAFT)

  const handleCategorySelect = useCallback((id: string) => {
    setActiveCategoryId(id)
    selectCategory(id)
  }, [])

  function set<K extends keyof ReserveWakePolicySettings>(key: K, value: ReserveWakePolicySettings[K]) {
    setSettings((current) => ({ ...current, [key]: value }))
  }

  function resetProviderDraft() {
    setProviderDraft(DEFAULT_PROVIDER_DRAFT)
  }

  function handleAddProvider() {
    const selectedMeshProvider = getMeshProvider(providerDraft.providerId)
    if (selectedMeshProvider.availability !== 'supported') return

    const normalizedName = providerDraft.name.trim() || selectedMeshProvider.defaultName
    set('providerOrder', [...settings.providerOrder, normalizedName])
    resetProviderDraft()
  }

  return (
    <section
      aria-labelledby="wake-policy-summary-heading"
      className="space-y-3.5 pt-2"
      data-screen-label="Configuration · wake-policy"
    >
      <SettingsSummaryBanner
        description="Tune when reserves wake, how long demand must persist before triggering a spin-up, and which provider categories to try first. Backend persistence is still being wired."
        status="preview only"
        title="Reserves"
      />

      <div className="grid min-w-0 gap-4 xl:grid-cols-[200px_minmax(0,1fr)]">
        <SettingsCategoryRail
          activeId={activeCategoryId}
          categories={WAKE_POLICY_CATEGORIES}
          footer={null}
          onSelect={handleCategorySelect}
        />

        <div className="min-w-0 space-y-3.5">
          <SettingsSection
            id="provider-order"
            icon={<ArrowUpDown aria-hidden="true" className="size-[18px]" strokeWidth={1.9} />}
            title="Providers"
            subtitle="Drag to reorder which reserve provider categories are tried first when demand rises"
          >
            <div className="space-y-3 border-t border-border-soft pt-3">
              <ProviderOrderList
                providers={settings.providerOrder}
                onReorder={(reordered) => set('providerOrder', reordered)}
              />
              <Button
                className="ui-control h-8 px-3.5 rounded-[var(--radius)] border text-[length:var(--density-type-control)]"
                onClick={() => setAddProviderOpen(true)}
                size="sm"
                type="button"
                variant="outline"
              >
                <Plus className="mr-1.5 size-3.5" aria-hidden="true" />
                Add provider
              </Button>
            </div>
          </SettingsSection>

          <SettingsSection
            id="wake-behavior"
            icon={<SlidersHorizontal aria-hidden="true" className="size-[18px]" strokeWidth={1.9} />}
            title="Wake behavior"
            subtitle="Control when reserves automatically spin up and how long they idle before sleeping"
          >
            <SettingsRow
              className="border-t-0"
              label="Wake mode"
              hint="Enable or pause automatic reserve wake-up when mesh utilization rises"
            >
              <SegmentedControl
                ariaLabel="Wake mode"
                name="wake-policy-auto-wake"
                options={WAKE_MODE_OPTIONS}
                value={String(settings.autoWakeEnabled)}
                variant="pill"
                onValueChange={(value) => set('autoWakeEnabled', value === 'true')}
              />
            </SettingsRow>

            <SettingsRow
              label="Utilization threshold"
              hint="Mesh GPU utilization percentage that triggers a reserve wake"
            >
              <SegmentedControl
                ariaLabel="Utilization threshold"
                name="wake-policy-utilization-threshold"
                options={UTILIZATION_THRESHOLD_OPTIONS}
                value={String(settings.thresholdPercent)}
                variant="pill"
                onValueChange={(value) => set('thresholdPercent', Number(value))}
              />
            </SettingsRow>

            <SettingsRow
              label="Sustained for"
              hint="How long utilization must stay above the threshold before waking a reserve"
            >
              <SegmentedControl
                ariaLabel="Sustained for"
                name="wake-policy-sustained-for"
                options={SUSTAINED_FOR_OPTIONS}
                value={String(settings.sustainedSeconds)}
                variant="pill"
                onValueChange={(value) => set('sustainedSeconds', Number(value))}
              />
            </SettingsRow>

            <SettingsRow
              label="Sleep idle reserves"
              hint="Minutes of inactivity before cloud reserves return to standby"
            >
              <SegmentedControl
                ariaLabel="Sleep idle reserves"
                name="wake-policy-sleep-idle"
                options={SLEEP_IDLE_OPTIONS}
                value={String(settings.idleMinutes)}
                variant="pill"
                onValueChange={(value) => set('idleMinutes', Number(value))}
              />
            </SettingsRow>
          </SettingsSection>
        </div>
      </div>

      <AddReserveProviderDialog
        confirmLabel="Add to priority list"
        description="Stage a reserve provider in the local priority list. This preview does not provision infrastructure."
        onDraftChange={setProviderDraft}
        onConfirm={handleAddProvider}
        onOpenChange={(open) => {
          setAddProviderOpen(open)
          if (!open) resetProviderDraft()
        }}
        open={addProviderOpen}
        providerDraft={providerDraft}
      />
    </section>
  )
}
