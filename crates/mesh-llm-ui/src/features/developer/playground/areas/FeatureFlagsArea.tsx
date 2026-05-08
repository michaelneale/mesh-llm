import { useMemo, useState, type ReactNode } from 'react'
import { Flag, Globe2, RotateCcw } from 'lucide-react'
import { SegmentedControl, type SegmentedControlOption } from '@/components/ui/SegmentedControl'
import { SidebarNavigation } from '@/components/ui/SidebarNavigation'
import { SettingsRow, SettingsSection } from '@/features/configuration/components/settings/SettingsScaffold'
import { cn } from '@/lib/cn'
import { type FeatureFlagDefinition, type FeatureFlagSectionId, useFeatureFlagSettings } from '@/lib/feature-flags'

const FEATURE_FLAG_STATE_OPTIONS = [
  { value: 'disabled', label: 'Off' },
  { value: 'enabled', label: 'On', selectedTone: 'accent' }
] satisfies readonly SegmentedControlOption[]

function isFeatureFlagState(value: string): value is 'disabled' | 'enabled' {
  return value === 'disabled' || value === 'enabled'
}

function JsonSyntaxPreview({ json }: { json: string }) {
  const tokens = useMemo<ReactNode[]>(() => {
    const parts: ReactNode[] = []
    const tokenPattern = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[{}[\],]/g
    let lastIndex = 0
    let tokenIndex = 0

    for (const match of json.matchAll(tokenPattern)) {
      if (match.index === undefined) continue
      if (match.index > lastIndex) parts.push(json.slice(lastIndex, match.index))

      const [token, stringToken, keySuffix, literalToken] = match
      const className = cn(
        stringToken && keySuffix && 'text-accent',
        stringToken && !keySuffix && 'text-good',
        literalToken && 'text-warn',
        !stringToken && !literalToken && /^[{}[\],]$/.test(token) && 'text-fg-faint',
        !stringToken && !literalToken && !/^[{}[\],]$/.test(token) && 'text-foreground'
      )

      parts.push(
        <span className={className} key={`${token}-${tokenIndex}`}>
          {token}
        </span>
      )
      tokenIndex += 1
      lastIndex = match.index + token.length
    }

    if (lastIndex < json.length) parts.push(json.slice(lastIndex))
    return parts
  }, [json])

  return <>{tokens}</>
}

function FeatureFlagRow({ flag }: { flag: FeatureFlagDefinition }) {
  const { getBaseValue, getEffectiveValue, isOverridden, resetOverride, setOverride } = useFeatureFlagSettings()
  const baseValue = getBaseValue(flag.path)
  const enabled = getEffectiveValue(flag.path)
  const overridden = isOverridden(flag.path)

  return (
    <SettingsRow label={flag.label} hint={flag.description}>
      <div className="grid grid-cols-[minmax(0,auto)_132px_28px] items-center justify-end gap-2">
        <span className="min-w-0 truncate rounded-full border border-border-soft bg-background px-2 py-0.5 font-mono text-[length:var(--density-type-annotation)] leading-none text-fg-dim">
          {overridden ? 'local override' : `default ${baseValue ? 'on' : 'off'}`}
        </span>
        <SegmentedControl
          ariaLabel={`${flag.label} state`}
          className="justify-self-end"
          itemClassName="min-w-[58px]"
          name={`feature-flag-${flag.sectionId}-${flag.key}`}
          onValueChange={(nextValue) => {
            if (isFeatureFlagState(nextValue)) setOverride(flag.path, nextValue === 'enabled')
          }}
          options={FEATURE_FLAG_STATE_OPTIONS}
          value={enabled ? 'enabled' : 'disabled'}
          variant="pill"
        />
        {overridden ? (
          <button
            aria-label={`Reset ${flag.label}`}
            className="ui-control grid size-7 place-items-center rounded-[var(--radius)] border p-0"
            onClick={() => resetOverride(flag.path)}
            type="button"
            title={`Reset ${flag.label}`}
          >
            <RotateCcw className="size-3" aria-hidden={true} />
          </button>
        ) : (
          <span aria-hidden={true} className="size-7" />
        )}
      </div>
    </SettingsRow>
  )
}

export function FeatureFlagsArea() {
  const { getEffectiveValue, overrides, resetAllOverrides, sections, storageKey } = useFeatureFlagSettings()
  const [activeSectionId, setActiveSectionId] = useState<FeatureFlagSectionId>(sections[0]?.id ?? 'global')
  const activeSection = sections.find((section) => section.id === activeSectionId) ?? sections[0]
  const activeOverrides = Object.values(overrides).reduce(
    (count, sectionOverrides) => count + Object.keys(sectionOverrides ?? {}).length,
    0
  )
  const previewJson = useMemo(() => JSON.stringify(overrides, null, 2), [overrides])

  if (!activeSection) return null

  return (
    <div className="grid gap-4 xl:grid-cols-[230px_minmax(0,1fr)]">
      <SidebarNavigation
        activeId={activeSection.id}
        ariaLabel="Feature flag groups"
        className="xl:sticky xl:top-[76px]"
        eyebrow="Flag groups"
        footer={
          <div className="space-y-2">
            <div>
              <span className="font-mono text-foreground">{activeOverrides}</span> local override
              {activeOverrides === 1 ? '' : 's'} stored.
            </div>
            <button
              className="ui-control inline-flex items-center gap-1.5 rounded-[var(--radius)] border px-2.5 py-1 text-[length:var(--density-type-caption)] font-medium"
              disabled={activeOverrides === 0}
              onClick={resetAllOverrides}
              type="button"
            >
              <RotateCcw className="size-3" aria-hidden={true} />
              Reset all
            </button>
          </div>
        }
        items={sections.map((section) => ({
          id: section.id,
          label: section.label,
          count: section.flags.length,
          icon:
            section.id === 'global' ? (
              <Globe2 aria-hidden={true} className="size-3.5" strokeWidth={1.7} />
            ) : (
              <Flag aria-hidden={true} className="size-3.5" strokeWidth={1.7} />
            )
        }))}
        onSelect={setActiveSectionId}
      />

      <div className="min-w-0 space-y-4">
        <SettingsSection
          id={`feature-flags-${activeSection.id}`}
          icon={
            activeSection.id === 'global' ? (
              <Globe2 className="size-4" aria-hidden={true} />
            ) : (
              <Flag className="size-4" aria-hidden={true} />
            )
          }
          title={`${activeSection.label} flags`}
          subtitle={activeSection.description}
        >
          {activeSection.flags.map((flag) => (
            <FeatureFlagRow flag={flag} key={flag.path} />
          ))}
        </SettingsSection>

        <section className="rounded-[var(--radius-lg)] border border-border bg-panel p-3.5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h3 className="text-[length:var(--density-type-control-lg)] font-semibold leading-tight text-foreground">
                Storage layer
              </h3>
              <p className="mt-1 text-[length:var(--density-type-caption)] leading-relaxed text-fg-faint">
                Only changed flags are written to localStorage. Defaults continue to come from the checked-in feature
                flag config.
              </p>
            </div>
            <span className="rounded-full border border-border-soft bg-background px-2 py-0.5 font-mono text-[length:var(--density-type-annotation)] text-fg-dim">
              {storageKey}
            </span>
          </div>
          <pre className="mt-3 max-h-56 overflow-auto rounded-[var(--radius)] border border-border-soft bg-background p-3 font-mono text-[length:var(--density-type-caption)] leading-relaxed text-fg-dim">
            <JsonSyntaxPreview json={previewJson} />
          </pre>
          <div className="mt-3 flex flex-wrap gap-2 text-[length:var(--density-type-caption)] text-fg-faint">
            {activeSection.flags.map((flag) => (
              <span
                key={flag.path}
                className="inline-flex rounded-full border border-border-soft bg-background px-2 py-0.5"
              >
                <span className="font-mono text-foreground">{flag.path}</span>
                <span className="px-1.5">is</span>
                <span className={getEffectiveValue(flag.path) ? 'text-good' : 'text-fg-faint'}>
                  {getEffectiveValue(flag.path) ? 'enabled' : 'disabled'}
                </span>
              </span>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
