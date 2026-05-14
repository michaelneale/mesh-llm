import { useMemo, useState } from 'react'
import { Settings2, Zap } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { StatusPill } from '@/components/ui/status-pill'
import { ReserveActionDialog } from '@/features/reserves/components/ReserveActionDialog'
import { ReservePolicySettingsForm } from '@/features/reserves/components/ReservePolicySettingsForm'
import { ReservePolicyTile } from '@/features/reserves/components/ReservePolicyTile'
import {
  DEFAULT_RESERVE_WAKE_POLICY_SETTINGS,
  reservePolicyTilesFromSettings
} from '@/features/reserves/lib/reserve-policy'
import type { ReserveWakePolicySettings } from '@/features/reserves/lib/reserve-types'
import { cn } from '@/lib/utils'

type ReservePolicyPanelProps = {
  className?: string
  defaultSettings?: ReserveWakePolicySettings
  mode?: 'reserves' | 'configuration'
  onOpenConfigurationTab?: () => void
  onSettingsChange?: (settings: ReserveWakePolicySettings) => void
  providers?: string[]
  settings?: ReserveWakePolicySettings
}

type PolicyDialogMode = 'policy' | 'autowake' | null

export function ReservePolicyPanel({
  className,
  defaultSettings = DEFAULT_RESERVE_WAKE_POLICY_SETTINGS,
  mode = 'reserves',
  onOpenConfigurationTab,
  onSettingsChange,
  providers,
  settings
}: ReservePolicyPanelProps) {
  const [internalSettings, setInternalSettings] = useState(defaultSettings)
  const [draftSettings, setDraftSettings] = useState(defaultSettings)
  const [dialogMode, setDialogMode] = useState<PolicyDialogMode>(null)
  const effectiveSettings = settings ?? internalSettings
  const tiles = useMemo(() => reservePolicyTilesFromSettings(effectiveSettings), [effectiveSettings])
  const autoWakeStatusLabel = effectiveSettings.autoWakeEnabled ? 'Auto-wake on' : 'Auto-wake paused'
  const isConfigurationMode = mode === 'configuration'

  function openDialog(nextMode: Exclude<PolicyDialogMode, null>) {
    setDraftSettings(effectiveSettings)
    setDialogMode(nextMode)
  }

  function saveDraftSettings() {
    if (!settings) setInternalSettings(draftSettings)
    onSettingsChange?.(draftSettings)
  }

  const description =
    'Preview reserve wake thresholds, provider order, and idle sleep rules before the backend configuration fields are wired in.'

  return (
    <>
      <Card
        className={cn('overflow-hidden rounded-[10px] bg-panel shadow-none', className)}
        data-testid="reserve-policy-panel"
      >
        <CardHeader
          className={cn(
            'border-b border-border/70 px-[14px] py-[10px]',
            isConfigurationMode ? 'space-y-2' : 'space-y-0'
          )}
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <CardTitle className="text-[12px] font-semibold leading-none tracking-[0.0125em] text-foreground">
                  Reserve policy
                </CardTitle>
                {isConfigurationMode ? (
                  <StatusPill
                    dot
                    label={autoWakeStatusLabel}
                    tone={effectiveSettings.autoWakeEnabled ? 'good' : 'neutral'}
                  />
                ) : null}
              </div>
              {isConfigurationMode ? <p className="type-caption mt-1 max-w-[68ch] text-fg-dim">{description}</p> : null}
            </div>
            {isConfigurationMode ? (
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  className="ui-control h-8 rounded-[var(--radius)] border px-3 text-[length:var(--density-type-control)]"
                  onClick={onOpenConfigurationTab ?? (() => openDialog('policy'))}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  <Settings2 aria-hidden="true" className="mr-1 size-3.5" />
                  Edit policy
                </Button>
                <Button
                  className="ui-control-primary h-8 rounded-[var(--radius)] px-3 text-[length:var(--density-type-control)]"
                  onClick={() => openDialog('autowake')}
                  size="sm"
                  type="button"
                  variant="default"
                >
                  <Zap aria-hidden="true" className="mr-1 size-3.5" />
                  Configure auto-wake
                </Button>
              </div>
            ) : (
              <button
                className="ui-link text-[11.5px] leading-none"
                onClick={onOpenConfigurationTab ?? (() => openDialog('policy'))}
                type="button"
              >
                Edit policy →
              </button>
            )}
          </div>
        </CardHeader>
        <CardContent className="p-[14px]">
          <div className={isConfigurationMode ? 'grid gap-[14px] lg:grid-cols-3' : 'grid gap-[14px] md:grid-cols-3'}>
            {tiles.map((tile) => (
              <ReservePolicyTile key={tile.title} tile={tile} />
            ))}
          </div>
          {isConfigurationMode ? (
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-border/70 pt-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge className="rounded-full px-2 py-0.5 text-[10px] uppercase tracking-[0.08em] text-fg-dim">
                  Preview only
                </Badge>
                <span className="type-caption text-fg-dim">
                  These controls shape the high-fidelity UI now, and backend persistence can attach later without
                  changing the interaction model.
                </span>
              </div>
              {onOpenConfigurationTab ? (
                <button
                  className="ui-link text-[length:var(--density-type-caption-lg)]"
                  onClick={onOpenConfigurationTab}
                  type="button"
                >
                  Open Reserves tab
                </button>
              ) : null}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <ReserveActionDialog
        confirmLabel={dialogMode === 'autowake' ? 'Apply auto-wake' : 'Save policy'}
        description={
          dialogMode === 'autowake'
            ? 'Tune the reserve wake threshold and warm-up window. This preview updates the visible cards only and does not send backend requests.'
            : 'Adjust how reserve providers are prioritized and when they go back to sleep. This preview edits local UI state only.'
        }
        onConfirm={saveDraftSettings}
        onOpenChange={(open) => {
          if (!open) setDialogMode(null)
        }}
        open={dialogMode !== null}
        title={dialogMode === 'autowake' ? 'Configure auto-wake' : 'Edit reserve policy'}
      >
        <ReservePolicySettingsForm
          onSettingsChange={setDraftSettings}
          providers={dialogMode === 'policy' ? providers : undefined}
          settings={draftSettings}
        />
      </ReserveActionDialog>
    </>
  )
}
