import { Cloud, Server, type LucideProps } from 'lucide-react'
import * as RadioGroup from '@radix-ui/react-radio-group'
import type { Dispatch, SetStateAction } from 'react'
import { Badge } from '@/components/ui/badge'
import { TextField } from '@/components/ui/TextField'
import { ReserveActionDialog } from '@/features/reserves/components/ReserveActionDialog'
import { MESH_PROVIDERS, getMeshProvider } from '@/features/reserves/lib/mesh-providers'
import type { ProviderDraft } from '@/features/reserves/lib/provider-draft'
import type { MeshProvider } from '@/features/reserves/lib/reserve-types'
import { cn } from '@/lib/utils'
export type { ProviderDraft } from '@/features/reserves/lib/provider-draft'

type AddReserveProviderDialogProps = {
  confirmLabel: string
  description: string
  onConfirm: () => void
  onDraftChange: Dispatch<SetStateAction<ProviderDraft>>
  onOpenChange: (open: boolean) => void
  open: boolean
  providerDraft: ProviderDraft
}

type MeshProviderIconProps = LucideProps & {
  provider: MeshProvider
}

function MeshProviderIcon({ provider, ...props }: MeshProviderIconProps) {
  return provider.icon === 'server' ? <Server {...props} /> : <Cloud {...props} />
}

export function AddReserveProviderDialog({
  confirmLabel,
  description,
  onConfirm,
  onDraftChange,
  onOpenChange,
  open,
  providerDraft
}: AddReserveProviderDialogProps) {
  const selectedProvider = getMeshProvider(providerDraft.providerId)
  const supportedProviderCount = MESH_PROVIDERS.filter((provider) => provider.availability === 'supported').length

  function handleProviderSelection(providerId: string) {
    const nextProvider = getMeshProvider(providerId)
    if (nextProvider.availability !== 'supported') return

    onDraftChange({
      providerId: nextProvider.id,
      name: '',
      region: ''
    })
  }

  return (
    <ReserveActionDialog
      confirmLabel={confirmLabel}
      description={description}
      onConfirm={onConfirm}
      onOpenChange={onOpenChange}
      open={open}
      title="Add reserve provider"
    >
      <div className="space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="max-w-[56ch]">
            <div className="type-label text-fg-faint">Provider type</div>
            <div className="mt-1 text-[length:var(--density-type-control)] leading-[1.45] text-fg-dim">
              Choose the provider family to add. Disabled providers stay visible so the roadmap is clear.
            </div>
          </div>
          <Badge className="h-5 shrink-0 rounded-full px-2 text-[10px] uppercase tracking-[0.08em] text-fg-dim">
            {supportedProviderCount} supported
          </Badge>
        </div>

        <RadioGroup.Root
          aria-label="Reserve provider type"
          className="grid gap-2 sm:grid-cols-2"
          onValueChange={handleProviderSelection}
          value={providerDraft.providerId}
        >
          {MESH_PROVIDERS.map((provider) => {
            const disabled = provider.availability !== 'supported'

            return (
              <RadioGroup.Item
                className={cn(
                  'group relative flex min-h-[118px] items-start gap-3 overflow-hidden rounded-[var(--radius)] border border-border-soft bg-panel-strong px-3.5 py-3 text-left outline-none transition-[border-color,background,box-shadow,opacity] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
                  'hover:border-border hover:bg-panel data-[state=checked]:border-accent data-[state=checked]:bg-[color:color-mix(in_oklab,var(--color-accent)_9%,var(--color-panel-strong))] data-[state=checked]:shadow-[var(--shadow-focus-accent)]',
                  disabled &&
                    'cursor-not-allowed border-border/70 bg-panel opacity-65 hover:border-border/70 hover:bg-panel data-[state=checked]:border-border data-[state=checked]:bg-panel data-[state=checked]:shadow-none'
                )}
                disabled={disabled}
                key={provider.id}
                title={provider.disabledReason ?? provider.description}
                value={provider.id}
              >
                <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-[var(--radius-sm)] border border-border bg-background text-fg-dim transition-colors">
                  <MeshProviderIcon provider={provider} className="size-4" aria-hidden="true" />
                </span>
                <span className="min-w-0 flex-1 space-y-1.5">
                  <span className="flex min-w-0 flex-wrap items-center gap-1.5">
                    <span className="text-[length:var(--density-type-control-lg)] font-semibold text-foreground">
                      {provider.name}
                    </span>
                    <Badge className="h-[18px] rounded-full px-1.5 text-[9.5px] uppercase tracking-[0.08em] text-fg-dim">
                      {provider.availability === 'supported' ? 'Available' : 'Soon'}
                    </Badge>
                  </span>
                  <span className="block text-[length:var(--density-type-caption)] leading-[1.45] text-fg-dim">
                    {provider.description}
                  </span>
                  {provider.disabledReason ? (
                    <span className="block text-[length:var(--density-type-caption)] font-medium text-fg-faint">
                      {provider.disabledReason}
                    </span>
                  ) : null}
                </span>
              </RadioGroup.Item>
            )
          })}
        </RadioGroup.Root>
      </div>

      <div className="space-y-3 rounded-[var(--radius)] border border-border-soft bg-panel-strong px-3.5 py-3">
        <div className="flex items-start gap-3">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-[var(--radius-sm)] border border-accent/60 bg-[color:color-mix(in_oklab,var(--color-accent)_10%,var(--color-background))] text-accent">
            <MeshProviderIcon provider={selectedProvider} className="size-4" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <div className="type-label text-fg-faint">{selectedProvider.name} settings</div>
            <div className="mt-1 text-[length:var(--density-type-control)] leading-[1.45] text-fg-dim">
              Name the row and location shown in the reserve priority list. Empty fields use safe defaults.
            </div>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          {selectedProvider.optionFields.map((field) => {
            const inputId = `reserve-provider-${field.id}`

            return (
              <TextField
                helperText={field.helper}
                id={inputId}
                key={field.id}
                label={field.label}
                onChange={(event) => onDraftChange((current) => ({ ...current, [field.id]: event.target.value }))}
                placeholder={field.placeholder}
                type="text"
                value={providerDraft[field.id]}
              />
            )
          })}
        </div>
      </div>
    </ReserveActionDialog>
  )
}
