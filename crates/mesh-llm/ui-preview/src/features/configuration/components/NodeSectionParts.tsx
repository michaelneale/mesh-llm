import type { Dispatch, SetStateAction } from 'react'
import { ModelConfigCard } from '@/features/configuration/components/ModelConfigCard'
import { ReservedConfigCard } from '@/features/configuration/components/ReservedConfigCard'
import { reservedVramSelectionId } from '@/features/configuration/lib/selection'
import type { ConfigAssign, ConfigModel, ConfigNode } from '@/features/app-tabs/types'

type SelectedModelConfigProps = {
  containerIdx: number
  readOnly: boolean
  selectedAssign: ConfigAssign | undefined
  selectedAssignContainerIdx: number
  node: ConfigNode
  models?: ConfigModel[]
  containerFreeGB: number
  setAssigns: Dispatch<SetStateAction<ConfigAssign[]>>
  onPick: (id: string | null) => void
}

export function SelectedModelConfig({
  containerIdx,
  readOnly,
  selectedAssign,
  selectedAssignContainerIdx,
  node,
  models,
  containerFreeGB,
  setAssigns,
  onPick
}: SelectedModelConfigProps) {
  if (readOnly || !selectedAssign || selectedAssignContainerIdx !== containerIdx) return null

  return (
    <ModelConfigCard
      key={selectedAssign.id}
      assign={selectedAssign}
      node={node}
      models={models}
      containerFreeGB={containerFreeGB}
      controlTabIndex={-1}
      onCtxChange={(ctx) =>
        setAssigns((items) => items.map((assign) => (assign.id === selectedAssign.id ? { ...assign, ctx } : assign)))
      }
      onRemove={() => {
        setAssigns((items) => items.filter((assign) => assign.id !== selectedAssign.id))
        onPick(null)
      }}
    />
  )
}

type SelectedReservedConfigProps = {
  selectedId?: string | null
  node: ConfigNode
  containerIdx: number
  locationLabel: string
  reservedGB: number
}

export function SelectedReservedConfig({
  selectedId,
  node,
  containerIdx,
  locationLabel,
  reservedGB
}: SelectedReservedConfigProps) {
  if (reservedGB <= 0 || selectedId !== reservedVramSelectionId(node.id, containerIdx)) return null

  return (
    <ReservedConfigCard
      key={`reserved-${node.id}-${containerIdx}`}
      locationLabel={locationLabel}
      reservedGB={reservedGB}
    />
  )
}
