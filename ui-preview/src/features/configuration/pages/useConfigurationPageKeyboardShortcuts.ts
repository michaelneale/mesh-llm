import { useEffect } from 'react'
import type { Placement } from '@/features/app-tabs/types'
import { isTextEditingTarget } from '@/features/configuration/pages/ConfigurationPage.helpers'

type UseConfigurationPageKeyboardShortcutsParams = {
  canUndo: boolean
  canRedo: boolean
  selectedAssignId: string | null
  saveConfiguration: () => void
  revertConfiguration: () => void
  undoConfigurationChange: () => void
  redoConfigurationChange: () => void
  selectNodeByOffset: (direction: -1 | 1) => void
  selectGpuSlotByOffset: (direction: -1 | 1) => void
  selectModelInCurrentGpu: (direction: -1 | 1, jumpToBoundary?: boolean) => void
  moveSelectedAssignByGpuOffset: (direction: -1 | 1) => void
  stepSelectedContext: (direction: -1 | 1, jumpToPower?: boolean) => void
  openCatalogForCurrentNode: () => void
  setCurrentNodePlacement: (placement: Placement) => boolean
  removeSelectedAssign: (assignId: string) => void
}

export function useConfigurationPageKeyboardShortcuts({
  canUndo,
  canRedo,
  selectedAssignId,
  saveConfiguration,
  revertConfiguration,
  undoConfigurationChange,
  redoConfigurationChange,
  selectNodeByOffset,
  selectGpuSlotByOffset,
  selectModelInCurrentGpu,
  moveSelectedAssignByGpuOffset,
  stepSelectedContext,
  openCatalogForCurrentNode,
  setCurrentNodePlacement,
  removeSelectedAssign
}: UseConfigurationPageKeyboardShortcutsParams) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || isTextEditingTarget(event.target)) return

      const dialogTarget = event.target instanceof Element ? event.target.closest('[role="dialog"]') : null
      if (dialogTarget) return

      if (event.key.toLowerCase() === 's' && event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey) {
        event.preventDefault()
        saveConfiguration()
      } else if (
        event.key.toLowerCase() === 'x' &&
        event.ctrlKey &&
        !event.metaKey &&
        !event.shiftKey &&
        !event.altKey
      ) {
        event.preventDefault()
        revertConfiguration()
      } else if (
        event.key.toLowerCase() === 'z' &&
        event.ctrlKey &&
        !event.metaKey &&
        !event.shiftKey &&
        !event.altKey
      ) {
        event.preventDefault()
        if (canUndo) undoConfigurationChange()
      } else if (
        event.key.toLowerCase() === 'r' &&
        event.ctrlKey &&
        !event.metaKey &&
        !event.shiftKey &&
        !event.altKey
      ) {
        event.preventDefault()
        if (canRedo) redoConfigurationChange()
      } else if (event.key === 'Tab' && !event.metaKey && !event.ctrlKey && !event.altKey) {
        event.preventDefault()
        selectNodeByOffset(event.shiftKey ? -1 : 1)
      } else if (event.key === 'ArrowDown' && event.shiftKey && !event.metaKey && !event.ctrlKey && !event.altKey) {
        event.preventDefault()
        moveSelectedAssignByGpuOffset(1)
      } else if (event.key === 'ArrowDown' && !event.metaKey && !event.ctrlKey && !event.altKey) {
        event.preventDefault()
        selectGpuSlotByOffset(1)
      } else if (event.key === 'ArrowUp' && event.shiftKey && !event.metaKey && !event.ctrlKey && !event.altKey) {
        event.preventDefault()
        moveSelectedAssignByGpuOffset(-1)
      } else if (event.key === 'ArrowUp' && !event.metaKey && !event.ctrlKey && !event.altKey) {
        event.preventDefault()
        selectGpuSlotByOffset(-1)
      } else if (event.key === 'ArrowRight' && !event.metaKey && !event.ctrlKey) {
        event.preventDefault()
        if (event.altKey) stepSelectedContext(1, event.shiftKey)
        else selectModelInCurrentGpu(1, event.shiftKey)
      } else if (event.key === 'ArrowLeft' && !event.metaKey && !event.ctrlKey) {
        event.preventDefault()
        if (event.altKey) stepSelectedContext(-1, event.shiftKey)
        else selectModelInCurrentGpu(-1, event.shiftKey)
      } else if (event.key.toLowerCase() === 'a' && !event.metaKey && !event.ctrlKey && !event.altKey) {
        event.preventDefault()
        openCatalogForCurrentNode()
      } else if (
        (event.key.toLowerCase() === 'p' || event.key.toLowerCase() === 's') &&
        !event.metaKey &&
        !event.ctrlKey &&
        !event.altKey
      ) {
        const placement: Placement = event.key.toLowerCase() === 'p' ? 'pooled' : 'separate'

        if (setCurrentNodePlacement(placement)) event.preventDefault()
      } else if (event.key === 'Delete' && selectedAssignId) {
        event.preventDefault()
        removeSelectedAssign(selectedAssignId)
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [
    canRedo,
    canUndo,
    moveSelectedAssignByGpuOffset,
    openCatalogForCurrentNode,
    redoConfigurationChange,
    removeSelectedAssign,
    revertConfiguration,
    saveConfiguration,
    selectGpuSlotByOffset,
    selectModelInCurrentGpu,
    selectNodeByOffset,
    selectedAssignId,
    setCurrentNodePlacement,
    stepSelectedContext,
    undoConfigurationChange
  ])
}
