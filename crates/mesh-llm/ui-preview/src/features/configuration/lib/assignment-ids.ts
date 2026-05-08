import type { ConfigAssign } from '@/features/app-tabs/types'

const ASSIGNMENT_ID_PREFIX = 'a-'

export function createAssignmentId(assigns: readonly ConfigAssign[]): string {
  const usedIds = new Set(assigns.map((assign) => assign.id))
  let nextIndex = assigns.length + 1

  while (usedIds.has(`${ASSIGNMENT_ID_PREFIX}${nextIndex}`)) {
    nextIndex += 1
  }

  return `${ASSIGNMENT_ID_PREFIX}${nextIndex}`
}
