import type { ModelSplit } from '../types/config';

export function clampResizeBoundaryStart(left: ModelSplit, right: ModelSplit, boundaryStart: number): number {
  const minBoundaryStart = left.start + 2;
  const maxBoundaryStart = right.end - 2;

  return Math.max(minBoundaryStart, Math.min(maxBoundaryStart, boundaryStart));
}
