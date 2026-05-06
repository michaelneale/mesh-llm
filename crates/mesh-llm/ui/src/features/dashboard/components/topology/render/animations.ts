import { hashString } from "../helpers";
import type { EntryAnimation } from "../types";

export function easeOutCubic(value: number) {
  return 1 - Math.pow(1 - value, 3);
}

export function cubicBezier(
  start: number,
  control1: number,
  control2: number,
  end: number,
  t: number,
) {
  const inv = 1 - t;
  return (
    inv * inv * inv * start +
    3 * inv * inv * t * control1 +
    3 * inv * t * t * control2 +
    t * t * t * end
  );
}

export function createTravelAnimation(
  nodeId: string,
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  startedAt: number,
  randomSource = `${nodeId}:entry`,
  hideLinksUntilSettled = false,
  isReposition = false,
): EntryAnimation {
  const horizontalSpan = toX - fromX;
  const verticalSpan = toY - fromY;
  const distance = Math.max(1, Math.hypot(horizontalSpan, verticalSpan));
  const normalX = -verticalSpan / distance;
  const normalY = horizontalSpan / distance;
  const bendDirection = hashString(`${randomSource}:bend`) > 0.5 ? 1 : -1;
  const bendAmount =
    Math.min(distance * 0.18, 82) *
    (0.6 + hashString(`${randomSource}:arc`) * 0.9) *
    bendDirection;
  const lateralJitter = (hashString(`${randomSource}:lateral`) - 0.5) * 42;

  return {
    fromX,
    fromY,
    control1X:
      fromX + horizontalSpan * 0.24 + normalX * bendAmount + lateralJitter,
    control1Y: fromY + verticalSpan * 0.18 + normalY * bendAmount,
    control2X:
      fromX +
      horizontalSpan * 0.72 -
      normalX * bendAmount * 0.56 -
      lateralJitter * 0.35,
    control2Y: fromY + verticalSpan * 0.82 - normalY * bendAmount * 0.56,
    toX,
    toY,
    normalX,
    normalY,
    meanderAmplitude:
      Math.min(distance * 0.06, 28) *
      (0.6 + hashString(`${randomSource}:meander`) * 0.8),
    meanderCycles: 1.15 + hashString(`${randomSource}:cycles`) * 1.35,
    meanderPhase: hashString(`${randomSource}:phase`) * Math.PI,
    startedAt,
    hideLinksUntilSettled,
    isReposition,
  };
}
