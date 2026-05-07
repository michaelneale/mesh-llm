import type { ColorTuple } from "../types";

export type LineMesh = {
  positions: Float32Array;
  colors: Float32Array;
  lineCoords: Float32Array;
};

type BuildLineMeshInput = {
  positions: Float32Array;
  colors: Float32Array;
  lineWidthPx: number;
  devicePixelRatio: number;
};

type LineSegment = {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  startColor: ColorTuple;
  endColor: ColorTuple;
  length: number;
};

function nearlyEqual(left: number, right: number, epsilon = 0.001) {
  return Math.abs(left - right) <= epsilon;
}

function colorsMatch(left: ColorTuple, right: ColorTuple) {
  return left.every((channel, index) => nearlyEqual(channel, right[index] ?? channel));
}

function shouldContinueChain(previous: LineSegment, next: LineSegment) {
  return (
    nearlyEqual(previous.endX, next.startX) &&
    nearlyEqual(previous.endY, next.startY) &&
    colorsMatch(previous.endColor, next.startColor)
  );
}

function pushQuadVertex(
  meshPositions: number[],
  meshColors: number[],
  meshLineCoords: number[],
  x: number,
  y: number,
  color: ColorTuple,
  along: number,
  side: number,
) {
  meshPositions.push(x, y);
  meshColors.push(color[0], color[1], color[2], color[3]);
  meshLineCoords.push(along, side);
}

export function buildLineMesh({
  positions,
  colors,
  lineWidthPx,
  devicePixelRatio,
}: BuildLineMeshInput): LineMesh {
  const segmentCount = Math.min(Math.floor(positions.length / 4), Math.floor(colors.length / 8));
  if (segmentCount === 0) {
    return {
      positions: new Float32Array(0),
      colors: new Float32Array(0),
      lineCoords: new Float32Array(0),
    };
  }

  const meshPositions: number[] = [];
  const meshColors: number[] = [];
  const meshLineCoords: number[] = [];
  const halfWidth = Math.max(0.5, (lineWidthPx * devicePixelRatio) / 2);
  const segments: LineSegment[] = [];

  for (let segmentIndex = 0; segmentIndex < segmentCount; segmentIndex += 1) {
    const positionOffset = segmentIndex * 4;
    const colorOffset = segmentIndex * 8;
    const startX = positions[positionOffset];
    const startY = positions[positionOffset + 1];
    const endX = positions[positionOffset + 2];
    const endY = positions[positionOffset + 3];
    const dx = endX - startX;
    const dy = endY - startY;
    const length = Math.hypot(dx, dy);
    if (!Number.isFinite(length) || length <= 0.001) {
      continue;
    }

    const startColor: ColorTuple = [
      colors[colorOffset],
      colors[colorOffset + 1],
      colors[colorOffset + 2],
      colors[colorOffset + 3],
    ];
    const endColor: ColorTuple = [
      colors[colorOffset + 4],
      colors[colorOffset + 5],
      colors[colorOffset + 6],
      colors[colorOffset + 7],
    ];

    segments.push({
      startX,
      startY,
      endX,
      endY,
      startColor,
      endColor,
      length,
    });
  }

  for (let segmentIndex = 0; segmentIndex < segments.length; ) {
    const chainStart = segmentIndex;
    let chainLength = segments[segmentIndex].length;
    while (
      segmentIndex + 1 < segments.length &&
      shouldContinueChain(segments[segmentIndex], segments[segmentIndex + 1])
    ) {
      segmentIndex += 1;
      chainLength += segments[segmentIndex].length;
    }

    const chainEnd = segmentIndex;
    let traversedLength = 0;

    for (let chainSegmentIndex = chainStart; chainSegmentIndex <= chainEnd; chainSegmentIndex += 1) {
      const segment = segments[chainSegmentIndex];
      const normalX = -(segment.endY - segment.startY) / segment.length;
      const normalY = (segment.endX - segment.startX) / segment.length;
      const offsetX = normalX * halfWidth;
      const offsetY = normalY * halfWidth;
      const startAlong = traversedLength / chainLength;
      const endAlong = (traversedLength + segment.length) / chainLength;

      const startLeftX = segment.startX - offsetX;
      const startLeftY = segment.startY - offsetY;
      const startRightX = segment.startX + offsetX;
      const startRightY = segment.startY + offsetY;
      const endLeftX = segment.endX - offsetX;
      const endLeftY = segment.endY - offsetY;
      const endRightX = segment.endX + offsetX;
      const endRightY = segment.endY + offsetY;

      pushQuadVertex(
        meshPositions,
        meshColors,
        meshLineCoords,
        startLeftX,
        startLeftY,
        segment.startColor,
        startAlong,
        -1,
      );
      pushQuadVertex(
        meshPositions,
        meshColors,
        meshLineCoords,
        startRightX,
        startRightY,
        segment.startColor,
        startAlong,
        1,
      );
      pushQuadVertex(
        meshPositions,
        meshColors,
        meshLineCoords,
        endLeftX,
        endLeftY,
        segment.endColor,
        endAlong,
        -1,
      );
      pushQuadVertex(
        meshPositions,
        meshColors,
        meshLineCoords,
        endLeftX,
        endLeftY,
        segment.endColor,
        endAlong,
        -1,
      );
      pushQuadVertex(
        meshPositions,
        meshColors,
        meshLineCoords,
        startRightX,
        startRightY,
        segment.startColor,
        startAlong,
        1,
      );
      pushQuadVertex(
        meshPositions,
        meshColors,
        meshLineCoords,
        endRightX,
        endRightY,
        segment.endColor,
        endAlong,
        1,
      );

      traversedLength += segment.length;
    }

    segmentIndex += 1;
  }

  return {
    positions: new Float32Array(meshPositions),
    colors: new Float32Array(meshColors),
    lineCoords: new Float32Array(meshLineCoords),
  };
}
