import { describe, it, expect } from "vitest";
import { formatShortDuration } from "./format-duration";

const S = 1;
const M = 60;
const H = 3_600;
const D = 86_400;
const W = 7 * D;
const MO = 30 * D;
const Y = 365 * D;

describe("formatShortDuration", () => {
  it.each([
    [null, "-"],
    [undefined, "-"],
    [0, "-"],
    [-1, "-"],
    [NaN, "-"],
    [Infinity, "-"],
    [-Infinity, "-"],
  ])("returns dash for %s", (input, expected) => {
    expect(formatShortDuration(input as number | null | undefined)).toBe(expected);
  });

  it.each([
    [1, "1s"],
    [20, "20s"],
    [59, "59s"],
  ])("formats %i seconds as seconds", (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected);
  });

  it.each([
    [60, "1m"],
    [120, "2m"],
    [3599, "59m"],
  ])("formats %i seconds as minutes", (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected);
  });

  it.each([
    [H, "1h"],
    [2 * H, "2h"],
    [23 * H + 59 * M + 59 * S, "23h"],
  ])("formats %i seconds as hours", (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected);
  });

  it.each([
    [D, "1d"],
    [2 * D, "2d"],
    [6 * D, "6d"],
  ])("formats %i seconds as days", (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected);
  });

  it.each([
    [W, "1w"],
    [2 * W, "2w"],
    [3 * W, "3w"],
  ])("formats %i seconds as exact weeks", (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected);
  });

  it.each([
    [9 * D, "1w 2d"],
    [12 * D, "1w 5d"],
    [20 * D, "2w 6d"],
    [27 * D, "3w 6d"],
  ])("formats %i seconds as compound weeks+days", (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected);
  });

  it.each([
    [MO, "1mo"],
    [2 * MO, "2mo"],
    [11 * MO, "11mo"],
  ])("formats %i seconds as exact months", (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected);
  });

  it.each([
    [MO + 10 * D, "1mo 10d"],
    [2 * MO + 15 * D, "2mo 15d"],
    [3 * MO + D, "3mo 1d"],
  ])("formats %i seconds as compound months+days", (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected);
  });

  it.each([
    [Y, "1y"],
    [2 * Y, "2y"],
  ])("formats %i seconds as exact years", (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected);
  });

  it.each([
    [Y + 2 * MO, "1y 2mo"],
    [2 * Y + 6 * MO, "2y 6mo"],
  ])("formats %i seconds as compound years+months", (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected);
  });

  it.each([
    [Y + 2 * MO + 2 * D, "1y 2mo 2d"],
    [Y + MO + 15 * D, "1y 1mo 15d"],
    [2 * Y + 3 * MO + 7 * D, "2y 3mo 7d"],
  ])("formats %i seconds as compound years+months+days", (input, expected) => {
    expect(formatShortDuration(input)).toBe(expected);
  });

  it("floors fractional seconds", () => {
    expect(formatShortDuration(59.9)).toBe("59s");
    expect(formatShortDuration(3601.5)).toBe("1h");
  });
});
