import { describe, expect, it } from "vitest";
import { extractObjectLabels } from "./image-describe";

describe("extractObjectLabels", () => {
  it("returns [] for empty input", () => {
    expect(extractObjectLabels("", "")).toEqual([]);
    expect(extractObjectLabels("   ", "caption")).toEqual([]);
  });

  it("splits on newlines, commas, and semicolons", () => {
    const raw = "red car\nblue bicycle, pedestrian; traffic light";
    expect(extractObjectLabels(raw, "")).toEqual([
      "red car",
      "blue bicycle",
      "pedestrian",
      "traffic light",
    ]);
  });

  it("treats florence bbox markers as label separators", () => {
    const raw = "golden retriever<loc_12><loc_34><loc_56><loc_78>blue collar<loc_1><loc_2><loc_3><loc_4>";
    expect(extractObjectLabels(raw, "")).toEqual([
      "golden retriever",
      "blue collar",
    ]);
  });

  it("dedupes case-insensitively", () => {
    const raw = "Cat\ncat\nCAT\ndog";
    expect(extractObjectLabels(raw, "")).toEqual(["Cat", "dog"]);
  });

  it("drops labels already present in the caption", () => {
    const caption = "A golden retriever sits on a green lawn.";
    const raw = "golden retriever, blue collar, red brick wall";
    expect(extractObjectLabels(raw, caption)).toEqual([
      "blue collar",
      "red brick wall",
    ]);
  });

  it("caps the list at 12 labels", () => {
    const raw = Array.from({ length: 20 }, (_, i) => `item ${i}`).join(", ");
    expect(extractObjectLabels(raw, "")).toHaveLength(12);
  });

  it("returns [] when everything is in the caption", () => {
    const caption = "red car driving past a blue bicycle";
    const raw = "red car; blue bicycle";
    expect(extractObjectLabels(raw, caption)).toEqual([]);
  });
});
