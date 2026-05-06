import { afterEach, describe, expect, it } from "vitest";

import {
  normalizeSection,
  pathnameForRoute,
  readRouteFromLocation,
  sectionFromPathname,
} from "./routes";

function setPath(path: string) {
  window.history.replaceState({}, "", path);
}

afterEach(() => {
  setPath("/");
});

describe("routes playground gating", () => {
  it("treats /playground as a first-class route in dev", () => {
    setPath("/playground");

    expect(sectionFromPathname("/playground", true)).toBe("playground");
    expect(readRouteFromLocation(true)).toEqual({ section: "playground", chatId: null });
    expect(pathnameForRoute({ section: "playground", chatId: null }, true)).toBe("/playground");
  });

  it("normalizes /playground to dashboard in production", () => {
    setPath("/playground");

    expect(normalizeSection("playground", false)).toBe("dashboard");
    expect(sectionFromPathname("/playground", false)).toBeNull();
    expect(readRouteFromLocation(false)).toEqual({ section: "dashboard", chatId: null });
    expect(pathnameForRoute({ section: "playground", chatId: null }, false)).toBe("/dashboard");
  });
});
