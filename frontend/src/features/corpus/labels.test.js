import { afterEach, describe, expect, it, vi } from "vitest";

import { consultation, eventLabel } from "./labels";

describe("consultation", () => {
  afterEach(() => vi.useRealTimers());

  it("is open through the whole closing day", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 4, 15, 23, 30));
    expect(consultation({ comment_end: "2026-05-15" })).toEqual({
      open: true,
      label: expect.stringMatching(/^Comentarios hasta 15/),
    });
  });

  it("is closed the day after and absent without a deadline", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 4, 16, 0, 5));
    expect(consultation({ comment_end: "2026-05-15" }).open).toBe(false);
    expect(consultation({ comment_end: null })).toBeNull();
  });
});

describe("eventLabel", () => {
  it("calls new drafts projects", () => {
    expect(eventLabel({ kind: "norma_nueva", norm: { status: "proyecto" } })).toBe("Nuevo proyecto");
    expect(eventLabel({ kind: "norma_nueva", norm: { status: "vigente" } })).toBe("Nueva");
  });
});
