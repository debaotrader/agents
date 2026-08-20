import { describe, expect, test } from "bun:test";
import {
  readAvailabilityConfig,
  renderAwayMessage,
} from "@/modules/availability/away";
import type { Schedule } from "@/modules/business-hours/hours";

// Decision table for the customer-facing out-of-hours copy (#153). Fixed instants, no real clock:
// 2024-01-07 is a Sunday, 2024-01-08 a Monday.
const MON_9_TO_17: Schedule = {
  windows: [{ day: 1, start: "09:00", end: "17:00" }],
  exceptions: [],
  timezone: "UTC",
};
const SUNDAY = new Date("2024-01-07T12:00:00Z");
// Monday 2024-01-08 09:00 UTC, rendered in each placeholder's own language.
const NEXT_PT = "segunda-feira, 08/01, 09:00";
const NEXT_EN = "Monday, 01/08, 09:00";

describe("readAvailabilityConfig", () => {
  test("absent/garbage block → empty copy, i.e. the pre-#153 silence", () => {
    expect(readAvailabilityConfig(undefined).awayMessage).toBe("");
    expect(readAvailabilityConfig({}).awayMessage).toBe("");
    expect(readAvailabilityConfig({ availability: 7 }).awayMessage).toBe("");
    expect(
      readAvailabilityConfig({ availability: { awayMessage: 42 } }).awayMessage,
    ).toBe("");
  });

  test("copy is trimmed, so whitespace never counts as configured", () => {
    expect(
      readAvailabilityConfig({ availability: { awayMessage: "  oi  " } })
        .awayMessage,
    ).toBe("oi");
    expect(
      readAvailabilityConfig({ availability: { awayMessage: "   " } })
        .awayMessage,
    ).toBe("");
  });
});

describe("renderAwayMessage", () => {
  test("no copy → nothing is sent (the feature is off by default)", () => {
    expect(
      renderAwayMessage({ copy: "", schedule: MON_9_TO_17, now: SUNDAY }),
    ).toEqual({ send: false, reason: "not_configured" });
    expect(
      renderAwayMessage({ copy: "   ", schedule: MON_9_TO_17, now: SUNDAY }),
    ).toEqual({ send: false, reason: "not_configured" });
  });

  test("copy with no placeholder goes out exactly as written", () => {
    expect(
      renderAwayMessage({
        copy: "Estamos fechados agora.",
        schedule: MON_9_TO_17,
        now: SUNDAY,
      }),
    ).toEqual({ send: true, text: "Estamos fechados agora." });
  });

  test("each placeholder renders the next opening in its own language", () => {
    expect(
      renderAwayMessage({
        copy: "Voltamos {proximo_atendimento}.",
        schedule: MON_9_TO_17,
        now: SUNDAY,
      }),
    ).toEqual({ send: true, text: `Voltamos ${NEXT_PT}.` });
    expect(
      renderAwayMessage({
        copy: "We are back {next_open}.",
        schedule: MON_9_TO_17,
        now: SUNDAY,
      }),
    ).toEqual({ send: true, text: `We are back ${NEXT_EN}.` });
  });

  test("every occurrence is replaced, and mixed copy stays in its own languages", () => {
    expect(
      renderAwayMessage({
        copy: "{proximo_atendimento} / {next_open} / {proximo_atendimento}",
        schedule: MON_9_TO_17,
        now: SUNDAY,
      }),
    ).toEqual({
      send: true,
      text: `${NEXT_PT} / ${NEXT_EN} / ${NEXT_PT}`,
    });
  });

  // The value comes from nextOpenAt, so it is exception-aware: with Monday taken by a holiday the
  // copy must promise Tuesday, not the weekly grid's Monday.
  test("the promised time skips a date exception", () => {
    const withHoliday: Schedule = {
      windows: [
        { day: 1, start: "09:00", end: "17:00" },
        { day: 2, start: "09:00", end: "17:00" },
      ],
      exceptions: [{ date: "2024-01-08", label: "Feriado", ranges: [] }],
      timezone: "UTC",
    };
    expect(
      renderAwayMessage({
        copy: "Voltamos {proximo_atendimento}.",
        schedule: withHoliday,
        now: SUNDAY,
      }),
    ).toEqual({ send: true, text: "Voltamos terça-feira, 09/01, 09:00." });
  });

  // Copy that promises a return time cannot be sent when there is no return time to promise: a
  // mutilated sentence and an invented one are both worse than the note the operator already gets.
  test("a schedule that never opens suppresses copy that promises a time", () => {
    const neverOpens: Schedule = {
      windows: [{ day: 1, start: "09:00", end: "17:00" }],
      exceptions: [
        {
          date: "2024-01-01",
          dateEnd: "2024-12-31",
          recurring: true,
          label: "Fechado indefinidamente",
          ranges: [],
        },
      ],
      timezone: "UTC",
    };
    expect(
      renderAwayMessage({
        copy: "Voltamos {proximo_atendimento}.",
        schedule: neverOpens,
        now: SUNDAY,
      }),
    ).toEqual({ send: false, reason: "no_next_open" });
  });

  test("copy that promises nothing still goes out on a schedule that never opens", () => {
    const neverOpens: Schedule = {
      windows: [{ day: 1, start: "09:00", end: "17:00" }],
      exceptions: [
        {
          date: "2024-01-01",
          dateEnd: "2024-12-31",
          recurring: true,
          ranges: [],
        },
      ],
      timezone: "UTC",
    };
    expect(
      renderAwayMessage({
        copy: "Estamos fechados.",
        schedule: neverOpens,
        now: SUNDAY,
      }),
    ).toEqual({ send: true, text: "Estamos fechados." });
  });
});
