import { describe, expect, test } from "bun:test";
import {
  claimAttendanceBoundary,
  crossesAttendanceBoundary,
  needsAttendanceStartProbe,
} from "@/graph/attendance-boundary";

// Decision table for the boundary claim shared by the three writers of a contact's memory thread
// (runtime, ingest, nudge). Every row is a state one of them can actually be in; the columns are the
// three things it then has to do.
describe("claimAttendanceBoundary", () => {
  const CURRENT = 77;
  const PREVIOUS = 42;

  const cases: {
    name: string;
    previous: number | null;
    inFlight: boolean;
    started: boolean;
    divider: boolean;
    advance: boolean;
    closed: number | null;
  }[] = [
    {
      // First activity ever on this thread. The row must be created even though nothing ended: without
      // it, resolve-time compaction has no marker to read and exits before doing anything.
      name: "fresh thread, no marker yet",
      previous: null,
      inFlight: false,
      started: false,
      divider: false,
      advance: true,
      closed: null,
    },
    {
      name: "fresh thread is unaffected by an invoke in flight",
      previous: null,
      inFlight: true,
      started: false,
      divider: false,
      advance: true,
      closed: null,
    },
    {
      name: "same attendance, marker already there",
      previous: CURRENT,
      inFlight: false,
      started: true,
      divider: false,
      advance: false,
      closed: null,
    },
    {
      // The plain boundary: previous attendance ended, nothing in the way, this attendance has not
      // put a message in the thread yet.
      name: "boundary consumed",
      previous: PREVIOUS,
      inFlight: false,
      started: false,
      divider: true,
      advance: true,
      closed: PREVIOUS,
    },
    {
      // Case 1: an invoke would erase the divider while the marker advanced for good. Defer both,
      // arm compaction anyway — the attendance that ended is compactable from its stamps alone.
      name: "boundary deferred while another invoke reads the thread",
      previous: PREVIOUS,
      inFlight: true,
      started: false,
      divider: false,
      advance: false,
      closed: PREVIOUS,
    },
    {
      // Case 2: the deferral above left the marker behind, so the boundary is still visible — but by
      // now this attendance has messages in the thread and an appended divider would land after them.
      name: "boundary consumed without a divider once the attendance started",
      previous: PREVIOUS,
      inFlight: false,
      started: true,
      divider: false,
      advance: true,
      closed: PREVIOUS,
    },
  ];

  for (const c of cases) {
    test(c.name, () => {
      expect(
        claimAttendanceBoundary({
          previousConversationId: c.previous,
          conversationId: CURRENT,
          anotherInvokeIsReading: c.inFlight,
          attendanceAlreadyStarted: c.started,
        }),
      ).toEqual({
        writeDivider: c.divider,
        advanceMarker: c.advance,
        closedConversationId: c.closed,
      });
    });
  }

  // Every case that ends an attendance must arm compaction for it, including the two that write
  // nothing to the thread. Compaction that is not armed here waits on a writer that may never come.
  test("every boundary row reports the closed attendance", () => {
    const boundaryRows = cases.filter(
      (c) => c.previous !== null && c.previous !== CURRENT,
    );
    expect(boundaryRows.length).toBe(3);
    for (const c of boundaryRows) expect(c.closed).toBe(PREVIOUS);
  });
});

describe("crossesAttendanceBoundary", () => {
  test("only a different, already-recorded conversation crosses", () => {
    expect(crossesAttendanceBoundary(null, 7)).toBe(false);
    expect(crossesAttendanceBoundary(7, 7)).toBe(false);
    expect(crossesAttendanceBoundary(6, 7)).toBe(true);
  });
});

describe("needsAttendanceStartProbe", () => {
  // The probe is a checkpointer read, and it only changes the answer on a boundary that is actually
  // being consumed. Asking in the other states would buy a round-trip per message written.
  test("only on a boundary with no invoke in the way", () => {
    expect(needsAttendanceStartProbe(6, 7, false)).toBe(true);
    expect(needsAttendanceStartProbe(6, 7, true)).toBe(false);
    expect(needsAttendanceStartProbe(7, 7, false)).toBe(false);
    expect(needsAttendanceStartProbe(null, 7, false)).toBe(false);
  });
});
