import { clipText, TEMPLATE_MESSAGE_MAX } from "@/modules/agents/text-caps";
import { nextOpenAt, type Schedule } from "@/modules/business-hours/hours";

// The customer-facing side of the availability gate. The gate silences the agent outside its schedule
// and tells the OPERATOR with a private note; until this module existed the CUSTOMER was told nothing,
// so from their side the business simply did not answer (issue #153). `awayMessage` is operator-authored
// copy sent as the persona bot from the same branch that posts the note: no model call, no tokens.
//
// Empty copy = the pre-#153 behavior (silence), so the block is additive for every existing agent.

export interface AvailabilityConfig {
  // What the customer receives while the agent is outside its schedule. Empty = send nothing.
  awayMessage: string;
}

export const AVAILABILITY_DEFAULTS: AvailabilityConfig = {
  awayMessage: "",
};

// The placeholder is the opt-in for interpolating the next opening: an operator who wants it writes it,
// which is why there is no companion boolean. The two spellings also pick the language of the rendered
// value — copy written in Portuguese asks for it in Portuguese — so the feature needs no locale setting
// of its own, and mixed-language output is impossible.
const NEXT_OPEN_PLACEHOLDERS: ReadonlyArray<{ token: string; locale: string }> =
  [
    { token: "{proximo_atendimento}", locale: "pt-BR" },
    { token: "{next_open}", locale: "en-US" },
  ];

export function readAvailabilityConfig(settings: unknown): AvailabilityConfig {
  const bag =
    settings && typeof settings === "object"
      ? (settings as Record<string, unknown>).availability
      : undefined;
  if (!bag || typeof bag !== "object") return { ...AVAILABILITY_DEFAULTS };
  const raw = (bag as Record<string, unknown>).awayMessage;
  // Clamped like every other operator free-text field: the write boundary refuses copy that is too
  // long, and this is the defense for a row that got one anyway (import, hand-edit, older write).
  return {
    awayMessage:
      typeof raw === "string" ? clipText(raw.trim(), TEMPLATE_MESSAGE_MAX) : "",
  };
}

// Weekday AND date, in both languages. A bare weekday reads fine for "closed for the night" and is
// ambiguous for exactly the closures #148 added: a year-end shutdown answers "Saturday", which is the
// Saturday eleven days out. One format that is never ambiguous beats two that are each right half the
// time. `hourCycle: "h23"` keeps midnight at 00:00 instead of the 24:00 some ICU builds render.
function formatNextOpen(at: Date, timezone: string, locale: string): string {
  return new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    weekday: "long",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).format(at);
}

export type AwayRender =
  | { send: false; reason: "not_configured" | "no_next_open" }
  | { send: true; text: string };

// The text the customer receives, or why they receive nothing.
//
// `no_next_open` is the case worth naming: the copy promises a return time ("we are back {next_open}")
// and the schedule never opens inside nextOpenAt's horizon, so there is no honest value to put there.
// Dropping the placeholder would leave the operator's sentence mutilated ("we are back ."), and filling
// it with anything else would be the product inventing a commitment nobody made. Copy that makes no
// promise is unaffected — it goes out as written.
export function renderAwayMessage(params: {
  copy: string;
  schedule: Schedule;
  now: Date;
}): AwayRender {
  const copy = params.copy.trim();
  if (!copy) return { send: false, reason: "not_configured" };

  const used = NEXT_OPEN_PLACEHOLDERS.filter((p) => copy.includes(p.token));
  if (used.length === 0) return { send: true, text: copy };

  const next = nextOpenAt(params.schedule, params.now);
  if (next === null) return { send: false, reason: "no_next_open" };

  let text = copy;
  for (const { token, locale } of used) {
    text = text.replaceAll(
      token,
      formatNextOpen(next, params.schedule.timezone, locale),
    );
  }
  return { send: true, text };
}
