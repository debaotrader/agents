import type { BaseMessage } from "@langchain/core/messages";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

export const NUDGE_MESSAGE_MAX_CHARS = 4000;

export type NudgeDecision =
  | { action: "send"; message: string }
  | { action: "skip" };

export class NudgeDecisionState {
  #decision: NudgeDecision | null = null;

  get decision(): NudgeDecision | null {
    return this.#decision;
  }

  record(decision: NudgeDecision): boolean {
    if (this.#decision !== null) return false;
    if (decision.action === "send") {
      const message = decision.message.trim();
      if (!message || message.length > NUDGE_MESSAGE_MAX_CHARS) return false;
      this.#decision = { action: "send", message };
    } else {
      this.#decision = decision;
    }
    return true;
  }
}

const finishNudgeSchema = z.discriminatedUnion("action", [
  z
    .object({
      action: z.literal("send"),
      message: z.string().trim().min(1).max(NUDGE_MESSAGE_MAX_CHARS),
    })
    .strict(),
  z.object({ action: z.literal("skip") }).strict(),
]);

export function buildNudgeDecisionTools(
  state: NudgeDecisionState,
): StructuredToolInterface[] {
  const finish = tool(
    async (decision: NudgeDecision) => {
      const recorded = state.record(decision);
      return recorded
        ? "Nudge decision recorded. End the turn now."
        : "A terminal nudge decision was already recorded. End the turn now.";
    },
    {
      name: "finish_nudge",
      description:
        "Terminally decide this proactive nudge. Use action=send with the exact single public message, or action=skip with no message. After calling it, end the turn and produce no text.",
      schema: finishNudgeSchema,
    },
  );
  const skip = tool(
    async (_args: { reason?: string }) => {
      state.record({ action: "skip" });
      return "Nudge skipped. End the turn now and produce no text.";
    },
    {
      name: "skip_reply",
      description:
        "Terminally skip this proactive nudge. After calling it, end the turn and produce no text.",
      schema: z.object({ reason: z.string().optional() }),
    },
  );
  return [finish, skip];
}

// Terminal decision names are capabilities, not operator-extensible aliases. A legacy HTTP tool or
// another source using either name must never shadow the state-writing implementation; providers also
// reject or ambiguously bind duplicate names. Preserve the first ordinary tool and let these two
// authoritative tools win regardless of source order.
export function mergeNudgeDecisionTools(
  tools: StructuredToolInterface[],
  state: NudgeDecisionState,
): StructuredToolInterface[] {
  const terminalTools = buildNudgeDecisionTools(state);
  const terminalNames = new Set(
    terminalTools.map((candidate) => candidate.name),
  );
  // These tools can reach the customer from inside ToolNode, before finish_nudge is validated. A
  // proactive turn has no deferred attachment/reaction pipeline, so exposing them would recreate a
  // second public-output owner. Handoff is intentionally absent: it records its customer line and
  // the nudge runner delivers that line only after the structured decision boundary.
  const publicSideEffectNames = new Set([
    "send_image",
    "react_to_message",
    "drive_send_file",
  ]);
  const seen = new Set<string>();
  const ordinaryTools = tools
    .filter((candidate) => {
      if (
        terminalNames.has(candidate.name) ||
        publicSideEffectNames.has(candidate.name) ||
        seen.has(candidate.name)
      ) {
        return false;
      }
      seen.add(candidate.name);
      return true;
    })
    .map(
      (candidate) =>
        new Proxy(candidate, {
          get(target, prop, receiver) {
            const value = Reflect.get(target, prop, receiver);
            if (
              typeof value !== "function" ||
              (prop !== "invoke" && prop !== "call")
            ) {
              return value;
            }
            return (...args: unknown[]) => {
              if (state.decision !== null) {
                return "A terminal nudge decision was already recorded. No further tool action was executed.";
              }
              return Reflect.apply(value, target, args);
            };
          },
        }),
    );
  return [...ordinaryTools, ...terminalTools];
}

function responseWasTruncated(message: BaseMessage): boolean {
  if (message.getType() !== "ai") return false;
  const metadata = message.response_metadata as Record<string, unknown>;
  const status =
    typeof metadata.status === "string"
      ? metadata.status.trim().toLowerCase()
      : "";
  if (status === "incomplete") return true;

  const incompleteDetails =
    typeof metadata.incomplete_details === "object" &&
    metadata.incomplete_details !== null
      ? (metadata.incomplete_details as Record<string, unknown>)
      : typeof metadata.incompleteDetails === "object" &&
          metadata.incompleteDetails !== null
        ? (metadata.incompleteDetails as Record<string, unknown>)
        : null;
  const raw =
    metadata.finish_reason ??
    metadata.finishReason ??
    metadata.stop_reason ??
    metadata.stopReason ??
    incompleteDetails?.reason;
  if (typeof raw !== "string") return false;
  return ["length", "max_tokens", "max_output_tokens"].includes(
    raw.trim().toLowerCase(),
  );
}

export function resolveNudgeDecision(
  state: NudgeDecisionState,
  messages: BaseMessage[],
): NudgeDecision | null {
  const lastHuman = messages.findLastIndex(
    (message) => message.getType() === "human",
  );
  if (messages.slice(lastHuman + 1).some(responseWasTruncated)) return null;
  return state.decision;
}
