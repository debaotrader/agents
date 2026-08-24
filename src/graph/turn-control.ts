import type { StructuredToolInterface } from "@langchain/core/tools";
import type { ChatwootClient } from "@/modules/chatwoot/client";

export type TurnTerminalReason = "skip_reply" | "human_intervention";

export class TurnTerminatedError extends Error {
  constructor(readonly reason: TurnTerminalReason) {
    super(`Turn terminated: ${reason}`);
    this.name = "TurnTerminatedError";
  }
}

export class TurnControl {
  readonly #abort = new AbortController();
  #reason: TurnTerminalReason | null = null;

  get signal(): AbortSignal {
    return this.#abort.signal;
  }

  get reason(): TurnTerminalReason | null {
    return this.#reason;
  }

  terminate(reason: TurnTerminalReason): boolean {
    if (this.#reason !== null) return false;
    this.#reason = reason;
    // skip_reply is called from inside the graph. Aborting that invocation while the tool is still
    // returning would turn a successful terminal decision into an execution error. A human
    // intervention is external, so it must interrupt the model/tool wait immediately.
    if (reason === "human_intervention") this.#abort.abort(reason);
    return true;
  }

  throwIfTerminal(): void {
    if (this.#reason !== null) throw new TurnTerminatedError(this.#reason);
  }
}

const activeTurns = new Map<string, Set<TurnControl>>();
const humanFences = new Map<
  string,
  { messageId: number; recordedAt: number }
>();
const HUMAN_FENCE_TTL_MS = 10 * 60_000;

function pruneHumanFences(now = Date.now()): void {
  for (const [threadId, fence] of humanFences) {
    if (now - fence.recordedAt > HUMAN_FENCE_TTL_MS)
      humanFences.delete(threadId);
  }
}

export function registerTurn(
  threadId: string,
  triggerMessageId?: number,
): TurnControl {
  pruneHumanFences();
  const control = new TurnControl();
  const turns = activeTurns.get(threadId) ?? new Set<TurnControl>();
  turns.add(control);
  activeTurns.set(threadId, turns);
  // A claimed debounce job can be between its DB claim and runtime registration when the human
  // webhook arrives. Keep a short message-id fence so that delayed registration is cancelled too,
  // not only turns already present in activeTurns. Chatwoot ids are monotonic per account.
  const fence = humanFences.get(threadId);
  if (
    fence &&
    triggerMessageId !== undefined &&
    triggerMessageId <= fence.messageId
  ) {
    control.terminate("human_intervention");
  }
  return control;
}

export function releaseTurn(threadId: string, control: TurnControl): void {
  const turns = activeTurns.get(threadId);
  if (!turns) return;
  turns.delete(control);
  if (turns.size === 0) activeTurns.delete(threadId);
}

export function cancelConversationTurns(
  threadId: string,
  humanMessageId?: number,
): number {
  pruneHumanFences();
  if (humanMessageId !== undefined) {
    const previous = humanFences.get(threadId);
    if (!previous || humanMessageId >= previous.messageId) {
      humanFences.set(threadId, {
        messageId: humanMessageId,
        recordedAt: Date.now(),
      });
    }
  }
  const turns = activeTurns.get(threadId);
  if (!turns) return 0;
  let cancelled = 0;
  for (const control of turns) {
    if (control.terminate("human_intervention")) cancelled += 1;
  }
  return cancelled;
}

export function isTurnTerminatedError(
  err: unknown,
): err is TurnTerminatedError {
  return err instanceof TurnTerminatedError;
}

// Every Chatwoot mutation goes through the per-turn client. Checking at call time fences public and
// private messages, attachments, typing, handoff, resolve, labels and any future client action.
export function guardChatwootClient(
  client: ChatwootClient,
  control: TurnControl,
): ChatwootClient {
  return new Proxy(client, {
    get(target, prop, receiver) {
      const value = Reflect.get(target, prop, receiver);
      if (typeof value !== "function") return value;
      return (...args: unknown[]) => {
        control.throwIfTerminal();
        return Reflect.apply(value, target, args);
      };
    },
  });
}

// HTTP, MCP, RAG, toolpack and native actions all expose invoke/call. Guarding both prevents a
// queued LangGraph ToolNode task from beginning an outward action after a human took the turn.
export function guardTools(
  tools: StructuredToolInterface[],
  control: TurnControl,
): StructuredToolInterface[] {
  return tools.map(
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
            control.throwIfTerminal();
            return Reflect.apply(value, target, args);
          };
        },
      }),
  );
}
