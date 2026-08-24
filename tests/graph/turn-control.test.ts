import { describe, expect, test } from "bun:test";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  cancelConversationTurns,
  guardChatwootClient,
  guardTools,
  registerTurn,
  releaseTurn,
  TurnControl,
  TurnTerminatedError,
} from "@/graph/turn-control";
import type { ChatwootClient } from "@/modules/chatwoot/client";

describe("turn control", () => {
  test("a human intervention cancels every overlapping turn for the conversation", () => {
    const first = registerTurn("tenant:instance:conversation");
    const second = registerTurn("tenant:instance:conversation");
    try {
      expect(cancelConversationTurns("tenant:instance:conversation")).toBe(2);
      expect(first.reason).toBe("human_intervention");
      expect(second.reason).toBe("human_intervention");
      expect(first.signal.aborted).toBe(true);
      expect(second.signal.aborted).toBe(true);
    } finally {
      releaseTurn("tenant:instance:conversation", first);
      releaseTurn("tenant:instance:conversation", second);
    }
  });

  test("a human message also fences a debounce turn that registers late", () => {
    const threadId = "tenant:instance:late-conversation";
    cancelConversationTurns(threadId, 200);

    const stale = registerTurn(threadId, 199);
    const newer = registerTurn(threadId, 201);
    try {
      expect(stale.reason).toBe("human_intervention");
      expect(newer.reason).toBeNull();
    } finally {
      releaseTurn(threadId, stale);
      releaseTurn(threadId, newer);
    }
  });

  test("no Chatwoot action or queued tool may start after cancellation", async () => {
    let chatwootCalls = 0;
    let toolCalls = 0;
    const rawClient = {
      sendMessage: async () => {
        chatwootCalls += 1;
      },
      sendPrivateNote: async () => {
        chatwootCalls += 1;
      },
      toggleStatus: async () => {
        chatwootCalls += 1;
      },
    } as unknown as ChatwootClient;
    const rawTool = tool(
      async () => {
        toolCalls += 1;
        return "ok";
      },
      { name: "external_action", description: "test", schema: z.object({}) },
    );
    const control = new TurnControl();
    const client = guardChatwootClient(rawClient, control);
    const [guardedTool] = guardTools([rawTool], control);
    if (!guardedTool) throw new Error("guarded tool missing");
    control.terminate("human_intervention");

    expect(() => client.sendMessage(42, "late")).toThrow(TurnTerminatedError);
    expect(() => client.sendPrivateNote(42, "late")).toThrow(
      TurnTerminatedError,
    );
    expect(() => client.toggleStatus(42, "open")).toThrow(TurnTerminatedError);
    expect(() => guardedTool.invoke({})).toThrow(TurnTerminatedError);
    expect(chatwootCalls).toBe(0);
    expect(toolCalls).toBe(0);
  });
});
