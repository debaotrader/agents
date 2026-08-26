import { describe, expect, test } from "bun:test";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import type { StructuredToolInterface } from "@langchain/core/tools";
import {
  buildNudgeDecisionTools,
  mergeNudgeDecisionTools,
  NUDGE_MESSAGE_MAX_CHARS,
  NudgeDecisionState,
  resolveNudgeDecision,
} from "@/graph/nudge-decision";

function byName(tools: StructuredToolInterface[], name: string) {
  const found = tools.find((candidate) => candidate.name === name);
  if (!found) throw new Error(`tool not found: ${name}`);
  return found;
}

describe("nudge public-output boundary", () => {
  test("terminal tools override legacy collisions and the final toolset is unique", () => {
    const legacySkip = { name: "skip_reply" } as StructuredToolInterface;
    const legacyFinish = { name: "finish_nudge" } as StructuredToolInterface;
    const firstSearch = { name: "search" } as StructuredToolInterface;
    const duplicateSearch = { name: "search" } as StructuredToolInterface;
    const sendImage = { name: "send_image" } as StructuredToolInterface;
    const react = { name: "react_to_message" } as StructuredToolInterface;
    const driveSend = { name: "drive_send_file" } as StructuredToolInterface;

    const tools = mergeNudgeDecisionTools(
      [
        legacySkip,
        legacyFinish,
        firstSearch,
        duplicateSearch,
        sendImage,
        react,
        driveSend,
      ],
      new NudgeDecisionState(),
    );
    const names = tools.map((candidate) => candidate.name);

    expect(names).toEqual(["search", "finish_nudge", "skip_reply"]);
    expect(new Set(names).size).toBe(names.length);
    expect(tools).not.toContain(legacySkip);
    expect(tools).not.toContain(legacyFinish);
    expect(tools).not.toContain(duplicateSearch);
  });

  test("free-form model reasoning without a terminal decision fails closed", () => {
    const state = new NudgeDecisionState();
    const decision = resolveNudgeDecision(state, [
      new HumanMessage("follow-up"),
      new AIMessage("The customer already replied, so I should skip this."),
    ]);

    expect(decision).toBeNull();
  });

  test("skip_reply stays terminal even if the model emits text afterwards", async () => {
    const state = new NudgeDecisionState();
    const tools = buildNudgeDecisionTools(state);

    await byName(tools, "skip_reply").invoke({ reason: "already answered" });
    await byName(tools, "finish_nudge").invoke({
      action: "send",
      message: "This must never be delivered.",
    });

    expect(
      resolveNudgeDecision(state, [
        new HumanMessage("follow-up"),
        new AIMessage("This must never be delivered."),
      ]),
    ).toEqual({ action: "skip" });
  });

  test("finish_nudge send preserves the exact bounded message", async () => {
    const state = new NudgeDecisionState();
    const tools = buildNudgeDecisionTools(state);

    await byName(tools, "finish_nudge").invoke({
      action: "send",
      message: "Oi! Ainda precisa de ajuda?",
    });

    expect(
      resolveNudgeDecision(state, [new HumanMessage("follow-up")]),
    ).toEqual({
      action: "send",
      message: "Oi! Ainda precisa de ajuda?",
    });
  });

  test("finish_reason=length invalidates an otherwise valid send decision", async () => {
    const state = new NudgeDecisionState();
    await byName(buildNudgeDecisionTools(state), "finish_nudge").invoke({
      action: "send",
      message: "Mensagem aparentemente completa.",
    });

    expect(
      resolveNudgeDecision(state, [
        new HumanMessage("follow-up"),
        new AIMessage({
          content: "",
          response_metadata: { finish_reason: "length" },
        }),
      ]),
    ).toBeNull();
  });

  test("OpenAI Responses status=incomplete invalidates a terminal decision", async () => {
    const state = new NudgeDecisionState();
    await byName(buildNudgeDecisionTools(state), "finish_nudge").invoke({
      action: "send",
      message: "Mensagem cortada pelo limite.",
    });

    expect(
      resolveNudgeDecision(state, [
        new HumanMessage("follow-up"),
        new AIMessage({
          content: "",
          response_metadata: {
            status: "incomplete",
            incomplete_details: { reason: "max_output_tokens" },
          },
        }),
      ]),
    ).toBeNull();
  });

  test.each([
    ["Anthropic", { stop_reason: "max_tokens" }],
    ["Gemini", { finishReason: "MAX_TOKENS" }],
  ])(
    "%s token-limit metadata invalidates a terminal decision",
    async (_provider, metadata) => {
      const state = new NudgeDecisionState();
      await byName(buildNudgeDecisionTools(state), "finish_nudge").invoke({
        action: "send",
        message: "Mensagem cortada pelo limite.",
      });

      expect(
        resolveNudgeDecision(state, [
          new HumanMessage("follow-up"),
          new AIMessage({ content: "", response_metadata: metadata }),
        ]),
      ).toBeNull();
    },
  );

  test("messages over the public cap never become a decision", async () => {
    const state = new NudgeDecisionState();
    await expect(
      byName(buildNudgeDecisionTools(state), "finish_nudge").invoke({
        action: "send",
        message: "x".repeat(NUDGE_MESSAGE_MAX_CHARS + 1),
      }),
    ).rejects.toThrow();
    expect(resolveNudgeDecision(state, [])).toBeNull();
  });
});
