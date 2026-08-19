import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  AIMessage,
  type BaseMessage,
  HumanMessage,
} from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { MemorySaver } from "@langchain/langgraph";
import { PrismaPg } from "@prisma/adapter-pg";
import { PrismaClient } from "@/../generated/prisma/client";
import { encryptJson } from "@/api/lib/crypto";
import { contactInboxThreadId } from "@/graph/checkpointer";
import { CONVERSATION_DIVIDER, MEMORY_HEAD_OPEN } from "@/graph/markers";
import { buildThreadStateGraph, THREAD_STATE_NODE } from "@/graph/thread-state";
import { type CompactPayload, runCompaction } from "@/modules/memory/compact";
import { seedChatwootInstance } from "../utils/chatwoot";
import { UsageReportingModel } from "../utils/scripted-models";

const appUrl = process.env.TEST_APP_DATABASE_URL;
const suUrl = process.env.MIGRATION_DATABASE_URL;
let dbUp = false;
let su: PrismaClient | undefined;
let app: PrismaClient | undefined;
if (appUrl && suUrl) {
  try {
    su = new PrismaClient({
      adapter: new PrismaPg({ connectionString: suUrl }),
    });
    await su.$queryRaw`SELECT 1`;
    app = new PrismaClient({
      adapter: new PrismaPg({ connectionString: appUrl }),
    });
    await app.$queryRaw`SELECT 1`;
    dbUp = true;
  } catch {
    dbUp = false;
  }
}
const appDb = app as PrismaClient;
const suDb = su as PrismaClient;

// The summarizer, scripted. `beforeReturn` is the whole reason this is not FakeListChatModel: the
// provider round-trip is the window during which a customer message can land on the thread, and the
// only way to test that window is to write into it from inside the call.
class SummarizerModel extends BaseChatModel {
  calls = 0;
  seen: string[] = [];
  constructor(
    private readonly reply: string,
    private readonly beforeReturn?: () => Promise<void>,
  ) {
    super({});
  }
  _llmType() {
    return "fake-summarizer";
  }
  async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    this.calls += 1;
    this.seen.push(messages.map((m) => String(m.content)).join("\n"));
    await this.beforeReturn?.();
    return {
      generations: [{ text: this.reply, message: new AIMessage(this.reply) }],
    };
  }
}

// A saver that lets a test write to the thread at an exact point in the compaction's checkpoint
// traffic. The window that matters is between the job's locked re-read and the update it derives
// from it: the advisory lock keeps INGESTION out of it, but a graph turn writes to this same thread
// holding no lock at all, so a customer message really can land there.
class HookedSaver extends MemorySaver {
  calls = 0;
  constructor(
    private readonly at: number,
    private readonly hook: () => Promise<void>,
  ) {
    super();
  }
  // biome-ignore lint/suspicious/noExplicitAny: mirrors the saver's own loose tuple typing
  override async getTuple(config: any): Promise<any> {
    const tuple = await super.getTuple(config);
    this.calls += 1;
    if (this.calls === this.at) await this.hook();
    return tuple;
  }
}

let tenantId = 0n;
let instanceId = 0n;
let agentId = 0n;

// A distinctive string that only ever exists in the seeded transcript, so a test can prove it did
// NOT leak into a place that promises to carry no message text.
const SEEDED_TEXT = "abacaxi-com-hortela-4471";

describe.skipIf(!dbUp)("memory compaction", () => {
  beforeAll(async () => {
    const t = await suDb.tenant.create({
      data: { name: "MC", slug: `mc-${process.pid}` },
    });
    tenantId = t.id;
    const inst = await seedChatwootInstance(suDb, {
      tenantId,
      accountId: 11,
      baseUrl: "https://chat.example.com",
      adminToken: encryptJson("ADMIN"),
    });
    instanceId = inst.id;
    const agent = await suDb.agent.create({
      data: {
        tenantId,
        name: "Atendente",
        systemPrompt: "Você é prestativa.",
        modelConfig: { provider: "openai", model: "gpt-5.4-mini" },
        settings: { memory: { compaction: { enabled: true } } },
      },
    });
    agentId = agent.id;
  });

  afterAll(async () => {
    if (tenantId) {
      for (const table of [
        "attendance_summaries",
        "execution_logs",
        "agent_threads",
        "conversations",
        "agents",
        "chatwoot_instances",
      ]) {
        await suDb.$executeRawUnsafe(
          `DELETE FROM ${table} WHERE tenant_id = ${tenantId}`,
        );
      }
      await suDb.$executeRawUnsafe(
        `DELETE FROM tenants WHERE id = ${tenantId}`,
      );
    }
    await suDb.$disconnect();
    await appDb.$disconnect();
  });

  async function setCompaction(enabled: boolean) {
    await suDb.agent.update({
      where: { id: agentId },
      data: { settings: { memory: { compaction: { enabled } } } },
    });
  }

  async function seedThread(
    saver: MemorySaver,
    threadId: string,
    messages: BaseMessage[],
  ) {
    const graph = buildThreadStateGraph(saver);
    for (const m of messages) {
      await graph.updateState(
        { configurable: { thread_id: threadId } },
        { messages: [m] },
        THREAD_STATE_NODE,
      );
    }
  }

  async function readThread(
    saver: MemorySaver,
    threadId: string,
  ): Promise<string[]> {
    const cp = await saver.get({ configurable: { thread_id: threadId } });
    const messages = ((
      cp?.channel_values as { messages?: BaseMessage[] } | undefined
    )?.messages ?? []) as BaseMessage[];
    return messages.map((m) => String(m.content));
  }

  // Two attendances on one thread: the first one (raw) is what a boundary trigger compacts.
  function twoAttendances(): BaseMessage[] {
    return [
      new HumanMessage(`quero marcar uma avaliação, ${SEEDED_TEXT}`),
      new AIMessage("Claro! Consegui terça 08h30, R$ 250."),
      new HumanMessage("pode ser, obrigado"),
      new HumanMessage(`${CONVERSATION_DIVIDER}\n\noi, voltei`),
      new AIMessage("Oi! Como posso ajudar?"),
    ];
  }

  function countFlowLines(threadId: string) {
    return suDb.executionLog.count({
      where: { tenantId, stage: "memory", threadId },
    });
  }

  // emitFlowEvent is fire-and-forget, so a test that reads too early sees zero and proves nothing.
  async function waitForFlowLines(threadId: string, n: number) {
    for (let i = 0; i < 40; i++) {
      if ((await countFlowLines(threadId)) >= n) return;
      await new Promise((r) => setTimeout(r, 25));
    }
  }

  const payload = (
    contactInboxId: number,
    conversationId: number,
    reason: "resolved" | "new_attendance",
  ): CompactPayload => ({
    instanceId,
    contactInboxId,
    conversationId,
    agentId,
    reason,
  });

  test("a closed attendance becomes one summary and the open one travels untouched", async () => {
    const saver = new MemorySaver();
    const contactInboxId = 5001;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    await seedThread(saver, threadId, twoAttendances());
    const model = new SummarizerModel("Ana marcou avaliação terça, R$ 250.");

    const res = await runCompaction(
      tenantId,
      payload(contactInboxId, 700, "new_attendance"),
      appDb,
      { checkpointer: saver, makeModel: () => model },
    );
    expect(res).toEqual({ outcome: "done" });

    const after = await readThread(saver, threadId);
    // head + the open attendance, in order. The three raw turns of the closed one are gone.
    expect(after.length).toBe(3);
    expect(after[0]).toStartWith(MEMORY_HEAD_OPEN);
    expect(after[0]).toContain("Ana marcou avaliação terça");
    expect(after[1]).toStartWith(CONVERSATION_DIVIDER);
    expect(after[2]).toBe("Oi! Como posso ajudar?");
    expect(after.some((c) => c.includes(SEEDED_TEXT))).toBe(false);

    // The summarizer read the closed turns and NOT the open ones — summarizing the conversation
    // still in progress is how a thread ends up with the same events described twice.
    expect(model.calls).toBe(1);
    expect(model.seen[0]).toContain(SEEDED_TEXT);
    expect(model.seen[0]).not.toContain("Oi! Como posso ajudar?");

    const rows = await suDb.attendanceSummary.findMany({ where: { tenantId } });
    expect(rows.length).toBe(1);
    expect(rows[0]?.conversationId).toBe(700);
    expect(rows[0]?.messageCount).toBe(3);
  });

  // Both triggers can fire for the same thread, and a job that failed late gets retried, so a second
  // run has to be free AND invisible. "Same content" is not enough: a run that rewrites the thread to
  // the same bytes still writes a checkpoint and still tells the operator a compaction happened.
  test("running it again touches nothing: no row, no generation, no write, no log line", async () => {
    const saver = new MemorySaver();
    const contactInboxId = 5002;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    await seedThread(saver, threadId, twoAttendances());
    const model = new SummarizerModel("resumo");

    const args = [
      tenantId,
      payload(contactInboxId, 701, "new_attendance"),
      appDb,
      { checkpointer: saver, makeModel: () => model },
    ] as const;
    await runCompaction(...args);
    const afterFirst = await readThread(saver, threadId);
    const checkpointAfterFirst = (
      await saver.get({ configurable: { thread_id: threadId } })
    )?.id;
    await waitForFlowLines(threadId, 1);

    await runCompaction(...args);

    expect(await readThread(saver, threadId)).toEqual(afterFirst);
    expect(model.calls).toBe(1);
    // No new checkpoint: the second run never wrote to the thread at all.
    expect(
      (await saver.get({ configurable: { thread_id: threadId } }))?.id,
    ).toBe(checkpointAfterFirst);
    const rows = await suDb.attendanceSummary.findMany({
      where: { tenantId, contactInboxId },
    });
    expect(rows.length).toBe(1);
    // And the trail still shows ONE compaction, not two.
    expect(await countFlowLines(threadId)).toBe(1);
  });

  // The failure this guards against loses a customer's message, silently. The summarizer takes
  // seconds, the thread is append-only during that time, and a rewrite computed from the OLD read
  // would delete whatever arrived in between.
  test("a message that arrives while the summarizer runs survives the rewrite", async () => {
    const saver = new MemorySaver();
    const contactInboxId = 5003;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    await seedThread(saver, threadId, twoAttendances());

    const model = new SummarizerModel("resumo", async () => {
      await seedThread(saver, threadId, [
        new HumanMessage("esqueci de perguntar uma coisa"),
      ]);
    });

    await runCompaction(
      tenantId,
      payload(contactInboxId, 702, "new_attendance"),
      appDb,
      { checkpointer: saver, makeModel: () => model },
    );

    const after = await readThread(saver, threadId);
    expect(after.at(-1)).toBe("esqueci de perguntar uma coisa");
    expect(after[0]).toStartWith(MEMORY_HEAD_OPEN);
    // and the open attendance is still whole
    expect(after.some((c) => c.startsWith(CONVERSATION_DIVIDER))).toBe(true);
  });

  // /reset deletes the whole thread (webhook.ts), and it can land while a compaction is mid-
  // summarize. Writing the rewrite anyway would put memory back that an operator just deleted on
  // purpose, rendered from rows the reset already cleared.
  test("a thread reset during the summarizer aborts the rewrite", async () => {
    const saver = new MemorySaver();
    const contactInboxId = 5008;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    await seedThread(saver, threadId, twoAttendances());
    const model = new SummarizerModel("resumo", async () => {
      await saver.deleteThread(threadId);
    });

    const res = await runCompaction(
      tenantId,
      payload(contactInboxId, 707, "new_attendance"),
      appDb,
      { checkpointer: saver, makeModel: () => model },
    );

    expect(res.outcome).toBe("fail");
    expect(await readThread(saver, threadId)).toEqual([]);
  });

  // The same race, with the customer typing again right after the reset. The message COUNT lines up
  // with what was summarized, so only the identity of the messages tells the two apart — and getting
  // it wrong deletes three messages the customer just sent.
  test("a reset followed by fresh messages is caught by identity, not by count", async () => {
    const saver = new MemorySaver();
    const contactInboxId = 5009;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    await seedThread(saver, threadId, twoAttendances());
    const model = new SummarizerModel("resumo", async () => {
      await saver.deleteThread(threadId);
      await seedThread(saver, threadId, [
        new HumanMessage("oi"),
        new HumanMessage("tudo bem?"),
        new HumanMessage("consegue me ajudar?"),
      ]);
    });

    const res = await runCompaction(
      tenantId,
      payload(contactInboxId, 708, "new_attendance"),
      appDb,
      { checkpointer: saver, makeModel: () => model },
    );

    expect(res.outcome).toBe("fail");
    expect(await readThread(saver, threadId)).toEqual([
      "oi",
      "tudo bem?",
      "consegue me ajudar?",
    ]);
  });

  // The narrow window, and the reason the update names the ids it removes instead of clearing the
  // channel: REMOVE_ALL_MESSAGES replaces the whole list with what the update carries, so a message
  // written here would be erased by a compaction that never read it.
  test("a message written between the locked re-read and the update survives", async () => {
    const contactInboxId = 5010;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    let saver: HookedSaver | undefined;
    // The job's checkpoint traffic is: getState, then the locked re-read, then the update's own
    // internal read. Firing on the second is what puts the write inside the gap.
    saver = new HookedSaver(2, async () => {
      if (!saver) return;
      await seedThread(saver, threadId, [
        new HumanMessage("mensagem que chegou no meio da reescrita"),
      ]);
    });
    await seedThread(saver, threadId, twoAttendances());
    const before = saver.calls;
    saver.calls = 0;
    expect(before).toBeGreaterThan(0);

    await runCompaction(
      tenantId,
      payload(contactInboxId, 709, "new_attendance"),
      appDb,
      {
        checkpointer: saver,
        makeModel: () => new SummarizerModel("resumo"),
      },
    );

    const after = await readThread(saver, threadId);
    expect(after).toContain("mensagem que chegou no meio da reescrita");
    expect(after[0]).toStartWith(MEMORY_HEAD_OPEN);
  });

  test("a resolve that was undone inside the grace window does not compact", async () => {
    const saver = new MemorySaver();
    const contactInboxId = 5004;
    const conversationId = 703;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    await seedThread(saver, threadId, twoAttendances());
    await suDb.conversation.create({
      data: {
        tenantId,
        chatwootInstanceId: instanceId,
        chatwootConversationId: conversationId,
        // reopened between the arm and the run
        status: "open",
        threadId: `${tenantId}:${instanceId}:${conversationId}`,
        lastEventAt: new Date(),
      },
    });
    const model = new SummarizerModel("resumo");

    const before = await readThread(saver, threadId);
    await runCompaction(
      tenantId,
      payload(contactInboxId, conversationId, "resolved"),
      appDb,
      { checkpointer: saver, makeModel: () => model },
    );

    expect(await readThread(saver, threadId)).toEqual(before);
    expect(model.calls).toBe(0);
    expect(
      await suDb.attendanceSummary.count({
        where: { tenantId, contactInboxId },
      }),
    ).toBe(0);
  });

  test("a resolved attendance compacts the whole thread down to its memory", async () => {
    const saver = new MemorySaver();
    const contactInboxId = 5005;
    const conversationId = 704;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    await seedThread(saver, threadId, [
      new HumanMessage(`bom dia, ${SEEDED_TEXT}`),
      new AIMessage("Bom dia! Agendado para 18/08."),
    ]);
    await suDb.conversation.create({
      data: {
        tenantId,
        chatwootInstanceId: instanceId,
        chatwootConversationId: conversationId,
        status: "resolved",
        threadId: `${tenantId}:${instanceId}:${conversationId}`,
        lastEventAt: new Date(),
      },
    });

    await runCompaction(
      tenantId,
      payload(contactInboxId, conversationId, "resolved"),
      appDb,
      {
        checkpointer: saver,
        makeModel: () => new SummarizerModel("Agendado 18/08."),
      },
    );

    const after = await readThread(saver, threadId);
    expect(after.length).toBe(1);
    expect(after[0]).toStartWith(MEMORY_HEAD_OPEN);
    expect(after[0]).toContain("Agendado 18/08.");
  });

  // The grace window on a resolve is long enough for the contact to come back and open a NEW
  // attendance. The resolved conversation stays resolved, so the status check still passes, and
  // treating the whole thread as closed would summarize the conversation the agent is in the middle
  // of — the memory would then describe a conversation that is still happening.
  test("a resolve whose thread already moved on cuts at the divider instead", async () => {
    const saver = new MemorySaver();
    const contactInboxId = 5011;
    const conversationId = 710;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    await seedThread(saver, threadId, twoAttendances());
    await suDb.conversation.create({
      data: {
        tenantId,
        chatwootInstanceId: instanceId,
        chatwootConversationId: conversationId,
        status: "resolved",
        threadId: `${tenantId}:${instanceId}:${conversationId}`,
        lastEventAt: new Date(),
      },
    });
    // The thread has already moved on to a newer conversation.
    await suDb.agentThread.create({
      data: {
        tenantId,
        chatwootInstanceId: instanceId,
        contactInboxId,
        threadId,
        lastConversationId: conversationId + 1,
      },
    });

    await runCompaction(
      tenantId,
      payload(contactInboxId, conversationId, "resolved"),
      appDb,
      {
        checkpointer: saver,
        makeModel: () => new SummarizerModel("resumo"),
      },
    );

    const after = await readThread(saver, threadId);
    // The open attendance survived: head + its divider + its reply, not a lone head.
    expect(after.length).toBe(3);
    expect(after[1]).toStartWith(CONVERSATION_DIVIDER);
    expect(after[2]).toBe("Oi! Como posso ajudar?");
  });

  // The row is committed before the rewrite, so a job that dies after summarizing comes back with
  // the summary already stored. Paying the provider again for the same attendance is what the unique
  // key was supposed to prevent, and the key alone does not prevent it.
  test("a retry reuses the stored summary instead of generating a second one", async () => {
    const contactInboxId = 5012;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    const model = new SummarizerModel("resumo do atendimento");

    // First attempt: the rewrite is sabotaged (the thread is reset mid-summary), so it fails AFTER
    // the row was written.
    const saverA = new MemorySaver();
    await seedThread(saverA, threadId, twoAttendances());
    const sabotaged = new SummarizerModel("resumo do atendimento", async () => {
      await saverA.deleteThread(threadId);
    });
    const first = await runCompaction(
      tenantId,
      payload(contactInboxId, 711, "new_attendance"),
      appDb,
      { checkpointer: saverA, makeModel: () => sabotaged },
    );
    expect(first.outcome).toBe("fail");
    expect(sabotaged.calls).toBe(1);
    expect(
      await suDb.attendanceSummary.count({
        where: { tenantId, contactInboxId },
      }),
    ).toBe(1);

    // The retry finds the same cut and the stored row, and must not call the model again.
    const saverB = new MemorySaver();
    await seedThread(saverB, threadId, twoAttendances());
    const second = await runCompaction(
      tenantId,
      payload(contactInboxId, 711, "new_attendance"),
      appDb,
      { checkpointer: saverB, makeModel: () => model },
    );
    expect(second).toEqual({ outcome: "done" });
    expect(model.calls).toBe(0);
    expect((await readThread(saverB, threadId))[0]).toContain(
      "resumo do atendimento",
    );
  });

  // A billed generation that nobody is waiting on is exactly how model spend goes missing from the
  // cost report: no customer notices, no latency shows up, and with compaction on by default this
  // runs once per closed attendance across every agent.
  test("the summary generation is billed to the tenant like any other", async () => {
    const saver = new MemorySaver();
    const contactInboxId = 5013;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    await seedThread(saver, threadId, twoAttendances());

    await runCompaction(
      tenantId,
      payload(contactInboxId, 712, "new_attendance"),
      appDb,
      {
        checkpointer: saver,
        makeModel: () => new UsageReportingModel(["resumo"]),
      },
    );

    // UsageCapture persists fire-and-forget, like the flow lines.
    let usage = null;
    for (let i = 0; i < 40 && !usage; i++) {
      usage = await suDb.llmUsage.findFirst({
        where: { tenantId, threadId, node: "memory_compact" },
      });
      if (!usage) await new Promise((r) => setTimeout(r, 25));
    }
    expect(usage).not.toBeNull();
    expect(usage?.promptTokens).toBeGreaterThan(0);
    expect(usage?.completionTokens).toBeGreaterThan(0);
  });

  // cancelPendingJob only reaches a job still PENDING, so a compaction already claimed — provider
  // call in flight — outlives the /reset that ran a second ago. Writing its row anyway would restore
  // memory the operator explicitly deleted, with nothing to say where it came from.
  test("a reset that lands mid-compaction stops the summary from being written", async () => {
    const saver = new MemorySaver();
    const contactInboxId = 5014;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    await seedThread(saver, threadId, twoAttendances());
    await suDb.agentThread.create({
      data: {
        tenantId,
        chatwootInstanceId: instanceId,
        contactInboxId,
        threadId,
        lastConversationId: 713,
      },
    });

    const before = await readThread(saver, threadId);
    const res = await runCompaction(
      tenantId,
      payload(contactInboxId, 713, "new_attendance"),
      appDb,
      {
        checkpointer: saver,
        makeModel: () =>
          // /reset deletes the AgentThread row; the id this job started with is the generation token.
          new SummarizerModel("resumo", async () => {
            await suDb.agentThread.deleteMany({
              where: {
                tenantId,
                chatwootInstanceId: instanceId,
                contactInboxId,
              },
            });
          }),
      },
    );

    expect(res).toEqual({ outcome: "done" });
    expect(
      await suDb.attendanceSummary.count({
        where: { tenantId, contactInboxId },
      }),
    ).toBe(0);
    expect(await readThread(saver, threadId)).toEqual(before);
  });

  test("the switch is honored at execution, not only at arming time", async () => {
    const saver = new MemorySaver();
    const contactInboxId = 5006;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    await seedThread(saver, threadId, twoAttendances());
    const model = new SummarizerModel("resumo");

    await setCompaction(false);
    try {
      const before = await readThread(saver, threadId);
      await runCompaction(
        tenantId,
        payload(contactInboxId, 705, "new_attendance"),
        appDb,
        { checkpointer: saver, makeModel: () => model },
      );
      expect(await readThread(saver, threadId)).toEqual(before);
      expect(model.calls).toBe(0);
    } finally {
      await setCompaction(true);
    }
  });

  // `detail` on ExecutionLog carries counts and ids, never message text. Compaction is the one stage
  // whose whole input is the customer's words, so the promise is worth asserting here.
  test("the turn trail records the compaction with counts only", async () => {
    const saver = new MemorySaver();
    const contactInboxId = 5007;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    await seedThread(saver, threadId, twoAttendances());

    await runCompaction(
      tenantId,
      payload(contactInboxId, 706, "new_attendance"),
      appDb,
      {
        checkpointer: saver,
        makeModel: () => new SummarizerModel("resumo do atendimento"),
      },
    );

    await waitForFlowLines(threadId, 1);
    const row = await suDb.executionLog.findFirst({
      where: { tenantId, stage: "memory", threadId },
    });
    expect(row).not.toBeNull();
    expect(row?.level).toBe("info");
    const detail = JSON.stringify(row?.detail ?? {});
    expect(detail).toContain('"messagesCompacted":3');
    expect(detail).not.toContain(SEEDED_TEXT);
    expect(detail).not.toContain("resumo do atendimento");
  });
});
