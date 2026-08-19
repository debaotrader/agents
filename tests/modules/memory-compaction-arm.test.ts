import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph";
import { PrismaPg } from "@prisma/adapter-pg";
import { PrismaClient } from "@/../generated/prisma/client";
import { encryptJson } from "@/api/lib/crypto";
import { contactInboxThreadId } from "@/graph/checkpointer";
import { CONVERSATION_DIVIDER } from "@/graph/markers";
import { runAgentTurn } from "@/graph/runtime";
import type { ChatwootClient } from "@/modules/chatwoot/client";
import { normalizeChatwootEvent } from "@/modules/chatwoot/normalize";
import type { NormalizedChatwootEvent } from "@/modules/chatwoot/types";
import { processChatwootDelivery } from "@/modules/chatwoot/webhook";
import { seedChatwootInstance } from "../utils/chatwoot";

// The other half of the compaction tests, and the half that would otherwise be missing entirely:
// `tests/modules/memory-compaction.test.ts` drives `runCompaction` directly, so every one of its
// cases passes with the ARMING dead. Nothing there notices if the webhook condition never fires, and
// a feature that is never armed is a feature that does nothing while its suite stays green.
//
// So this drives the real receiver (`processChatwootDelivery`, no seam for the arm) with the payload
// Chatwoot actually sends on a resolve, and looks for the job row.

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

const INBOX_ID = 61;
const CONTACT_INBOX_ID = 61_000;
let tenantId = 0n;
let instanceId = 0n;
let agentId = 0n;
let deliverySeq = 0;

// Everything outbound lands on a double: this suite is about a DB row appearing, and a real fetch
// would only add flakiness.
const realFetch = globalThis.fetch;

describe.skipIf(!dbUp)("memory compaction: arming from the webhook", () => {
  beforeAll(async () => {
    globalThis.fetch = (async (
      _input: RequestInfo | URL,
      _init?: RequestInit,
    ) =>
      new Response(JSON.stringify({}), {
        status: 200,
        headers: { "content-type": "application/json" },
      })) as typeof globalThis.fetch;
    const t = await suDb.tenant.create({
      data: { name: "MCA", slug: `mca-${process.pid}` },
    });
    tenantId = t.id;
    const inst = await seedChatwootInstance(suDb, {
      tenantId,
      accountId: 12,
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
        settings: { debounce: { enabled: false } },
      },
    });
    agentId = agent.id;
    await suDb.chatwootAgentBot.create({
      data: {
        tenantId,
        chatwootInstanceId: instanceId,
        agentId,
        chatwootAgentBotId: 9,
        accessToken: encryptJson("BOT"),
        webhookSecret: encryptJson("S"),
        webhookRouteTokenHash: `mca-route-${process.pid}`,
        name: "Atendente",
      },
    });
    await suDb.inbox.create({
      data: {
        tenantId,
        chatwootInstanceId: instanceId,
        chatwootInboxId: INBOX_ID,
        name: "Suporte",
        agentId,
      },
    });
  });

  afterAll(async () => {
    globalThis.fetch = realFetch;
    if (!dbUp) return;
    for (const table of [
      "scheduler_jobs",
      "chatwoot_webhook_deliveries",
      "conversations",
      "inboxes",
      "chatwoot_agent_bots",
      "agents",
      "chatwoot_instances",
    ]) {
      await suDb
        .$executeRawUnsafe(`DELETE FROM ${table} WHERE tenant_id = ${tenantId}`)
        .catch(() => {});
    }
    await suDb.$executeRawUnsafe(`DELETE FROM tenants WHERE id = ${tenantId}`);
    await suDb.$disconnect();
    await appDb.$disconnect();
  });

  function convPayload(convId: number, status: string, updatedAt: number) {
    return {
      id: convId,
      inbox_id: INBOX_ID,
      status,
      contact_inbox: { id: CONTACT_INBOX_ID + convId },
      meta: { assignee_type: null, assignee: null },
      channel: "Channel::Api",
      last_activity_at: Math.floor(Date.now() / 1000),
      updated_at: updatedAt,
    };
  }

  // NOTE: A conversation_* event carries the conversation's fields at the ROOT of the payload, not
  // nested under `conversation` the way a message event does (see normalize.ts).
  async function deliver(event: string, conversation: Record<string, unknown>) {
    deliverySeq += 1;
    const n = normalizeChatwootEvent({ event, ...conversation });
    if (!n) throw new Error("payload did not normalize");
    const delivery = await suDb.chatwootWebhookDelivery.create({
      data: {
        tenantId,
        chatwootInstanceId: instanceId,
        deliveryId: `mca-${process.pid}-${deliverySeq}`,
        event,
        status: "PENDING",
      },
      select: { id: true },
    });
    await processChatwootDelivery({
      tenantId,
      instanceId,
      deliveryRowId: delivery.id,
      agentBotId: 9,
      normalized: n,
      base: appDb,
    });
  }

  function jobFor(convId: number) {
    return suDb.schedulerJob.findFirst({
      where: {
        tenantId,
        kind: "MEMORY_COMPACT",
        dedupeKey: contactInboxThreadId(
          tenantId,
          instanceId,
          CONTACT_INBOX_ID + convId,
        ),
      },
    });
  }

  // NOTE: A strictly increasing stamp, NOT the wall clock. The mirror orders writes by the
  // conversation's own `updated_at`, and two rounds landing inside the same second would make the
  // second one look stale — the mirror would drop it, no transition would be seen, and the test
  // would fail for a reason that has nothing to do with what it is testing.
  let stamp = Math.floor(Date.now() / 1000);
  async function resolve(convId: number) {
    stamp += 1;
    await deliver(
      "conversation_updated",
      convPayload(convId, "pending", stamp),
    );
    stamp += 1;
    await deliver(
      "conversation_status_changed",
      convPayload(convId, "resolved", stamp),
    );
  }

  test("a conversation transitioning to resolved arms the compaction job", async () => {
    const convId = 401;
    await resolve(convId);

    const job = await jobFor(convId);
    expect(job).not.toBeNull();
    expect(job?.status).toBe("PENDING");
    const payload = job?.payload as Record<string, unknown>;
    expect(payload.reason).toBe("resolved");
    expect(payload.conversationId).toBe(convId);
    expect(payload.contactInboxId).toBe(CONTACT_INBOX_ID + convId);
    expect(payload.agentId).toBe(String(agentId));
    // The grace window: a resolve can be undone, so the job must not be due immediately.
    expect(job?.runAt.getTime()).toBeGreaterThan(Date.now() + 60_000);
  });

  test("a re-delivered resolve does not stack a second job", async () => {
    const convId = 402;
    await resolve(convId);
    await resolve(convId);

    const jobs = await suDb.schedulerJob.count({
      where: {
        tenantId,
        kind: "MEMORY_COMPACT",
        dedupeKey: contactInboxThreadId(
          tenantId,
          instanceId,
          CONTACT_INBOX_ID + convId,
        ),
      },
    });
    expect(jobs).toBe(1);
  });

  // The OTHER arm: a new attendance opening on the thread. Its ordering is the whole finding — the
  // job looks for the divider, and the divider only reaches the checkpointer when the turn's invoke
  // writes it. Armed before that, a due-now job can be claimed first, find no boundary, and retire
  // itself as a no-op; the input guardrail makes it deterministic rather than a race, because it can
  // return before the graph is invoked at all.
  test("the boundary arm happens only after the divider is persisted", async () => {
    const contactInboxId = 62_500;
    const threadId = contactInboxThreadId(tenantId, instanceId, contactInboxId);
    const checkpointer = new MemorySaver();
    const sent: number[] = [];
    const client = {
      sendMessage: async (conversationId: number) => {
        sent.push(conversationId);
        return {};
      },
      toggleTyping: async () => ({}),
      getMessages: async () => [],
    } as unknown as ChatwootClient;

    // Asserted from INSIDE the model call, which is the only place that can tell the two orderings
    // apart: at this point the invoke has not returned, so an arm placed before it would already
    // have left a row.
    let jobsAtModelTime = -1;
    const model = {
      invoke: async () => {
        jobsAtModelTime = await suDb.schedulerJob.count({
          where: { tenantId, kind: "MEMORY_COMPACT", dedupeKey: threadId },
        });
        return new AIMessage("Claro!");
      },
      bindTools: (_t: unknown) => ({
        invoke: async () => {
          jobsAtModelTime = await suDb.schedulerJob.count({
            where: { tenantId, kind: "MEMORY_COMPACT", dedupeKey: threadId },
          });
          return new AIMessage("Claro!");
        },
      }),
    };

    const turn = (convId: number, messageId: number) =>
      runAgentTurn({
        tenantId,
        instanceId,
        agentBotId: 9,
        event: {
          event: "message_created",
          conversationId: convId,
          inboxId: INBOX_ID,
          status: "pending",
          assigneeType: null,
          assigneeId: null,
          assigneeName: null,
          contactInboxId,
          message: {
            id: messageId,
            content: "oi",
            messageType: "incoming",
            private: false,
          },
        } as NormalizedChatwootEvent,
        base: appDb,
        deps: {
          makeModel: () => model as unknown as BaseChatModel,
          makeClient: async () => client,
          checkpointer,
        },
      });

    for (const convId of [501, 502]) {
      await suDb.conversation.create({
        data: {
          tenantId,
          chatwootInstanceId: instanceId,
          chatwootConversationId: convId,
          contactInboxId,
          status: "pending",
          threadId: `${tenantId}:${instanceId}:${convId}`,
          lastEventAt: new Date(),
        },
      });
    }
    // First attendance: no previous conversation on this thread, so no boundary and no arm.
    expect(await turn(501, 9001)).toBe("posted");
    expect(
      await suDb.schedulerJob.count({
        where: { tenantId, kind: "MEMORY_COMPACT", dedupeKey: threadId },
      }),
    ).toBe(0);

    // Second attendance on the same thread: the divider rides in, and the job is armed after it.
    expect(await turn(502, 9002)).toBe("posted");
    expect(jobsAtModelTime).toBe(0);
    const job = await suDb.schedulerJob.findFirst({
      where: { tenantId, kind: "MEMORY_COMPACT", dedupeKey: threadId },
    });
    expect(job).not.toBeNull();
    const boundaryPayload = (job?.payload ?? {}) as Record<string, unknown>;
    expect(boundaryPayload.reason).toBe("new_attendance");
    expect(boundaryPayload.conversationId).toBe(501);

    // And by the time that job exists, the boundary it looks for is really in the thread.
    const cp = await checkpointer.get({
      configurable: { thread_id: threadId },
    });
    const messages = ((
      cp?.channel_values as { messages?: { content: unknown }[] } | undefined
    )?.messages ?? []) as { content: unknown }[];
    expect(
      messages.some((m) => String(m.content).startsWith(CONVERSATION_DIVIDER)),
    ).toBe(true);
  });

  // The dedupeKey is the THREAD, so one row serves every attendance this contact will ever have.
  // `attempts` is what the scheduler retires a job on, and it does not reset on re-arm — so without
  // this, four transient failures spread over months make the NEXT attendance dead-letter on its
  // first failure, and that contact never compacts again.
  test("each attendance gets its own retry budget", async () => {
    const convId = 404;
    await resolve(convId);
    const key = contactInboxThreadId(
      tenantId,
      instanceId,
      CONTACT_INBOX_ID + convId,
    );
    await suDb.schedulerJob.updateMany({
      where: { tenantId, kind: "MEMORY_COMPACT", dedupeKey: key },
      data: { attempts: 4, status: "DEAD" },
    });

    await resolve(convId);

    const job = await suDb.schedulerJob.findFirst({
      where: { tenantId, kind: "MEMORY_COMPACT", dedupeKey: key },
    });
    expect(job?.attempts).toBe(0);
    expect(job?.status).toBe("PENDING");
  });

  test("an agent with compaction off arms nothing at all", async () => {
    const convId = 403;
    await suDb.agent.update({
      where: { id: agentId },
      data: {
        settings: {
          debounce: { enabled: false },
          memory: { compaction: { enabled: false } },
        },
      },
    });
    try {
      await resolve(convId);
      expect(await jobFor(convId)).toBeNull();
    } finally {
      await suDb.agent.update({
        where: { id: agentId },
        data: { settings: { debounce: { enabled: false } } },
      });
    }
  });
});
