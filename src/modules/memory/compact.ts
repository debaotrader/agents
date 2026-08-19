import {
  type BaseMessage,
  HumanMessage,
  RemoveMessage,
} from "@langchain/core/messages";
import type { BaseCheckpointSaver } from "@langchain/langgraph-checkpoint";
import type { PrismaClient } from "@/../generated/prisma/client";
import logger from "@/api/lib/logger";
import { contactInboxThreadId, getCheckpointer } from "@/graph/checkpointer";
import { createChatModel } from "@/graph/models";
import { loadAgentConfig } from "@/graph/prepare";
import { buildThreadStateGraph, THREAD_STATE_NODE } from "@/graph/thread-state";
import { withEntityLock } from "@/lib/locks";
import { runScopedOn, type TenantContext } from "@/lib/tenancy";
import { emitFlowEvent } from "@/modules/flowlog/service";
import { enqueueJob } from "@/modules/scheduler/service";
import { type JobResult, registerJobHandler } from "@/modules/scheduler/worker";
import { renderMemoryHead, selectClosedPrefix } from "./cut";
import { readMemoryConfig } from "./settings";
import { summarizeAttendance } from "./summarize";

// Memory compaction: when an attendance ends, its raw turns on the contact's thread are replaced by
// one summary of it, so the thread becomes "N summarized attendances + the current one, raw".
//
// Cutting on the attendance boundary rather than at an arbitrary token offset is the whole point:
// the boundary already exists (CONVERSATION_DIVIDER, Conversation.status, AgentThread
// .lastConversationId), it is what a human agent's notes actually look like, and it is explainable
// to an operator — "8 atendimentos resumidos + o atual" is a sentence; "dropped 40k tokens from the
// middle" is not.
//
// Everything here runs OFF the hot path, as a scheduler job, after the reply was posted. No customer
// ever waits on the summarizer.

const GRACE_ON_RESOLVE_MS = 15 * 60_000;

function sysCtx(tenantId: bigint): TenantContext {
  return { tenantId, userId: null, role: "TENANT_ADMIN" };
}

// Why the trigger fired, which is the only thing the cut cannot work out on its own: "resolved"
// means the conversation the thread is CURRENTLY on has ended, so there is no open attendance to
// protect; "new_attendance" means a later conversation already opened, and the cut finds it by its
// divider.
export type CompactionReason = "resolved" | "new_attendance";

export interface ArmCompactionParams {
  tenantId: bigint;
  instanceId: bigint;
  contactInboxId: number;
  // The attendance that ended.
  conversationId: number;
  agentId: bigint;
  reason: CompactionReason;
  // The per-agent switch, already resolved by the caller (readMemoryConfig). Passed in rather than
  // re-read here so a call site that already holds the agent's config does not open a query, and so
  // this function has one job.
  enabled: boolean;
  base?: PrismaClient;
}

// Enqueues (or re-arms) the one compaction job for this thread. Best-effort by contract: a failure
// to arm must never break the webhook or the turn that called it.
export async function armCompaction(
  p: ArmCompactionParams,
): Promise<"armed" | "disabled" | "failed"> {
  if (!p.enabled) return "disabled";
  const threadId = contactInboxThreadId(
    p.tenantId,
    p.instanceId,
    p.contactInboxId,
  );
  try {
    await enqueueJob({
      tenantId: p.tenantId,
      kind: "MEMORY_COMPACT",
      // GUARANTEE 1 of 3 against compacting twice: SchedulerJob is unique on
      // (tenant, kind, dedupeKey) and enqueueJob upserts, so both triggers firing for the same
      // thread collapse into ONE row instead of two jobs racing each other over the same messages.
      dedupeKey: threadId,
      runAt: new Date(
        Date.now() + (p.reason === "resolved" ? GRACE_ON_RESOLVE_MS : 0),
      ),
      // This dedupeKey is the THREAD, so the same row is reused by every attendance this contact ever
      // has. Each attendance is new work and gets its own retry budget; otherwise failures accumulate
      // across months and one bad day retires compaction for that contact permanently.
      resetAttempts: true,
      payload: {
        instanceId: String(p.instanceId),
        contactInboxId: p.contactInboxId,
        conversationId: p.conversationId,
        agentId: String(p.agentId),
        reason: p.reason,
      },
      base: p.base,
    });
    return "armed";
  } catch (err) {
    logger.warn({ err }, "memory: could not arm compaction");
    return "failed";
  }
}

export interface CompactPayload {
  instanceId: bigint;
  contactInboxId: number;
  conversationId: number;
  agentId: bigint;
  reason: CompactionReason;
}

function parsePayload(raw: Record<string, unknown>): CompactPayload | null {
  const instanceId = raw.instanceId;
  const agentId = raw.agentId;
  const contactInboxId = raw.contactInboxId;
  const conversationId = raw.conversationId;
  if (
    typeof instanceId !== "string" ||
    typeof agentId !== "string" ||
    typeof contactInboxId !== "number" ||
    typeof conversationId !== "number"
  ) {
    return null;
  }
  try {
    return {
      instanceId: BigInt(instanceId),
      agentId: BigInt(agentId),
      contactInboxId,
      conversationId,
      reason: raw.reason === "resolved" ? "resolved" : "new_attendance",
    };
  } catch {
    return null;
  }
}

export interface CompactionDeps {
  checkpointer?: BaseCheckpointSaver;
  makeModel?: typeof createChatModel;
}

export async function runCompaction(
  tenantId: bigint,
  payload: CompactPayload,
  base: PrismaClient,
  deps: CompactionDeps = {},
): Promise<JobResult> {
  const { instanceId, contactInboxId, conversationId, agentId, reason } =
    payload;
  const graphThreadId = contactInboxThreadId(
    tenantId,
    instanceId,
    contactInboxId,
  );

  const loaded = await runScopedOn(base, sysCtx(tenantId), async (db) => {
    const agent = await db.agent.findUnique({
      where: { id: agentId },
      select: { settings: true },
    });
    // NOTE: The switch is re-read at execution, not trusted from arming time: a job can sit in the
    // queue past the moment an operator turns compaction off, and the operator's last word wins.
    if (!agent || !readMemoryConfig(agent.settings).compaction.enabled) {
      return "off" as const;
    }
    // A conversation that was reopened inside the grace window is NOT a closed attendance, and
    // compacting it would hand the model a summary of the very conversation it is still in the
    // middle of. The boundary trigger picks it up later, when a genuinely new attendance opens.
    if (reason === "resolved") {
      const conv = await db.conversation.findUnique({
        where: {
          tenantId_chatwootInstanceId_chatwootConversationId: {
            tenantId,
            chatwootInstanceId: instanceId,
            chatwootConversationId: conversationId,
          },
        },
        select: { status: true },
      });
      if (conv && conv.status !== "resolved") return "reopened" as const;
    }
    // Which conversation the thread is on RIGHT NOW. The resolve trigger waits out a grace window,
    // and a contact can open a new attendance inside it: the resolved conversation stays resolved,
    // so the status check above passes, and treating the whole thread as closed would summarize the
    // conversation the agent is in the middle of. When the thread has moved on, the divider is the
    // boundary again.
    const thread = await db.agentThread.findUnique({
      where: {
        tenantId_chatwootInstanceId_contactInboxId: {
          tenantId,
          chatwootInstanceId: instanceId,
          contactInboxId,
        },
      },
      select: { lastConversationId: true },
    });
    const cfg = await loadAgentConfig(db, {
      tenantId,
      instanceId,
      conversationId,
      agentId,
      threadId: graphThreadId,
    });
    if (!cfg) return null;
    return { cfg, lastConversationId: thread?.lastConversationId ?? null };
  });
  if (loaded === "off" || loaded === "reopened" || loaded === null) {
    return { outcome: "done" };
  }
  const cfg = loaded.cfg;
  const attendanceIsCurrent =
    loaded.lastConversationId === null ||
    loaded.lastConversationId === conversationId;

  const checkpointer = deps.checkpointer ?? (await getCheckpointer());
  const graph = buildThreadStateGraph(checkpointer);
  const threadCfg = { configurable: { thread_id: graphThreadId } };
  const state = await graph.getState(threadCfg);
  const messages = ((state.values as { messages?: BaseMessage[] } | undefined)
    ?.messages ?? []) as BaseMessage[];

  const cut = selectClosedPrefix(messages, {
    currentAttendanceClosed: reason === "resolved" && attendanceIsCurrent,
  });
  // GUARANTEE 3 of 3 against compacting twice: this is not a flag anyone has to remember to check.
  // After a compaction the thread holds the head plus the open attendance, so a second run finds an
  // empty closed chunk and stops HERE, before spending a generation. Running the job twice costs a
  // state read.
  if (cut.closed.length === 0) return { outcome: "done" };

  const summaryKey = {
    tenantId_chatwootInstanceId_contactInboxId_conversationId: {
      tenantId,
      chatwootInstanceId: instanceId,
      contactInboxId,
      conversationId,
    },
  };
  // A retry must not pay for a second generation. The row is committed before the rewrite, so a job
  // that failed AFTER summarizing comes back with the summary already stored: reuse it. The count is
  // the identity check — a reopened attendance that grew before closing again has a different cut,
  // and reusing a summary written from fewer messages would delete the rest without describing it.
  const existing = await runScopedOn(base, sysCtx(tenantId), (db) =>
    db.attendanceSummary.findUnique({
      where: summaryKey,
      select: { summary: true, messageCount: true },
    }),
  );
  let summary: string;
  if (existing && existing.messageCount === cut.closed.length) {
    summary = existing.summary;
  } else {
    const makeModel = deps.makeModel ?? createChatModel;
    const model = makeModel({
      ...cfg.mc,
      apiKey: cfg.apiKey,
      baseURL: cfg.credentialBaseUrl ?? cfg.mc.baseURL,
    });
    // Outside every lock: this is a provider round-trip, and holding a Postgres advisory lock across
    // the wire would block ingestion on this thread for as long as the model takes.
    const result = await summarizeAttendance(model, cut.closed);
    if (result.error) return { outcome: "fail", error: result.error };
    summary = result.summary;
  }

  // The row is committed BEFORE the thread is rewritten, on purpose. The two failure orders are not
  // equally bad: a row written whose rewrite never lands means the same turns get summarized again
  // later and the memory says something twice, while a rewrite that lands with no row means the
  // attendance is simply gone. Duplicated memory is recoverable by reading it; lost memory is not.
  if (summary) {
    await runScopedOn(base, sysCtx(tenantId), (db) =>
      // GUARANTEE 2 of 3: one row per attendance, forever. `upsert` rather than create+catch —
      // a P2002 caught inside an aborted transaction cannot recover with an update.
      db.attendanceSummary.upsert({
        where: summaryKey,
        create: {
          tenantId,
          chatwootInstanceId: instanceId,
          contactInboxId,
          conversationId,
          summary,
          messageCount: cut.closed.length,
        },
        update: { summary, messageCount: cut.closed.length },
      }),
    );
  }

  const rewrite = await runScopedOn(base, sysCtx(tenantId), (db) =>
    // The lock ingestion also takes, so an ingested message cannot interleave with the rewrite.
    // It is NOT the whole story: a graph TURN writes to this thread without taking any lock, which
    // is why the update below names the messages it removes instead of clearing the channel.
    withEntityLock(db, `ingest:${graphThreadId}`, async () => {
      const fresh = await graph.getState(threadCfg);
      const current = ((
        fresh.values as { messages?: BaseMessage[] } | undefined
      )?.messages ?? []) as BaseMessage[];
      const consumed = [...(cut.head ? [cut.head] : []), ...cut.closed];
      // The thread is append-only between the read and this write, so the messages we summarized
      // must still be its prefix. If they are not, something rewrote the thread underneath us (the
      // /reset command deletes it outright) and the safe move is to abandon this attempt rather than
      // delete messages we never read. A shorter thread is covered by the same comparison: past its
      // end `current[i]` is undefined, which never equals an id.
      for (let i = 0; i < consumed.length; i++) {
        if (current[i]?.id !== consumed[i]?.id) return "changed" as const;
      }
      const rows = await db.attendanceSummary.findMany({
        where: {
          tenantId,
          chatwootInstanceId: instanceId,
          contactInboxId,
        },
        orderBy: { createdAt: "asc" },
        select: { conversationId: true, summary: true, createdAt: true },
      });
      const head = renderMemoryHead(rows);
      // The update REMOVES BY ID and never clears the channel. REMOVE_ALL_MESSAGES would have been
      // shorter, and wrong: it replaces the whole list with what this update carries, so a message
      // appended between the read above and this write would be erased. Ingestion is held off by the
      // lock, but a graph TURN takes no lock and writes to this same thread, so that window is real
      // and a customer's message is what falls into it. Naming the ids leaves everything else alone,
      // whenever it arrived.
      //
      // The head reuses the id of the FIRST message it replaces, which is what keeps it at the front:
      // the reducer replaces a same-id message in place and appends an unknown-id one at the end, and
      // a memory head sitting after the conversation is not a header, it is a footnote.
      const survivorId = consumed[0]?.id;
      const dropped = consumed.filter((m) => m.id !== survivorId);
      await graph.updateState(
        threadCfg,
        {
          messages: [
            ...(head && survivorId
              ? [new HumanMessage({ id: survivorId, content: head.content })]
              : []),
            ...dropped.map((m) => new RemoveMessage({ id: m.id as string })),
            // NOTE: With no head to keep (every summary came back empty), the survivor has nothing
            // to become, so it is removed like the rest.
            ...(head
              ? []
              : survivorId
                ? [new RemoveMessage({ id: survivorId })]
                : []),
          ],
        },
        THREAD_STATE_NODE,
      );
      return "ok" as const;
    }),
  );
  if (rewrite === "changed") {
    return { outcome: "fail", error: "thread changed during compaction" };
  }

  emitFlowEvent(
    {
      tenantId,
      turnId: crypto.randomUUID(),
      source: "inbox",
      agentId,
      threadId: graphThreadId,
      base,
    },
    {
      stage: "memory",
      level: "info",
      status: "ok",
      detail: {
        attendanceConversationId: conversationId,
        messagesCompacted: cut.closed.length,
        summaryChars: summary.length,
        reason,
      },
    },
  );
  return { outcome: "done" };
}

const compactHandler = async (
  job: { tenantId: bigint; payload: Record<string, unknown> },
  base: PrismaClient,
): Promise<JobResult> => {
  const payload = parsePayload(job.payload);
  // A payload this process cannot read will never become readable, so retrying it only delays the
  // dead-letter. Nothing to compact is not a failure.
  if (!payload) return { outcome: "done" };
  return runCompaction(job.tenantId, payload, base);
};

let registered = false;
export function registerMemoryHandlers(): void {
  if (registered) return;
  registerJobHandler("MEMORY_COMPACT", compactHandler);
  registered = true;
}
