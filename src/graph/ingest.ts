import { AIMessage, HumanMessage } from "@langchain/core/messages";
import type { BaseCheckpointSaver } from "@langchain/langgraph-checkpoint";
import type { PrismaClient } from "@/../generated/prisma/client";
import basePrisma from "@/api/lib/prisma";
import { withEntityLock } from "@/lib/locks";
import { runScopedOn, type TenantContext } from "@/lib/tenancy";
import { getCheckpointer } from "./checkpointer";
import { isTurnInFlight } from "./inflight";
import { conversationDividerMessage, conversationStamp } from "./markers";
import { buildThreadStateGraph, THREAD_STATE_NODE } from "./thread-state";

// Continuous ingestion: fold a conversation message into the agent's graph memory thread WITHOUT
// running a model, so the agent has full context even for messages no turn handled — a customer
// message it stayed silent on (out of hours, a human handling, test/disabled aside), or a HUMAN
// agent's reply sent while it was silent. The seam is graph.updateState, which appends to the
// thread's MessagesAnnotation channel via the same reducer the real turn uses.
//
// At-most-once: the delivery ledger dedups re-deliveries, message_created gating ignores edits, and a
// monotonic per-thread watermark (AgentThread.lastSyncedMessageId, CAS under a per-thread advisory
// lock) is defense-in-depth against a re-delivery that slips a new delivery UUID. The lock also
// serializes concurrent ingestions on one thread so two appends can't clobber each other's checkpoint.

function sysCtx(tenantId: bigint): TenantContext {
  return { tenantId, userId: null, role: "TENANT_ADMIN" };
}

export type IngestRole = "customer" | "human_agent";

export interface IngestMessageParams {
  tenantId: bigint;
  instanceId: bigint;
  // Chatwoot display_id — only used for the per-thread "new conversation" divider marker.
  conversationId: number;
  // The native ContactInbox id: the AgentThread key (== the graph thread's discriminator).
  contactInboxId: number;
  // The graph memory thread to append to (tenant:instance:ci:<contactInboxId>).
  graphThreadId: string;
  // Chatwoot message id — the monotonic watermark guarding against re-append.
  messageId: number;
  role: IngestRole;
  // The message body: a rendered customer message (renderInboundMessage) or a human agent's raw text.
  text: string;
  // The human agent's display name, for the <atendente> marker (role "human_agent" only).
  agentName?: string | null;
  base?: PrismaClient;
  checkpointer?: BaseCheckpointSaver;
  // Fired when this message OPENED a new attendance on the thread, carrying the display_id of the
  // one that just ended. A callback rather than a direct call because the work it triggers (arming
  // memory compaction) opens its own transaction, and this one runs under an advisory lock — so it
  // is invoked only after the lock is released.
  onAttendanceClosed?: (previousConversationId: number) => Promise<void> | void;
}

// Strip quotes/newlines from a human agent's display name so it can't break the <atendente nome="…">
// marker. The name is our own staff (low risk), but keep the marker well-formed.
function sanitizeName(name: string): string {
  return name
    .replace(/["\n\r]+/g, " ")
    .trim()
    .slice(0, 80);
}

export async function ingestMessageIntoThread(
  params: IngestMessageParams,
): Promise<"ingested" | "skipped"> {
  const base = params.base ?? basePrisma;
  const {
    tenantId,
    instanceId,
    conversationId,
    contactInboxId,
    graphThreadId,
    messageId,
    role,
  } = params;
  if (!params.text.trim()) return "skipped";
  const checkpointer = params.checkpointer ?? (await getCheckpointer());
  const graph = buildThreadStateGraph(checkpointer);

  const done = await runScopedOn(base, sysCtx(tenantId), (db) =>
    withEntityLock(db, `ingest:${graphThreadId}`, async () => {
      const key = {
        tenantId_chatwootInstanceId_contactInboxId: {
          tenantId,
          chatwootInstanceId: instanceId,
          contactInboxId,
        },
      };
      const row = await db.agentThread.findUnique({
        where: key,
        select: { lastSyncedMessageId: true, lastConversationId: true },
      });
      // Monotonic watermark: never re-append a message already folded into the thread.
      if (
        row?.lastSyncedMessageId != null &&
        messageId <= row.lastSyncedMessageId
      ) {
        return { outcome: "skipped" as const, closedConversationId: null };
      }

      // A message that starts a NEW conversation on this thread gets the fresh-attendance divider
      // (same as the reactive turn). Human-agent messages count: an agent who opens the conversation
      // sends the first message of it, and skipping them here left that message sitting inside the
      // PREVIOUS attendance — summarized and removed with it when the customer finally replied.
      const prevConv = row?.lastConversationId ?? null;
      const boundary = prevConv != null && prevConv !== conversationId;
      // An invoke that started earlier on this thread is a read-modify-write of the WHOLE channel:
      // it saves the state it loaded plus its own messages, erasing whatever landed meanwhile. So a
      // boundary crossed while one is reading cannot be CONSUMED here — the divider would be erased
      // and the marker below would advance for good, and the messages of the attendance that just
      // ended would then carry no boundary the cut can find, leaving the due job a no-op that no
      // later message re-arms. The marker stays put instead, and the next message on this thread
      // writes the divider with no invoke in the way. Same reasoning as the reactive turn
      // (src/graph/runtime.ts), and the same conclusion: only the PROMPT is at stake, since the cut
      // reads the conversation stamped on each message, not the divider.
      const anotherInvokeIsReading = boundary && isTurnInFlight(graphThreadId);
      const newConversation = boundary && !anotherInvokeIsReading;

      // Customer → HumanMessage; human agent → AIMessage (the business side of the dialogue, so the
      // model reads it as an already-given reply), disambiguated by the <atendente> marker so it never
      // mistakes a colleague's words for its OWN prior output. Every one of them carries the
      // conversation it belongs to, which is what the compaction cut reads; the divider on top is
      // prompt content, and goes through the factory because nothing else can make a message COUNT as
      // one — the text alone never does, or a customer could type it (src/graph/markers.ts).
      const stamp = conversationStamp(conversationId);
      const msg =
        role === "human_agent"
          ? new AIMessage({
              content: `<atendente${params.agentName ? ` nome="${sanitizeName(params.agentName)}"` : ""}>\n${params.text}\n</atendente>`,
              additional_kwargs: stamp,
            })
          : newConversation
            ? conversationDividerMessage(conversationId, params.text)
            : new HumanMessage({
                content: params.text,
                additional_kwargs: stamp,
              });

      await graph.updateState(
        { configurable: { thread_id: graphThreadId } },
        {
          messages:
            // The human agent's own message is an AIMessage, so the divider cannot ride inside it the
            // way it does on a customer turn: it goes ahead of it, as its own message.
            newConversation && role === "human_agent"
              ? [conversationDividerMessage(conversationId), msg]
              : [msg],
        },
        THREAD_STATE_NODE,
      );

      // Advance the watermark; customer messages also advance the divider marker (turns do the same).
      await db.agentThread.upsert({
        where: key,
        create: {
          tenantId,
          chatwootInstanceId: instanceId,
          contactInboxId,
          threadId: graphThreadId,
          lastSyncedMessageId: messageId,
          lastConversationId: conversationId,
        },
        update: {
          lastSyncedMessageId: messageId,
          // Held back when an invoke owns the thread: see anotherInvokeIsReading above. The synced
          // watermark still advances — it guards at-most-once append, and rewinding it would trade a
          // lost divider for a duplicated message.
          lastConversationId: anotherInvokeIsReading
            ? prevConv
            : conversationId,
        },
      });
      return {
        outcome: "ingested" as const,
        // Armed even when the boundary was not consumed. The attendance that just ended is
        // compactable right now — its boundary lives on the messages, not on the divider this call
        // declined to write — and withholding the arm would make it wait on a next message that may
        // never come.
        closedConversationId: boundary ? prevConv : null,
      };
    }),
  );

  if (done.closedConversationId !== null) {
    await params.onAttendanceClosed?.(done.closedConversationId);
  }
  return done.outcome;
}
