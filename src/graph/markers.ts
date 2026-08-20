import { type BaseMessage, HumanMessage } from "@langchain/core/messages";

// The system markers that ride INSIDE messages of the graph memory thread.
//
// They live in a near-leaf module — messages only, no Prisma, no tenancy, no checkpointer — because
// the code that DECIDES where an attendance begins and ends (src/modules/memory/cut.ts) is a pure
// function over an array of messages, and it must stay that way.
//
// Why the markers are messages and not a SystemMessage: the agent node drops every system message
// from the history before the model call (src/graph/graph.ts), because a second system message is
// rejected outright by some providers. A system-role marker would therefore be invisible at exactly
// the moment it matters.
//
// RECOGNIZED BY METADATA, WRITTEN ONLY HERE. The marker text still travels in the content, because
// that is what the model reads, but nothing decides anything from that text. A customer whose message
// happens to start with one of these tags would otherwise be read as a system marker — and this repo
// is public, so "happens to" includes "chose to". The sharp end is the memory head: a message taken
// for the head is excluded from the summary and then REPLACED by the rendered head, so a customer's
// words would be deleted without ever having been summarized. Metadata cannot be typed into a chat.

const MARKER_KWARG = "fazerMarker";
type SystemMarker = "divider" | "memory_head";

function hasMarker(message: BaseMessage, marker: SystemMarker): boolean {
  return message.additional_kwargs?.[MARKER_KWARG] === marker;
}

// Folded into the first human turn of a NEW conversation when the contact-inbox thread already
// carries memory from a prior one. Written by both the reactive turn (src/graph/runtime.ts) and the
// silent-message ingestion (src/graph/ingest.ts) — the first as its own message, the second prepended
// to the customer's text, which is why the factory takes the trailing text.
export const CONVERSATION_DIVIDER =
  "(Contexto do sistema: início de uma nova conversa com este mesmo contato. As mensagens anteriores são de atendimentos passados; não presuma que o assunto continua, trate isto como um novo atendimento.)";

// The compacted memory of already-closed attendances, rendered from attendance_summaries and kept as
// the FIRST message of the thread. Recognizing it matters as much as writing it: the head is rebuilt
// from the rows on every compaction, so it must never be fed back to the summarizer — that is the
// difference between summarizing each attendance once and re-summarizing a summary forever.
export const MEMORY_HEAD_OPEN = "<atendimentos-anteriores>";
export const MEMORY_HEAD_CLOSE = "</atendimentos-anteriores>";

export function conversationDividerMessage(
  trailingText?: string,
): HumanMessage {
  return new HumanMessage({
    content: trailingText
      ? `${CONVERSATION_DIVIDER}\n\n${trailingText}`
      : CONVERSATION_DIVIDER,
    additional_kwargs: { [MARKER_KWARG]: "divider" satisfies SystemMarker },
  });
}

// `id` reuses the id of the message the head replaces, which is what keeps it at the front of the
// channel (the reducer replaces a same-id message in place and appends an unknown-id one at the end).
export function memoryHeadMessage(content: string, id?: string): HumanMessage {
  return new HumanMessage({
    ...(id ? { id } : {}),
    content,
    additional_kwargs: { [MARKER_KWARG]: "memory_head" satisfies SystemMarker },
  });
}

export function isConversationDivider(message: BaseMessage): boolean {
  return hasMarker(message, "divider");
}

export function isMemoryHead(message: BaseMessage): boolean {
  return hasMarker(message, "memory_head");
}
