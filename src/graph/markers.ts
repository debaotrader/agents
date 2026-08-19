// The system markers that ride INSIDE messages of the graph memory thread.
//
// They live in a leaf module, with no imports, for one reason: the code that DECIDES where an
// attendance begins and ends (src/modules/memory/cut.ts) is a pure function, and pulling these
// strings out of the ingestion module would drag Prisma, the tenancy layer and the checkpointer into
// something that only reads an array of messages.
//
// Why the markers are text inside a message and not a SystemMessage: the agent node drops every
// system message from the history before the model call (src/graph/graph.ts), because a second
// system message is rejected outright by some providers. A system-role marker would therefore be
// invisible at exactly the moment it matters.

// Folded into the first human turn of a NEW conversation when the contact-inbox thread already
// carries memory from a prior one. Written by both the reactive turn (src/graph/runtime.ts) and the
// silent-message ingestion (src/graph/ingest.ts), byte-identical, because the compaction cut finds
// the boundary by matching this exact prefix.
export const CONVERSATION_DIVIDER =
  "(Contexto do sistema: início de uma nova conversa com este mesmo contato. As mensagens anteriores são de atendimentos passados; não presuma que o assunto continua, trate isto como um novo atendimento.)";

// The compacted memory of already-closed attendances, rendered from attendance_summaries and kept as
// the FIRST message of the thread. Recognizing it matters as much as writing it: the head is rebuilt
// from the rows on every compaction, so it must never be fed back to the summarizer — that is the
// difference between summarizing each attendance once and re-summarizing a summary forever.
export const MEMORY_HEAD_OPEN = "<atendimentos-anteriores>";
export const MEMORY_HEAD_CLOSE = "</atendimentos-anteriores>";

export function isConversationDivider(text: string): boolean {
  return text.startsWith(CONVERSATION_DIVIDER);
}

export function isMemoryHead(text: string): boolean {
  return text.startsWith(MEMORY_HEAD_OPEN);
}
