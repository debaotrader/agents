import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  type BaseMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import logger from "@/api/lib/logger";
import { contentToText } from "@/graph/message-text";
import { runModelCall } from "@/graph/model-limit";

// Condenses the raw turns of a closed attendance into the memory the agent keeps of it.
//
// Shaped after src/modules/guardrails/analyze.ts, and for the same reason: this is a model call that
// happens outside a turn, so it needs its own timeout and an explicit "could not be produced" state.
// A summary that came back empty and a summary that never ran are the same value without it, and
// they call for opposite actions — the first means the attendance had nothing worth remembering, the
// second means the thread must be left exactly as it is and retried.
//
// It is not on any customer's critical path: the job runs after the reply was posted.

const SUMMARIZE_TIMEOUT_MS = 60_000;

// The summary is prepended to every future turn of this contact, once per closed attendance, so its
// size is a recurring cost. Long enough for what was agreed, short enough that twenty of them do not
// become the context problem they were built to solve.
export const ATTENDANCE_SUMMARY_MAX = 1200;

// How much raw transcript is handed to the summarizer. A thread that accumulated many attendances
// (compaction newly enabled, or a run that kept failing) can be larger than the model's own window,
// and a call that fails on size would never recover on retry. Clipped from the FRONT, keeping the
// most recent turns, because that is the half a later attendance is most likely to refer back to.
const TRANSCRIPT_MAX_CHARS = 60_000;

export const TRANSCRIPT_TAG = "<transcricao>";
const TRANSCRIPT_CLOSE = "</transcricao>";

// Anything in the transcript that reads as the fence's own tag, in every spelling it could take. The
// text inside is written by the customer, who would otherwise be able to close the fence and address
// the summarizer directly — and what the summarizer writes is what the agent believes forever after.
const FENCE_TAG = /<\s*\/?\s*transcricao[^>]*>/gi;

// Escolhido por medição, não por gosto: bateria A/B com n=24 a 32 por célula em gpt-5.4-mini, sobre
// dois diálogos (um simples; um com o valor mudando no meio, uma restrição dita uma única vez logo no
// começo e nenhum fechamento). O que a bateria comparou e o que ela achou:
//
//   retenção factual  este prompt acerta todos os fatos em 32/32 no diálogo simples e 31/32 no
//                     difícil (a falha foi omitir o NOME do cliente). Invenção: 0 em toda célula.
//   variante A        a mesma coisa com quatro bullets enumerando o que preservar: mesma retenção,
//                     mesma ausência de invenção, resumo 26% mais longo no diálogo difícil, e
//                     escrita vazada em 8/64 contra 4/128 deste (p≈0,01).
//   variante C        este mais "usando o mesmo alfabeto dela do começo ao fim", tentando matar o
//                     vazamento: não mexeu no vazamento (1/32 contra 2/32) e DERRUBOU a retenção do
//                     nome para 27/32 no diálogo difícil. Rejeitada.
//
// "Escrita vazada" é um pedaço em cirílico/persa/bengali no meio de uma frase em português ("com
// обещa de retorno", "com মূল্য de R$ 250,00"): ~3% dos resumos, artefato do modelo e não do texto
// acima, já que a única tentativa direta de corrigi-lo custou retenção. O sentido sobrevive, mas
// aquilo fica gravado e reaparece em todo turno seguinte, então está registrado aqui como conhecido
// e medido em vez de descoberto por um operador. Não há pós-processamento tirando caractere
// não-latino: a regra do idioma é deliberada, e um cliente que fala russo tem que receber memória em
// cirílico.
const SYSTEM_PROMPT = `Você registra a memória de um atendimento que acabou, para o atendente que vai falar com este mesmo cliente da próxima vez.

Escreva um resumo curto do atendimento entre as tags de transcrição, guardando o que um próximo atendimento precisaria saber.

Regras:
- Escreva no mesmo idioma da conversa.
- Só registre o que está na transcrição. Não deduza, não complete e não invente nada.
- Se algo ficou ambíguo, diga que ficou ambíguo em vez de escolher uma versão.
- Não escreva saudações, não se dirija ao cliente e não faça perguntas.
- Responda apenas com o resumo, sem preâmbulo e sem formatação de título.`;

export interface AttendanceSummaryResult {
  // The summary text, already clipped. Empty when nothing was produced.
  summary: string;
  // Set when the summary could not be produced at all (model error, timeout, empty completion). The
  // caller must leave the thread untouched and let the job retry.
  error?: string;
}

// One line per message, in order. Tool CALLS travel as the tool's name and tool RESULTS do not
// travel at all: their payloads are the heaviest part of a tool-driven thread and the least
// summarizable, and whatever the agent actually did with a result it said out loud in the reply that
// follows. Sending them would spend the summarizer's window on ids and ISO timestamps, and would
// hand a second model call customer data that never reached the customer.
export function renderTranscript(messages: BaseMessage[]): string {
  const lines: string[] = [];
  for (const m of messages) {
    const type = m.getType();
    if (type === "tool") continue;
    const text = contentToText(m.content).trim();
    if (type === "human") {
      if (text) lines.push(`cliente: ${text}`);
      continue;
    }
    if (text) lines.push(`atendente: ${text}`);
    const calls = (m as { tool_calls?: { name?: unknown }[] }).tool_calls;
    if (Array.isArray(calls) && calls.length > 0) {
      const names = calls
        .map((c) => (typeof c?.name === "string" ? c.name : "?"))
        .join(", ");
      lines.push(`atendente [usou ferramenta: ${names}]`);
    }
  }
  const joined = lines.join("\n").replace(FENCE_TAG, "");
  return joined.length > TRANSCRIPT_MAX_CHARS
    ? joined.slice(joined.length - TRANSCRIPT_MAX_CHARS)
    : joined;
}

export async function summarizeAttendance(
  model: BaseChatModel,
  messages: BaseMessage[],
): Promise<AttendanceSummaryResult> {
  const transcript = renderTranscript(messages);
  // NOTE: An attendance whose messages carry no text at all (only tool traffic) has nothing to
  // remember. That is a legitimate empty summary, not a failure, so it must not carry `error`.
  if (!transcript.trim()) return { summary: "" };

  try {
    const res = await runModelCall(() =>
      model.invoke(
        [
          new SystemMessage(SYSTEM_PROMPT),
          // NOTE: The transcript is never interpolated into the system prompt. Everything in a
          // system message reads to the model as an instruction from the operator, and this text was
          // written by the customer.
          new HumanMessage(
            `${TRANSCRIPT_TAG}\n${transcript}\n${TRANSCRIPT_CLOSE}`,
          ),
        ],
        { signal: AbortSignal.timeout(SUMMARIZE_TIMEOUT_MS) },
      ),
    );
    const text = contentToText(res.content).trim();
    if (!text) return { summary: "", error: "empty completion" };
    return { summary: text.slice(0, ATTENDANCE_SUMMARY_MAX) };
  } catch (err) {
    logger.warn({ err }, "memory: attendance summary failed, thread untouched");
    return {
      summary: "",
      error: err instanceof Error ? err.message : String(err),
    };
  }
}
