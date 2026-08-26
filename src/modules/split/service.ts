import type { ChatwootClient } from "@/modules/chatwoot/client";
import { type FlowContext, withFlowStage } from "@/modules/flowlog/service";

// Humanized delivery: preserve the reply as one exact public message and pace it with a typing
// indicator + a proportional delay. The legacy pure splitReply helper remains for config/editor
// compatibility, but customer delivery has a single writer and never fragments canonical content.

export interface SplitConfig {
  enabled: boolean;
  // Paragraphs longer than this are further split on sentence boundaries.
  maxChars: number;
  // Words-per-minute used to size the typing delay before each balloon.
  typingWpm: number;
  minDelayMs: number;
  maxDelayMs: number;
  // Safety cap on the number of balloons (extra content is merged into the last).
  maxChunks: number;
}

export const SPLIT_DEFAULTS: SplitConfig = {
  // On by default: replying in a few shorter messages with a brief typing pause reads as more
  // human (n8n parity). The added latency is small and bounded by maxDelayMs; opt-out per agent.
  enabled: true,
  maxChars: 600,
  // ~250 wpm: a brisk, natural typing cadence (the pause stays well under maxDelayMs).
  typingWpm: 250,
  minDelayMs: 800,
  maxDelayMs: 8000,
  maxChunks: 6,
};

function clampInt(
  v: unknown,
  min: number,
  max: number,
  fallback: number,
): number {
  if (typeof v !== "number" || !Number.isFinite(v)) return fallback;
  return Math.min(Math.max(Math.round(v), min), max);
}

export function readSplitConfig(settings: unknown): SplitConfig {
  const s =
    settings && typeof settings === "object"
      ? (settings as Record<string, unknown>).split
      : undefined;
  if (!s || typeof s !== "object") return { ...SPLIT_DEFAULTS };
  const bag = s as Record<string, unknown>;
  const maxChars = clampInt(bag.maxChars, 80, 4000, SPLIT_DEFAULTS.maxChars);
  return {
    enabled:
      typeof bag.enabled === "boolean" ? bag.enabled : SPLIT_DEFAULTS.enabled,
    maxChars,
    typingWpm: clampInt(bag.typingWpm, 40, 1000, SPLIT_DEFAULTS.typingWpm),
    minDelayMs: clampInt(bag.minDelayMs, 0, 10_000, SPLIT_DEFAULTS.minDelayMs),
    maxDelayMs: clampInt(bag.maxDelayMs, 0, 30_000, SPLIT_DEFAULTS.maxDelayMs),
    maxChunks: clampInt(bag.maxChunks, 1, 12, SPLIT_DEFAULTS.maxChunks),
  };
}

// Split into balloons: by paragraph (blank line), then any over-long paragraph by sentence, then cap
// the count (extra balloons merged into the last). Always returns at least one non-empty chunk.
export function splitReply(text: string, cfg: SplitConfig): string[] {
  const trimmed = text.trim();
  if (!trimmed) return [];
  const paragraphs = trimmed
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean);
  const chunks: string[] = [];
  for (const p of paragraphs) {
    if (p.length <= cfg.maxChars) {
      chunks.push(p);
      continue;
    }
    // Over-long paragraph → accumulate sentences up to maxChars.
    let buf = "";
    for (const sentence of p.split(/(?<=[.!?…])\s+/)) {
      const next = buf ? `${buf} ${sentence}` : sentence;
      if (next.length > cfg.maxChars && buf) {
        chunks.push(buf);
        buf = sentence;
      } else {
        buf = next;
      }
    }
    if (buf) chunks.push(buf);
  }
  if (chunks.length === 0) return [trimmed];
  if (chunks.length <= cfg.maxChunks) return chunks;
  // Merge the overflow into the last allowed balloon.
  const head = chunks.slice(0, cfg.maxChunks - 1);
  const tail = chunks.slice(cfg.maxChunks - 1).join("\n\n");
  return [...head, tail];
}

export function typingDelayMs(chunk: string, cfg: SplitConfig): number {
  const words = chunk.split(/\s+/).filter(Boolean).length;
  const ms = (words / cfg.typingWpm) * 60_000;
  return Math.min(Math.max(Math.round(ms), cfg.minDelayMs), cfg.maxDelayMs);
}

const realSleep = (ms: number): Promise<void> =>
  new Promise((r) => setTimeout(r, ms));

// Sends exactly one reply. Typing toggles are best-effort (admin-token, may be unsupported on a
// channel) and never block the send. The sleep is injectable for tests.
export async function deliverReply(
  client: ChatwootClient,
  conversationId: number,
  reply: string,
  cfg: SplitConfig,
  sleep: (ms: number) => Promise<void> = realSleep,
  flow?: FlowContext,
  // Asked before EACH balloon. A split reply is several sends with a typing pause between them, so
  // one answer taken before the loop covers only the first: a run called off after balloon two would
  // keep typing the rest into a conversation the operator was told had been cleared. Returns how
  // many actually landed, so the caller still reports what the customer received.
  calledOff: () => Promise<boolean> = async () => false,
): Promise<number> {
  return withFlowStage(
    flow,
    "split",
    { detail: { enabled: cfg.enabled } },
    async () => {
      if (!cfg.enabled) {
        await client.sendMessage(conversationId, reply);
        return 1;
      }
      await client.toggleTyping(conversationId, true).catch(() => undefined);
      await sleep(typingDelayMs(reply, cfg));
      if (await calledOff()) {
        await client.toggleTyping(conversationId, false).catch(() => undefined);
        return 0;
      }
      await client.sendMessage(conversationId, reply);
      await client.toggleTyping(conversationId, false).catch(() => undefined);
      return 1;
    },
  );
}
