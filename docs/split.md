# Single-message + typing delivery

Every reactive turn has exactly one customer-facing text owner. Text is delivered byte-for-byte in one public Chatwoot message, optionally preceded by a proportional typing delay. This preserves canonical blocks and prevents a single model turn from becoming a burst of public balloons. Audio replies remain one voice note.

## Module (`src/modules/split/service.ts`)

- `splitReply(text, cfg)` — legacy pure helper retained for settings/editor compatibility. The delivery path does not call it.
- `typingDelayMs(text, cfg)` — `words / typingWpm × 60s`, clamped to `[minDelayMs, maxDelayMs]`.
- `deliverReply(client, conversationId, reply, cfg, sleep?)` — disabled → one exact `sendMessage`; enabled → `toggleTyping(on)` → `sleep(delay)` → one exact `sendMessage` → `toggleTyping(off)`. Typing toggles are **best-effort** (admin token, `.catch` swallows failures). `sleep` is injectable.

`client.toggleTyping(id, on)` = `POST …/conversations/{id}/toggle_typing_status { typing_status }` (admin token — not in the bot allowlist). The runtime threads an injectable `sleep` via `RuntimeDeps`.

## Configuration

Per-agent `agent.settings.split` (`readSplitConfig`): `enabled` (default `true`), `maxChars` (default 600), `typingWpm` (250), `minDelayMs` (800), `maxDelayMs` (8000), `maxChunks` (6). The split sizing fields remain accepted for backwards compatibility; reactive customer delivery never fragments the reply. Writable over REST (`PATCH /v1/agents/:id`) + MCP (`agent_settings_get`/`agent_settings_set`, the `split` block).

## Interaction notes

- `runLoadedTurn` wraps the Chatwoot client in `TurnControl`; a public human intervention aborts generation and fences the pending send before it can start.
- Holds the processing/job a bit longer (within the scheduler reaper window). Typical replies finish well under the stale threshold.

Read before touching `src/modules/split/*`, `client.toggleTyping`, or the text-delivery branch in `runLoadedTurn`.
