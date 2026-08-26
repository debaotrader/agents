import { describe, expect, test } from "bun:test";
import {
  buildQuoteResolver,
  classifyLastPublicMessage,
  lastPublicMessageIsFromAgentBot,
  parseChatwootMessages,
  parseChatwootMessagesPage,
  pendingIncoming,
  toRenderable,
} from "@/modules/chatwoot/messages";
import { renderInboundMessage } from "@/modules/chatwoot/render";

describe("parseChatwootMessages", () => {
  test("parses { payload } with integer message_type, sorted by id", () => {
    const rows = parseChatwootMessages({
      payload: [
        { id: 2, content: "b", message_type: 1, private: false },
        { id: 1, content: "a", message_type: 0, private: false },
      ],
    });
    expect(rows.map((r) => r.id)).toEqual([1, 2]);
    expect(rows[0]).toEqual({
      id: 1,
      content: "a",
      messageType: "incoming",
      private: false,
      attachmentTypes: [],
      transcribedText: null,
      imageDescription: null,
      extractedText: null,
      attachmentName: null,
      location: null,
      inReplyTo: null,
      isReaction: false,
      senderId: null,
      senderType: null,
    });
    expect(rows[1]?.messageType).toBe("outgoing");
  });

  test("parses sender identity used by proactive-message eligibility", () => {
    const [row] = parseChatwootMessages({
      payload: [
        {
          id: 9,
          content: "Oi!",
          message_type: 1,
          sender: { id: 5, type: "agent_bot" },
        },
      ],
    });
    expect(row?.senderId).toBe(5);
    expect(row?.senderType).toBe("agent_bot");
  });

  test("accepts a bare array and tolerates the webhook string form", () => {
    const rows = parseChatwootMessages([
      { id: 5, content: "x", message_type: "incoming" },
    ]);
    expect(rows[0]?.messageType).toBe("incoming");
  });

  test("drops items without a numeric id", () => {
    const rows = parseChatwootMessages({
      payload: [
        { content: "no id" },
        { id: 3, content: "ok", message_type: 0 },
      ],
    });
    expect(rows.map((r) => r.id)).toEqual([3]);
  });

  test("maps activity/template/unknown types to a non-incoming bucket", () => {
    const rows = parseChatwootMessages({
      payload: [
        { id: 1, content: "a", message_type: 2 },
        { id: 2, content: "b", message_type: 3 },
        { id: 3, content: "c", message_type: 99 },
      ],
    });
    expect(rows.map((r) => r.messageType)).toEqual([
      "activity",
      "template",
      "other",
    ]);
  });

  test("extracts attachment types, transcribed_text meta, and in_reply_to", () => {
    const rows = parseChatwootMessages({
      payload: [
        {
          id: 10,
          content: "",
          message_type: 0,
          attachments: [
            {
              id: 1,
              file_type: "audio",
              meta: { transcribed_text: "oi tudo bem" },
            },
          ],
          content_attributes: { in_reply_to: 7 },
        },
      ],
    });
    expect(rows[0]?.attachmentTypes).toEqual(["audio"]);
    expect(rows[0]?.transcribedText).toBe("oi tudo bem");
    expect(rows[0]?.inReplyTo).toBe(7);
  });

  // NOTE: Issue #45 — the debounce re-fetch path must carry the pin the same way the direct path
  // does, and the maps-URL basename ("maps") must stop leaking as a fake file name.
  test("location attachment: coordinates ride the REST row into the renderable", () => {
    const rows = parseChatwootMessages({
      payload: [
        {
          id: 11,
          content: "",
          message_type: 0,
          attachments: [
            {
              id: 2,
              file_type: "location",
              coordinates_lat: -23.5505,
              coordinates_long: -46.6333,
              fallback_title: "Padaria do Zé",
              data_url: "https://maps.google.com/maps?q=-23.5505,-46.6333",
            },
          ],
        },
      ],
    });
    const row = rows[0];
    expect(row).toBeDefined();
    if (!row) return;
    const renderable = toRenderable(row);
    expect(renderable.location).toEqual({
      latitude: -23.5505,
      longitude: -46.6333,
      title: "Padaria do Zé",
    });
    const out = renderInboundMessage(renderable);
    expect(out).toBe(
      '<localização latitude="-23.5505" longitude="-46.6333" titulo="Padaria do Zé">',
    );
  });
});

describe("lastPublicMessageIsFromAgentBot", () => {
  test("allows only when the last relevant public message is from our bot", () => {
    const messages = parseChatwootMessages({
      payload: [
        {
          id: 1,
          content: "Oi!",
          message_type: 1,
          sender: { id: 5, type: "agent_bot" },
        },
        {
          id: 2,
          content: "activity",
          message_type: 2,
          sender: { id: 7, type: "user" },
        },
        {
          id: 3,
          content: "nota",
          message_type: 1,
          private: true,
          sender: { id: 7, type: "user" },
        },
      ],
    });

    expect(lastPublicMessageIsFromAgentBot(messages, 5)).toBe(true);
  });

  test("blocks the incident shape: contact emoji followed by a human reaction", () => {
    const messages = parseChatwootMessages({
      payload: [
        {
          id: 10,
          content: "Tudo certo por aqui.",
          message_type: 1,
          sender: { id: 5, type: "agent_bot" },
        },
        {
          id: 11,
          content: "🫡",
          message_type: 0,
          sender: { id: 99, type: "contact" },
        },
        {
          id: 12,
          content: "👊",
          message_type: 1,
          content_attributes: { is_reaction: true },
          sender: { id: 7, type: "user" },
        },
      ],
    });

    expect(lastPublicMessageIsFromAgentBot(messages, 5)).toBe(false);
  });

  test("fails closed for missing or malformed sender identity", () => {
    const messages = parseChatwootMessages({
      payload: [{ id: 1, content: "Oi!", message_type: 1 }],
    });
    expect(lastPublicMessageIsFromAgentBot(messages, 5)).toBe(false);
    expect(lastPublicMessageIsFromAgentBot([], 5)).toBe(false);
  });
});

describe("follow-up message-page attribution", () => {
  test("rejects a malformed page instead of treating it as an empty history", () => {
    expect(parseChatwootMessagesPage({ payload: "not-an-array" })).toBeNull();
    expect(
      parseChatwootMessagesPage({ payload: [{ content: "missing id" }] }),
    ).toBeNull();
  });

  test("classifies exact bot ownership and preserves the observed message id", () => {
    const page = parseChatwootMessagesPage({
      payload: [
        {
          id: 42,
          content: "Oi!",
          message_type: 1,
          sender: { id: 5, type: "agent_bot" },
        },
      ],
    });
    expect(page).not.toBeNull();
    expect(classifyLastPublicMessage(page?.messages ?? [], 5)).toEqual({
      kind: "ours",
      messageId: 42,
    });
  });

  test("treats an outgoing row without sender identity as uncertain", () => {
    const page = parseChatwootMessagesPage({
      payload: [{ id: 43, content: "Oi!", message_type: 1 }],
    });
    expect(classifyLastPublicMessage(page?.messages ?? [], 5)).toEqual({
      kind: "unknown",
      messageId: 43,
    });
  });

  test("a public reaction is definitively non-bot even without sender identity", () => {
    const page = parseChatwootMessagesPage({
      payload: [
        {
          id: 44,
          content: "👊",
          message_type: 1,
          content_attributes: { is_reaction: true },
        },
      ],
    });
    expect(classifyLastPublicMessage(page?.messages ?? [], 5)).toEqual({
      kind: "other",
      messageId: 44,
    });
  });
});

describe("pendingIncoming", () => {
  const msgs = parseChatwootMessages({
    payload: [
      { id: 1, content: "oi", message_type: 0, private: false },
      { id: 2, content: "tudo bem?", message_type: 0, private: false },
      { id: 3, content: "(nota privada)", message_type: 0, private: true },
      { id: 4, content: "resposta", message_type: 1, private: false },
      { id: 5, content: "   ", message_type: 0, private: false },
    ],
  });

  test("watermark null → incoming, non-private, non-empty only", () => {
    expect(pendingIncoming(msgs, null).map((m) => m.id)).toEqual([1, 2]);
  });

  test("watermark excludes already-handled ids", () => {
    expect(pendingIncoming(msgs, 1).map((m) => m.id)).toEqual([2]);
    expect(pendingIncoming(msgs, 2).map((m) => m.id)).toEqual([]);
  });

  test("includes an incoming voice note (empty content, has attachment)", () => {
    const withAudio = parseChatwootMessages({
      payload: [
        {
          id: 1,
          content: "",
          message_type: 0,
          attachments: [{ id: 9, file_type: "audio" }],
        },
      ],
    });
    expect(pendingIncoming(withAudio, null).map((m) => m.id)).toEqual([1]);
  });
});

describe("buildQuoteResolver", () => {
  const msgs = parseChatwootMessages({
    payload: [
      { id: 10, content: "Qual o horário?", message_type: 0 },
      {
        id: 11,
        content: "",
        message_type: 0,
        attachments: [
          {
            id: 1,
            file_type: "audio",
            meta: { transcribed_text: "ouça isto" },
          },
        ],
      },
      { id: 12, content: "   ", message_type: 0 },
    ],
  });

  test("resolves a quoted message's text by id (content or transcription)", () => {
    const resolve = buildQuoteResolver(msgs);
    expect(resolve(10)).toBe("Qual o horário?");
    // Voice note: falls back to the written-back transcription.
    expect(resolve(11)).toBe("ouça isto");
    // Whitespace-only / unknown ids resolve to null.
    expect(resolve(12)).toBeNull();
    expect(resolve(999)).toBeNull();
  });
});
