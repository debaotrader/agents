-- The customer-facing out-of-hours message (issue #153) gets its own watermark instead of sharing the
-- operator note's. The two run on different clocks (note: once per conversation; message: once per
-- local day) and, more to the point, a conversation whose note went out earlier today must still
-- receive the message the first time an operator writes one -- which a shared column cannot express,
-- because it does not record WHICH notice it stamped.
--
-- Additive, nullable, no backfill: NULL reads as "never sent", which is the correct state for every
-- existing row (the feature did not exist).
ALTER TABLE "conversations" ADD COLUMN "away_message_sent_at" TIMESTAMP(3);
