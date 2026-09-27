# Vera Challenge Bot

A candidate submission for the magicpin AI Challenge (`docs_reference/challenge-brief.md` +
`challenge-testing-brief.md`). Implements the full 5-endpoint HTTP contract, a
4-context LLM composer, and multi-turn conversation handling (auto-reply
detection, intent-transition, hostile/off-topic handling, graceful exit).

## Repo layout

```
bot.py                      # FastAPI app: the 5 HTTP endpoints + plain compose()
composer.py                 # category+merchant+trigger+customer -> message (Gemini or Anthropic)
conversation_handlers.py    # multi-turn respond(): auto-reply/intent/hostile logic
generate_submission.py      # builds submission.jsonl from dataset/test_pairs.json
requirements.txt
render.yaml                 # Render Blueprint (one-click deploy config)
.github/workflows/keepalive.yml   # cron that pings /v1/healthz to prevent sleep
dataset/                    # generated base dataset (categories/merchants/customers/triggers)
  seeds/                    # the original *_seed.json files, kept for provenance
  test_pairs.json           # the 30 canonical (merchant, trigger) pairs
generate_dataset.py         # deterministic expander: seeds -> full 5/50/200/100 dataset
submission.jsonl            # 30 composed messages, one per test pair
judge_simulator.py          # magicpin's local test harness (unmodified)
docs_reference/             # the challenge briefs, verbatim, for anyone opening this repo cold
```

## LLM provider — Gemini by default

`composer.py` supports three providers, switched with `LLM_PROVIDER`:

| Provider | Env vars | Notes |
|---|---|---|
| **Groq** (default) | `GROQ_API_KEY`, optional `GROQ_MODEL` (default `openai/gpt-oss-120b`) | Free, no credit card, sign up at [console.groq.com/keys](https://console.groq.com/keys). A much higher free daily request cap than Gemini's free tier — the reliable default for generating all 30 submissions or running a full judge session without hitting a quota wall. Groq's model lineup changes over time
| Gemini | `LLM_PROVIDER=gemini`, `GEMINI_API_KEY`, optional `GEMINI_MODEL` (default `gemini-3.8-flash`) | Free tier at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey). **Caveat:** the free tier caps some models at just 20 requests/day **per Google Cloud project** (not per key — a new key on the same project does not reset it)
| Anthropic | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` | Needs a funded Anthropic Console account |

## Approach

**One composer, dispatched by `trigger.kind`.** Rather than a different prompt
per category (5x duplication) or per trigger kind (20x duplication),
`composer.py` builds a single structured prompt containing all four contexts
plus a short per-`kind` "framing hint" (`KIND_FRAMING` dict) that tells the
model how to angle the message (loss-aversion for `perf_dip`, curiosity for
`research_digest`, action-mode for `active_planning_intent`, etc.). This
keeps category voice, merchant state, and trigger urgency all visible to the
model at once, so it can trade off between them instead of composing in
isolation.

**Constraints enforced in code, not just in the prompt.** The rubric and
testing brief both call out hard failure modes (URL in body → -3, malformed
JSON → -2, verbatim repeat → -2). Prompting an LLM not to do these things is
necessary but not sufficient, so:
- URLs are regex-stripped from the body post-hoc (`_strip_urls`).
- Anti-repetition is checked against `prior_bodies` before returning.
- Suppression-key dedup and trigger-expiry checks happen in `bot.py` before
  the composer is even called, independent of what the LLM decides.
- The composer's output is parsed against a strict `BODY:/CTA:/RATIONALE:`
  format, with an enum-validated CTA, so malformed LLM output degrades to a
  safe default rather than propagating an invalid shape.

**Deterministic and fail-safe.** `temperature=0` for the LLM call. If no key
is set for the configured provider, or the API call throws/times out, a
rule-based fallback (`_fallback_compose`) still produces a spec-shaped
message anchored on the most relevant real fact in the contexts (the top
digest item's title + source + actionable line, or the merchant's real perf
numbers) — never a generic "grow your business" filler, and never a
fabricated fact. This means the bot degrades gracefully instead of returning
empty/malformed actions if the LLM provider has an outage mid-test.

**Multi-turn handling as pattern-matched dispatch, not another LLM call.**
`conversation_handlers.py` checks, in order: hostile → not-interested →
auto-reply (exact-repeat OR canned-phrase heuristic, escalating
send→wait→end over 3 occurrences, matching the brief's worked example
verbatim) → explicit intent-commit ("let's do it", "yes send it", etc., which
skips straight to an action-mode reply instead of re-qualifying) → off-topic
redirect → otherwise, falls through to the same `composer.compose()` used for
first touches, with the merchant's latest message folded into the trigger
payload as `merchant_last_message` so the model can continue the thread
naturally. This keeps the "boring but important" conversational rules fast
and deterministic, and reserves the LLM call for the parts that actually need
open-ended composition.

## Tradeoffs

- **No persistent store.** Everything (`CONTEXTS`, `CONVERSATIONS`,
  suppression state) lives in process memory, per the spec ("storing in
  memory is fine; just don't restart between calls"). This is intentionally
  simple for a 60-minute test window; a real deployment would put this in
  Redis (as production Vera already does per `docs_reference/engagement-research.md`).
- **Suppression cooldown is a flat 6h window**, not full state-machine
  tracking of each trigger's own `expires_at`. Simpler and safe against spam,
  but a more sophisticated version would let high-urgency triggers (e.g.
  `supply_alert`, urgency=5) override the cooldown.
- **The rule-based fallback is deliberately plain**, not LLM-quality. It
  exists purely so the bot never returns malformed/empty output under an LLM
  outage — it will score far lower on Specificity/Engagement-compulsion than
  the LLM path.
- **Auto-reply detection is heuristic (regex + verbatim-repeat), not a
  classifier**, matching the brief's own stated heuristic ("same message
  verbatim 3+ times = auto-reply") plus common WhatsApp Business canned
  phrases. It will miss auto-replies phrased outside that list.
- **No retrieval/embeddings** over the digest items — with only ~5 digest
  items per category, the composer just includes all of them (trimmed) in
  the prompt. This is the right tradeoff at this data scale, not at
  production scale.

## What additional context would have helped most

1. **A real `merchant.identity.languages` → code-mix ratio guide** — the
   dataset says `["en","hi"]` but not *how much* Hindi vs English a given
   merchant actually prefers.
2. **A canonical list of taboo *phrases* per category**, not just taboo
   *words* — real anti-patterns are often phrase-level.
3. **Ground-truth "good message" examples per trigger *kind*** — the brief
   gives 2 worked examples total; a few more per kind would sharpen the
   `KIND_FRAMING` hints.


Give the challenge organizers your public URL plus this repo (for `bot.py`,
`submission.jsonl`, `conversation_handlers.py`, and this `README.md`).
