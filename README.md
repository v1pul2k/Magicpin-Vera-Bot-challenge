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
| **Groq** (default) | `GROQ_API_KEY`, optional `GROQ_MODEL` (default `openai/gpt-oss-120b`) | Free, no credit card, sign up at [console.groq.com/keys](https://console.groq.com/keys). A much higher free daily request cap than Gemini's free tier — the reliable default for generating all 30 submissions or running a full judge session without hitting a quota wall. Groq's model lineup changes over time — if you get a `model_not_found` error, list what's currently available to your key at `GET https://api.groq.com/openai/v1/models` and set `GROQ_MODEL` to a current chat/instruct model. |
| Gemini | `LLM_PROVIDER=gemini`, `GEMINI_API_KEY`, optional `GEMINI_MODEL` (default `gemini-3.8-flash`) | Free tier at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey). **Caveat:** the free tier caps some models at just 20 requests/day **per Google Cloud project** (not per key — a new key on the same project does not reset it). If you hit `429 RESOURCE_EXHAUSTED` with `quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier`, wait for the daily reset, create a genuinely new project via AI Studio, or just use Groq. |
| Anthropic | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` | Needs a funded Anthropic Console account |

If the configured provider's key is missing or a call fails, the composer
falls through the other providers that have a key set (`gemini` → `groq` →
`anthropic`, skipping whichever one you set as primary since it's already
been tried), and if none succeed, falls back to a deterministic rule-based
template so the bot never returns malformed/empty output. You can set more
than one provider's key at once for automatic fallback (e.g. both
`GROQ_API_KEY` and `GEMINI_API_KEY`) with no restart needed if one runs out
mid-run.

**Never commit an API key to git.** Set it as an environment variable (see
"Running locally" below) or, once deployed, in the Render dashboard's
Environment tab. `render.yaml` marks both key vars `sync: false` for exactly
this reason — Render will prompt you to paste the value there instead of
reading it from the repo.

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

## Running locally

```powershell
pip install -r requirements.txt

# Groq (default — free, no credit card, generous daily request cap)
$env:LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "your-groq-key-here"

# Optional: also set a Gemini key as a secondary provider.
$env:GEMINI_API_KEY = "your-gemini-key-here"

uvicorn bot:app --host 0.0.0.0 --port 8080
```

Without any key set, the bot still runs — it just uses the plain rule-based
fallback in `composer.py` instead of calling an LLM.

Self-test with magicpin's harness (needs its own LLM key, for the judge role
— separate from your bot's key; can be the same Gemini key):

```powershell
$env:BOT_URL = "http://localhost:8080"
python judge_simulator.py
```

Edit the `CONFIGURATION` section at the top of `judge_simulator.py` first
(`LLM_PROVIDER = "gemini"`, `LLM_API_KEY = "..."`).

Regenerate `submission.jsonl` (the 30 canonical test-pair messages) without
going through HTTP at all:

```powershell
python generate_submission.py
```

---

## Deploying: GitHub → Render, and keeping it alive with a cron

### 1. Push this folder to GitHub

```bash
cd vera-bot
git init
git add .
git commit -m "Vera challenge bot: 4-context composer + 5-endpoint harness"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

### 2. Connect the repo to Render

**Option A — one-click via the Blueprint (`render.yaml`), recommended:**

1. Go to <https://dashboard.render.com> → **New** → **Blueprint**.
2. Connect your GitHub repo — Render reads `render.yaml` automatically.
3. Click **Apply**. Because the key vars are marked `sync: false`, Render
   prompts you to paste the actual value in the dashboard rather than
   storing it in git — paste your `GEMINI_API_KEY` there (Environment tab).
4. Deploy runs automatically, giving you a URL like
   `https://vera-challenge-bot.onrender.com`.

**Option B — manual Web Service (no `render.yaml`):**

1. Dashboard → **New** → **Web Service** → connect the repo.
2. Runtime: **Python 3**. Build: `pip install -r requirements.txt`.
   Start: `uvicorn bot:app --host 0.0.0.0 --port $PORT`.
3. Health check path: `/v1/healthz`.
4. Add env vars `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` under Environment.

Every push to `main` auto-redeploys either way.

### 3. Verify the deployment

```bash
curl https://<your-service>.onrender.com/v1/healthz
curl https://<your-service>.onrender.com/v1/metadata
```

### 4. Keep it alive with a cron (GitHub Actions)

Render's **free** tier spins down after ~15 minutes idle, and the next
request pays a cold-start penalty that could eat into the judge's 30s
timeout. `.github/workflows/keepalive.yml` (already committed) pings
`/v1/healthz` every 10 minutes to keep the instance warm.

To activate it:
1. Repo → **Settings → Secrets and variables → Actions → New repository secret**.
2. Name: `BOT_URL`. Value: your Render URL (no trailing slash).
3. Done — it's already scheduled. Trigger it immediately from the **Actions** tab if you want to confirm it works right away.

### 5. Submit

Give the challenge organizers your public URL plus this repo (for `bot.py`,
`submission.jsonl`, `conversation_handlers.py`, and this `README.md`).
