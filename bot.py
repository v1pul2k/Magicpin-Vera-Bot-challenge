"""
bot.py — Vera-challenge candidate bot.

Implements the 5-endpoint HTTP contract from `challenge-testing-brief.md`:
  GET  /v1/healthz
  GET  /v1/metadata
  POST /v1/context
  POST /v1/tick
  POST /v1/reply

Also exposes a plain-Python `compose(category, merchant, trigger, customer)`
function (challenge-brief.md §7.1 shape) so the same logic can be scored
offline via `generate_submission.py` without going through HTTP.

Run locally:
    pip install -r requirements.txt
    export LLM_PROVIDER=groq
    export GROQ_API_KEY=gsk_...            # from console.groq.com/keys
    uvicorn bot:app --host 0.0.0.0 --port 8080

(Or use Gemini/Anthropic instead: LLM_PROVIDER=gemini + GEMINI_API_KEY, or
LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY. Without any key, the bot still
runs on a rule-based fallback.)
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

import composer
from conversation_handlers import ConversationState, respond

app = FastAPI(title="Vera Challenge Bot")

START_TIME = time.time()

# ---------------------------------------------------------------------------
# In-memory state (fine per the spec: "storing in memory is fine; just don't
# restart between calls"). A single process handles the whole test window.
# ---------------------------------------------------------------------------

# (scope, context_id) -> {"version": int, "payload": dict}
CONTEXTS: dict[tuple[str, str], dict] = {}

# conversation_id -> ConversationState
CONVERSATIONS: dict[str, ConversationState] = {}

# suppression_key -> last_sent_at (epoch seconds) — dedup across ticks
SUPPRESSED_UNTIL: dict[str, float] = {}
SUPPRESSION_COOLDOWN_SECONDS = 6 * 3600  # don't resend the same suppression_key within 6h

# per-merchant list of (conversation_id) already opened this test, so a tick
# doesn't open two parallel conversations for the same merchant+trigger.
MERCHANT_ACTIVE_TRIGGERS: dict[str, set[str]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_context(scope: str, context_id: str) -> Optional[dict]:
    entry = CONTEXTS.get((scope, context_id))
    return entry["payload"] if entry else None


def _category_for_merchant(merchant: dict) -> Optional[dict]:
    slug = merchant.get("category_slug")
    return _get_context("category", slug) if slug else None


def _contexts_loaded_counts() -> dict[str, int]:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in CONTEXTS.keys():
        counts[scope] = counts.get(scope, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# GET /v1/healthz
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": _contexts_loaded_counts(),
    }


# ---------------------------------------------------------------------------
# GET /v1/metadata
# ---------------------------------------------------------------------------

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.environ.get("TEAM_NAME", "Solo Builder"),
        "team_members": os.environ.get("TEAM_MEMBERS", "").split(",") if os.environ.get("TEAM_MEMBERS") else [],
        "model": composer.active_model_label(),
        "approach": (
            "Single composer dispatched by trigger.kind over 4 injected contexts "
            "(category/merchant/trigger/customer); code-enforced constraints "
            "(no URLs, single CTA, anti-repetition, suppression dedup) layered "
            "on top of an LLM-authored body (Gemini or Anthropic, provider-"
            "switchable); deterministic rule-based fallback keeps the bot "
            "spec-compliant if the LLM call fails."
        ),
        "contact_email": os.environ.get("CONTACT_EMAIL", "team@example.com"),
        "version": "1.0.0",
        "submitted_at": os.environ.get("SUBMITTED_AT", _now_iso()),
    }


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: Optional[str] = None


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return {"accepted": False, "reason": "invalid_scope", "details": f"unknown scope '{body.scope}'"}

    key = (body.scope, body.context_id)
    cur = CONTEXTS.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}

    CONTEXTS[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


def _should_suppress(suppression_key: str, now_epoch: float) -> bool:
    last = SUPPRESSED_UNTIL.get(suppression_key)
    return bool(last and now_epoch - last < SUPPRESSION_COOLDOWN_SECONDS)


def _trigger_expired(trigger: dict, now_iso: str) -> bool:
    expires = trigger.get("expires_at")
    if not expires:
        return False
    try:
        exp = datetime.fromisoformat(expires.replace("Z", "+00:00"))
        now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        return now > exp
    except Exception:
        return False


@app.post("/v1/tick")
async def tick(body: TickBody):
    now_epoch = time.time()
    tick_start_time = time.time()
    actions: list[dict] = []

    for trg_id in body.available_triggers:
        if time.time() - tick_start_time > 25:
            break
        if len(actions) >= 20:  # tick action cap per spec §5
            break

        trigger = _get_context("trigger", trg_id)
        if not trigger:
            continue
        if _trigger_expired(trigger, body.now):
            continue

        suppression_key = trigger.get("suppression_key", trg_id)
        if _should_suppress(suppression_key, now_epoch):
            continue

        merchant_id = trigger.get("merchant_id")
        merchant = _get_context("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue

        category = _category_for_merchant(merchant)
        if not category:
            continue

        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            customer = _get_context("customer", customer_id)

        seen = MERCHANT_ACTIVE_TRIGGERS.setdefault(merchant_id, set())
        if trg_id in seen:
            continue  # already opened a conversation for this exact trigger

        composed = composer.compose(category=category, merchant=merchant, trigger=trigger, customer=customer)

        conversation_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:8]}"
        state = ConversationState(
            conversation_id=conversation_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            trigger_id=trg_id,
        )
        state.sent_bodies.append(composed["body"])
        state.turns.append({"from": "vera", "message": composed["body"]})
        CONVERSATIONS[conversation_id] = state
        SUPPRESSED_UNTIL[suppression_key] = now_epoch
        seen.add(trg_id)

        actions.append(
            {
                "conversation_id": conversation_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": composed["send_as"],
                "trigger_id": trg_id,
                "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
                "template_params": [merchant.get("identity", {}).get("name", ""), composed["body"]],
                "body": composed["body"],
                "cta": composed["cta"],
                "suppression_key": suppression_key,
                "rationale": composed["rationale"],
            }
        )

    return {"actions": actions}


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: int = 0


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    state = CONVERSATIONS.get(body.conversation_id)
    if state is None:
        # Judge may open a replay scenario without a prior /v1/tick (Phase 4).
        state = ConversationState(
            conversation_id=body.conversation_id,
            merchant_id=body.merchant_id or "",
            customer_id=body.customer_id,
        )
        CONVERSATIONS[body.conversation_id] = state

    merchant = _get_context("merchant", body.merchant_id) if body.merchant_id else None
    if not merchant:
        # Unknown merchant — degrade gracefully rather than 500.
        return {"action": "end", "rationale": "No merchant context available for this conversation_id; cannot compose a reply."}

    category = _category_for_merchant(merchant)
    customer = _get_context("customer", body.customer_id) if body.customer_id else None
    trigger = _get_context("trigger", state.trigger_id) if state.trigger_id else None

    if state.ended:
        return {"action": "end", "rationale": "Conversation already closed."}

    result = respond(
        state=state,
        merchant_message=body.message,
        category=category or {},
        merchant=merchant,
        trigger=trigger,
        customer=customer,
    )
    return result


# ---------------------------------------------------------------------------
# Optional teardown (spec §11): wipe state at end of test.
# ---------------------------------------------------------------------------

@app.post("/v1/teardown")
async def teardown():
    CONTEXTS.clear()
    CONVERSATIONS.clear()
    SUPPRESSED_UNTIL.clear()
    MERCHANT_ACTIVE_TRIGGERS.clear()
    return {"status": "wiped"}


# ---------------------------------------------------------------------------
# Plain-Python compose() — challenge-brief.md §7.1 shape, used by
# generate_submission.py for offline scoring against the 30 test pairs.
# ---------------------------------------------------------------------------

def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
    return composer.compose(category=category, merchant=merchant, trigger=trigger, customer=customer)


@app.get("/")
async def root():
    return {"service": "vera-challenge-bot", "status": "up", "docs": "/docs"}
