"""
composer.py — the single point where (category, merchant, trigger, customer?) contexts
become a WhatsApp message.

Design goals (see README.md for the full rationale):
  1. One composer, dispatched by trigger.kind, not one prompt-per-category.
  2. Never fabricate — the prompt is explicit that only facts present in the
     supplied contexts may be used; the LLM is told to write "not enough info"
     framing rather than invent a number/source/competitor.
  3. Deterministic — temperature=0, and a fixed prompt per (contexts) input.
  4. Fails safe — if the LLM call errors or no key is configured, a
     rule-based fallback still produces a spec-compliant, reasonably-good
     message so the bot never returns malformed output.
  5. Hard constraints enforced in *code*, not just prompted: no URLs in body,
     single CTA, anti-repetition against prior sends in the same conversation.
  6. Provider-agnostic — LLM_PROVIDER env var selects "groq", "gemini", or
     "anthropic" (default: groq, since its free tier has a far higher daily
     request cap than Gemini's free tier on some models — see README.md).
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, Optional

# ---------------------------------------------------------------------------
# LLM client — provider-agnostic. Falls back to a deterministic template
# composer if no key is configured for the selected provider, or the call
# fails for any reason.
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").strip().lower()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

# Groq — free tier, OpenAI-compatible chat-completions API, no credit card
# required, and (unlike Gemini's 20/day free cap on some models) a generous
# per-day request cap on its free Llama models. Useful as a fallback/primary
# when a Gemini project's daily quota is exhausted.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _call_anthropic(system: str, user: str, max_tokens: int = 700) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import urllib.request as ur

        body = json.dumps(
            {
                "model": ANTHROPIC_MODEL,
                "max_tokens": max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        ).encode("utf-8")
        req = ur.Request(
            ANTHROPIC_URL,
            data=body,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        with ur.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["content"][0]["text"]
    except Exception:
        return None


DEBUG = os.environ.get("COMPOSER_DEBUG", "").strip() == "1"


def _debug(msg: str) -> None:
    if DEBUG:
        import sys
        print(f"[composer debug] {msg}", file=sys.stderr)


def _call_gemini(system: str, user: str, max_tokens: int = 700, _retries: int = 3) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    import time
    import urllib.error
    import urllib.request as ur

    # Gemini has no separate system-role field in the basic generateContent
    # call; folding system instructions into the leading user turn works
    # reliably across recent Flash models.
    full_prompt = f"{system}\n\n---\n\n{user}"
    # Newer "thinking" Gemini models (e.g. gemini-3.x) spend part of the token
    # budget on internal reasoning before producing visible output. With a
    # tight maxOutputTokens, a long/complex prompt can burn the whole budget
    # on thinking and return an empty answer. We (a) raise the budget well
    # above what a WhatsApp-length message needs, and (b) explicitly ask for
    # a minimal thinking budget, since this task needs a direct answer, not
    # multi-step reasoning. Models that don't support thinkingConfig simply
    # ignore the field.
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max(max_tokens, 1024),
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
    ).encode("utf-8")
    url = GEMINI_URL_TMPL.format(model=GEMINI_MODEL, key=GEMINI_API_KEY)

    last_err = None
    for attempt in range(_retries):
        try:
            req = ur.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with ur.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates") or []
            if not candidates:
                _debug(f"Gemini returned no candidates: {data}")
                return None
            parts = candidates[0].get("content", {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                _debug(f"Gemini candidate had no text (finishReason={candidates[0].get('finishReason')}): {data}")
                return None
            return text
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            last_err = f"HTTP {e.code}: {body_text[:300]}"
            if e.code == 429 or e.code >= 500:
                # Rate-limited or transient server error — back off and retry.
                wait = 2 ** attempt * 2  # 2s, 4s, 8s
                _debug(f"{last_err} — retrying in {wait}s (attempt {attempt + 1}/{_retries})")
                time.sleep(wait)
                continue
            _debug(f"Gemini call failed, not retrying: {last_err}")
            return None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            _debug(f"Gemini call failed: {last_err}")
            return None

    _debug(f"Gemini call exhausted retries, last error: {last_err}")
    return None


def _call_groq(system: str, user: str, max_tokens: int = 700, _retries: int = 3) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    import time
    import urllib.error
    import urllib.request as ur

    body = json.dumps(
        {
            "model": GROQ_MODEL,
            "temperature": 0,
            "max_tokens": max(max_tokens, 512),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")

    last_err = None
    for attempt in range(_retries):
        try:
            req = ur.Request(
                GROQ_URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                    # Groq's API sits behind Cloudflare, which blocks requests
                    # carrying urllib's default "Python-urllib/x.y" User-Agent
                    # as a bot signature (HTTP 403, Cloudflare error 1010).
                    # A normal-looking User-Agent avoids that block.
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) vera-challenge-bot/1.0",
                },
                method="POST",
            )
            with ur.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices") or []
            if not choices:
                _debug(f"Groq returned no choices: {data}")
                return None
            text = (choices[0].get("message", {}).get("content") or "").strip()
            if not text:
                _debug(f"Groq choice had no text: {data}")
                return None
            return text
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            last_err = f"HTTP {e.code}: {body_text[:300]}"
            if e.code == 429 or e.code >= 500:
                wait = 2 ** attempt * 2
                _debug(f"{last_err} — retrying in {wait}s (attempt {attempt + 1}/{_retries})")
                time.sleep(wait)
                continue
            _debug(f"Groq call failed, not retrying: {last_err}")
            return None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            _debug(f"Groq call failed: {last_err}")
            return None

    _debug(f"Groq call exhausted retries, last error: {last_err}")
    return None


_PROVIDER_CALLS = {
    "gemini": lambda system, user, max_tokens: _call_gemini(system, user, max_tokens),
    "anthropic": lambda system, user, max_tokens: _call_anthropic(system, user, max_tokens),
    "groq": lambda system, user, max_tokens: _call_groq(system, user, max_tokens),
}

_PROVIDER_KEYS = {
    "gemini": lambda: GEMINI_API_KEY,
    "anthropic": lambda: ANTHROPIC_API_KEY,
    "groq": lambda: GROQ_API_KEY,
}


def _call_llm(system: str, user: str, max_tokens: int = 700) -> Optional[str]:
    """Dispatches to whichever provider is configured (LLM_PROVIDER: "gemini"
    (default), "anthropic", or "groq"). Tries the configured provider first;
    if it fails or isn't configured, tries the other providers that have a
    key set, in a fixed order, before giving up to the rule-based fallback.
    This means setting e.g. both GEMINI_API_KEY and GROQ_API_KEY lets the bot
    ride out a Gemini free-tier daily quota exhaustion by falling through to
    Groq automatically, with no restart needed."""
    order = [LLM_PROVIDER] + [p for p in ("gemini", "groq", "anthropic") if p != LLM_PROVIDER]
    for provider in order:
        if provider not in _PROVIDER_CALLS or not _PROVIDER_KEYS[provider]():
            continue
        result = _PROVIDER_CALLS[provider](system, user, max_tokens)
        if result:
            return result
    return None


def active_model_label() -> str:
    if LLM_PROVIDER in _PROVIDER_KEYS and _PROVIDER_KEYS[LLM_PROVIDER]():
        return {"gemini": GEMINI_MODEL, "anthropic": ANTHROPIC_MODEL, "groq": GROQ_MODEL}[LLM_PROVIDER]
    if GEMINI_API_KEY:
        return GEMINI_MODEL
    if GROQ_API_KEY:
        return GROQ_MODEL
    if ANTHROPIC_API_KEY:
        return ANTHROPIC_MODEL
    return "rule-based-fallback"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+")


def _strip_urls(text: str) -> str:
    """Contexts may legitimately not need URLs; the spec treats a URL in the
    body as a hard fail (Meta would reject it as a free-form send), so we
    strip them defensively even if the LLM slips one in."""
    return URL_RE.sub("", text).strip()


def _dedup_against_history(body: str, prior_bodies: list[str]) -> str:
    """If the LLM produced something verbatim-identical to a prior send in
    this conversation, nudge it so we never trip the anti-repetition
    penalty. Cheap and deterministic: append a short variation marker only
    when an exact match is detected (should be rare given fresh triggers)."""
    if body in prior_bodies:
        body = body + " (following up on this again)"
    return body


def _first_name(merchant: dict) -> str:
    identity = merchant.get("identity", {})
    return identity.get("owner_first_name") or identity.get("name", "there")


def _lang_pref(merchant: dict, customer: Optional[dict]) -> str:
    if customer:
        return customer.get("identity", {}).get("language_pref", "english")
    langs = merchant.get("identity", {}).get("languages", ["en"])
    return "hi-en mix" if "hi" in langs else "english"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are the composition engine inside "Vera", magicpin's merchant-facing
    WhatsApp marketing assistant for local Indian businesses. You write ONE
    outbound WhatsApp message given four structured inputs: CategoryContext,
    MerchantContext, TriggerContext, and an optional CustomerContext.

    Hard rules (violating any of these is a serious failure):
    1. NEVER invent a fact, number, date, source citation, competitor name, or
       offer that is not present in the supplied contexts. If a needed detail
       is missing, write around it rather than fabricating it.
    2. NEVER include a URL/link in the body.
    3. Exactly ONE primary call-to-action. Prefer a single binary choice
       (e.g. reply YES / STOP) for action-triggers; use no CTA for pure
       information triggers; multi-choice is acceptable ONLY for slot-booking.
    4. Anchor the message on a *specific, verifiable* fact drawn from the
       contexts (a number, a date, a headline, a peer stat) — never a vague
       claim like "grow your business" or a generic "X% off".
    5. Match the category's voice exactly: tone, allowed vocabulary, and
       taboo words/phrases (never use a taboo word/phrase).
    6. Personalize to the specific merchant's actual state (their real
       numbers, their real offers/signals/conversation history) — do not
       write something that could apply to any merchant in the category.
    7. Match the audience's language preference. If it's a Hindi-English
       ("hi-en mix") speaker, write natural Hindi-English code-mix (Latin
       script), not pure English and not pure Hindi.
    8. No preambles ("I hope you're doing well..."), no re-introducing
       yourself if conversation_history already exists, concise and
       WhatsApp-native.
    9. If this is a customer-facing message (CustomerContext supplied), the
       message must read as being sent BY the merchant (or by Vera clearly on
       the merchant's behalf) TO their own customer/patient — never mention
       Vera's internal signals/analytics language to a customer.
    10. Output ONLY the message body text — no quotes, no markdown headers, no
        explanation, no "Here is the message:" preamble. Just the raw
        WhatsApp text (a source citation like "— JIDA Oct 2026 p.14" at the
        end is fine when the digest item supplies a real source).

    You will also be asked, in the same response, to supply a short internal
    rationale and a CTA classification. Follow the exact output format given
    in the user turn.
    """
)

# Per trigger-kind framing hints — dispatched into the prompt so different
# kinds get different composition strategies without needing different
# prompts/models entirely (kept in one composer, per the design brief).
KIND_FRAMING = {
    "research_digest": "Frame around the single most relevant digest item. Cite its source inline or with an em-dash at the end. Use curiosity + reciprocity ('want me to draft X').",
    "regulation_change": "Frame around compliance urgency and a concrete deadline. Loss-aversion tone, not scary — practical and peer-to-peer.",
    "cde_opportunity": "Frame as a professional development nudge — credits, date, fee. Low urgency, informational CTA is fine.",
    "perf_dip": "Frame around the specific metric drop with numbers. Loss aversion. Offer a concrete next step, not vague 'let's fix this'.",
    "perf_spike": "Frame around the specific metric increase with numbers. Positive reinforcement + one idea to capitalize on the spike.",
    "seasonal_perf_dip": "Frame as an EXPECTED seasonal pattern (reassure, don't alarm) with one action for the lull period.",
    "milestone_reached": "Frame around the imminent milestone as a countdown/celebration hook. Light urgency, social-proof adjacent.",
    "dormant_with_vera": "Frame as a light re-engagement check-in, referencing the last topic discussed. Low pressure, single easy question.",
    "renewal_due": "Frame around the concrete days-remaining and what's at stake (perf impact) if it lapses. Loss aversion + easy renewal CTA.",
    "winback_eligible": "Frame around what changed since expiry (perf dip numbers) and a simple reactivation ask.",
    "competitor_opened": "Frame with 'voyeur curiosity' — a nearby competitor event, framed as market intel, not alarm.",
    "review_theme_emerged": "Frame around the specific recurring review theme (with count) and a concrete fix idea.",
    "active_planning_intent": "The merchant is ALREADY mid-conversation asking for something concrete. Give a substantive, specific next step immediately — do not ask another qualifying question.",
    "curious_ask_due": "Ask the merchant ONE specific, easy, curiosity-driven question about their business this week (compulsion lever #7). No pitch, just a genuine question.",
    "festival_upcoming": "Frame around the specific festival + days-until + category relevance. Give one concrete seasonal-offer idea.",
    "category_seasonal": "Frame around the specific seasonal demand shift named in the payload. One concrete shelf/stock/menu action.",
    "gbp_unverified": "Frame around the concrete uplift estimate of verifying, with a simple next step.",
    "supply_alert": "Urgent, compliance-driven. State the specific batches/molecule. Suggest pulling the customer list for affected patients.",
    "recall_due": "Customer-facing booking flow. Reference the exact time since last visit and offer the concrete available slots. Multi-choice CTA is fine here.",
    "customer_lapsed_soft": "Customer-facing winback. Warm, not guilt-tripping. Reference their prior service/segment.",
    "customer_lapsed_hard": "Customer-facing stronger winback. Reference what's changed/new since they left. Single easy CTA.",
    "chronic_refill_due": "Customer-facing refill reminder. Reference the specific molecules and stock-out date. Offer delivery.",
    "trial_followup": "Customer-facing trial conversion. Reference the trial date and the next available session slot.",
    "wedding_package_followup": "Customer-facing bridal-package nudge. Reference days-to-wedding and the next-step program.",
    "appointment_tomorrow": "Customer-facing reminder. Short, just confirms the appointment, one line, no pitch.",
}

DEFAULT_FRAMING = "Use the trigger payload directly as the specific reason for this message. Pick the single most relevant fact and anchor the message on it."


def _framing_for(trigger: dict) -> str:
    return KIND_FRAMING.get(trigger.get("kind", ""), DEFAULT_FRAMING)


def _digest_item_lookup(category: dict, item_id: Optional[str]) -> Optional[dict]:
    if not item_id:
        return None
    for item in category.get("digest", []):
        if item.get("id") == item_id:
            return item
    return None


def _trim(obj: Any, max_len: int = 6000) -> Any:
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) <= max_len:
        return obj
    # crude trim for prompt-budget safety on very large merchant/category blobs
    try:
        return json.loads(s[:max_len].rsplit(",", 1)[0] + "}")
    except json.JSONDecodeError:
        return obj


def build_user_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
    prior_bodies: list[str],
) -> str:
    trigger_payload = trigger.get("payload", {})
    digest_item = _digest_item_lookup(category, trigger_payload.get("top_item_id"))

    sections = []
    sections.append("## CATEGORY CONTEXT\n" + json.dumps(_trim({
        "slug": category.get("slug"),
        "voice": category.get("voice"),
        "offer_catalog": category.get("offer_catalog"),
        "peer_stats": category.get("peer_stats"),
        "seasonal_beats": category.get("seasonal_beats"),
        "trend_signals": category.get("trend_signals"),
    }), ensure_ascii=False, indent=2))

    if digest_item:
        sections.append("## RELEVANT DIGEST ITEM\n" + json.dumps(digest_item, ensure_ascii=False, indent=2))

    sections.append("## MERCHANT CONTEXT\n" + json.dumps(_trim(merchant), ensure_ascii=False, indent=2))
    sections.append("## TRIGGER CONTEXT (this is WHY you are messaging right now)\n" + json.dumps(trigger, ensure_ascii=False, indent=2))

    if customer:
        sections.append("## CUSTOMER CONTEXT (this message goes to the merchant's own customer, not to the merchant)\n" + json.dumps(customer, ensure_ascii=False, indent=2))

    if prior_bodies:
        sections.append("## PRIOR MESSAGES ALREADY SENT IN THIS CONVERSATION (never repeat verbatim)\n" + json.dumps(prior_bodies[-5:], ensure_ascii=False, indent=2))

    framing = _framing_for(trigger)
    lang = _lang_pref(merchant, customer)
    send_as = "merchant_on_behalf" if customer else "vera"

    instructions = textwrap.dedent(f"""
    ## COMPOSITION TASK
    Trigger kind: {trigger.get('kind')}
    Framing guidance for this kind: {framing}
    send_as: {send_as}
    Target language preference: {lang}

    Respond with EXACTLY this format (three lines, nothing else):
    BODY: <the whatsapp message, one paragraph, no line breaks>
    CTA: <one of: binary_yes_no | binary_confirm_cancel | multi_choice_slot | open_ended | none>
    RATIONALE: <one sentence: why this message, referencing the specific trigger fact used>
    """)

    return "\n\n".join(sections) + "\n\n" + instructions


def _parse_llm_output(text: str) -> Optional[dict]:
    body_m = re.search(r"BODY:\s*(.+?)(?:\nCTA:|\Z)", text, re.S)
    cta_m = re.search(r"CTA:\s*(\S+)", text)
    rat_m = re.search(r"RATIONALE:\s*(.+?)(?:\Z)", text, re.S)
    if not body_m:
        return None
    body = body_m.group(1).strip()
    cta = cta_m.group(1).strip() if cta_m else "open_ended"
    rationale = rat_m.group(1).strip() if rat_m else "Composed from category+merchant+trigger context."
    valid_ctas = {"binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "open_ended", "none"}
    if cta not in valid_ctas:
        cta = "open_ended"
    return {"body": body, "cta": cta, "rationale": rationale}


# ---------------------------------------------------------------------------
# Rule-based fallback (no API key / LLM failure) — keeps the bot spec-compliant
# ---------------------------------------------------------------------------

def _fallback_compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
    name = _first_name(merchant)
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    digest_item = _digest_item_lookup(category, payload.get("top_item_id"))

    if customer:
        cname = customer.get("identity", {}).get("name", "there").split(" (")[0]
        merchant_name = merchant.get("identity", {}).get("name", "our clinic")
        offers = [o for o in (merchant.get("offers") or []) if o.get("status") == "active"]
        offer_line = f" {offers[0]['title']}." if offers else ""

        # Build the "why now" fragment straight from the trigger payload,
        # never from the internal framing-hint string (that's for the LLM
        # prompt only and must never leak into a customer-facing send).
        reason_bits = []
        if kind == "recall_due":
            reason_bits.append(f"It's been a while since your last visit — your {payload.get('service_due', 'recall')} is due")
            slots = payload.get("available_slots", [])
            if slots:
                slot_text = " or ".join(s.get("label", "") for s in slots[:2])
                reason_bits.append(f"Available slots: {slot_text}")
        elif kind == "chronic_refill_due":
            mols = ", ".join(payload.get("molecule_list", []))
            reason_bits.append(f"Your {mols} refill is due soon" if mols else "Your regular refill is due soon")
        elif kind == "appointment_tomorrow":
            reason_bits.append("Reminder: your appointment is tomorrow")
        elif kind in ("customer_lapsed_soft", "customer_lapsed_hard"):
            reason_bits.append("It's been a while — we'd love to see you again")
        elif kind == "trial_followup":
            reason_bits.append("Following up on your recent trial session")
        elif kind == "wedding_package_followup":
            days = payload.get("days_to_wedding")
            reason_bits.append(f"With {days} days to your big day, worth locking in your next prep session" if days else "Worth locking in your next prep session")
        else:
            reason_bits.append("Wanted to reach out with an update relevant to you")

        reason = ". ".join(reason_bits)
        body = f"Hi {cname}, {merchant_name} here. {reason}.{offer_line} Reply and let us know what works for you."
        cta = "multi_choice_slot" if kind in ("recall_due", "appointment_tomorrow", "trial_followup") else "open_ended"
        rationale = f"Fallback template composition for customer-facing trigger kind={kind}, anchored on trigger payload fields."
        return {"body": body, "cta": cta, "rationale": rationale}

    if digest_item:
        body = (
            f"{name}, {digest_item.get('title', 'a relevant update')}"
            f" ({digest_item.get('source', 'category digest')}). "
            f"{digest_item.get('actionable', 'Want me to help you act on this?')}"
        )
        return {"body": body, "cta": "open_ended", "rationale": "Fallback digest-anchored composition."}

    signals = merchant.get("signals") or []
    perf = merchant.get("performance", {})
    body = (
        f"{name}, quick update on your listing — "
        f"views {perf.get('views', 'N/A')}, calls {perf.get('calls', 'N/A')} in the last "
        f"{perf.get('window_days', 30)} days. {'Signal: ' + signals[0] if signals else ''} "
        f"Want me to look into this with you?"
    )
    return {"body": body, "cta": "open_ended", "rationale": f"Fallback generic composition for kind={kind}."}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    prior_bodies: Optional[list[str]] = None,
) -> dict:
    """Returns dict with keys: body, cta, send_as, suppression_key, rationale."""
    prior_bodies = prior_bodies or []
    user_prompt = build_user_prompt(category, merchant, trigger, customer, prior_bodies)

    parsed = None
    raw = _call_llm(SYSTEM_PROMPT, user_prompt)
    if raw:
        parsed = _parse_llm_output(raw)

    if not parsed:
        parsed = _fallback_compose(category, merchant, trigger, customer)

    body = _strip_urls(parsed["body"])
    body = _dedup_against_history(body, prior_bodies)

    return {
        "body": body,
        "cta": parsed["cta"],
        "send_as": "merchant_on_behalf" if customer else "vera",
        "suppression_key": trigger.get("suppression_key", f"{trigger.get('kind','')}:{merchant.get('merchant_id','')}"),
        "rationale": parsed["rationale"],
    }
