"""
conversation_handlers.py — turn-by-turn reply logic (the "open challenges"
from challenge-brief.md §12: auto-reply detection, intent-transition
handling, knowing when to stop).

Exposes `respond(state, merchant_message) -> dict` per the optional
deliverable spec (§7.4), and is imported directly by bot.py's /v1/reply
endpoint so there's exactly one implementation of this logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import composer

# ---------------------------------------------------------------------------
# Signal detectors
# ---------------------------------------------------------------------------

AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"our team will (respond|get back|reach out)",
    r"automated (assistant|response|reply)",
    r"we (have received|received) your message",
    r"aapki jaankari ke liye .*shukriya",
    r"hamari team tak pahunchane",
    r"currently (unavailable|away|closed)",
    r"business hours",
]

HOSTILE_PATTERNS = [
    r"stop (messaging|texting|contacting)",
    r"useless",
    r"spam",
    r"harass",
    r"leave me alone",
    r"\bshut up\b",
    r"don'?t (message|text|contact) (me|again)",
]

NOT_INTERESTED_PATTERNS = [
    r"not interested",
    r"no thanks",
    r"nahi chahiye",
    r"remove me",
    r"unsubscribe",
]

INTENT_COMMIT_PATTERNS = [
    r"\blet'?s do it\b",
    r"\bgo ahead\b",
    r"\byes,? (please )?(send|do it|proceed|confirm)\b",
    r"\bok(ay)?,? let'?s\b",
    r"\bi want to join\b",
    r"\bmujhe .*(karna|karni) hai\b",
    r"\bsure,? (go|do it|send)\b",
    r"\bconfirm\b",
    r"^yes$",
]

OFF_TOPIC_HINTS = [
    r"\bgst\b", r"\bloan\b", r"\binsurance\b", r"\btax\b", r"license",
]


def _matches_any(patterns: list[str], text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def is_auto_reply(message: str) -> bool:
    return _matches_any(AUTO_REPLY_PATTERNS, message)


def is_hostile(message: str) -> bool:
    return _matches_any(HOSTILE_PATTERNS, message)


def is_not_interested(message: str) -> bool:
    return _matches_any(NOT_INTERESTED_PATTERNS, message)


def is_intent_commit(message: str) -> bool:
    return _matches_any(INTENT_COMMIT_PATTERNS, message)


def is_off_topic(message: str) -> bool:
    return _matches_any(OFF_TOPIC_HINTS, message)


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    trigger_id: Optional[str] = None
    turns: list[dict] = field(default_factory=list)  # [{from, message}]
    sent_bodies: list[str] = field(default_factory=list)
    consecutive_auto_replies: int = 0
    ended: bool = False

    def last_from_role(self, role: str) -> Optional[str]:
        for t in reversed(self.turns):
            if t["from"] == role:
                return t["message"]
        return None

    def repeated_auto_reply_count(self, message: str) -> int:
        """How many of the most-recent consecutive merchant turns equal this
        exact message (used for the 'same message 3+ times = auto-reply'
        heuristic from the brief)."""
        count = 0
        for t in reversed(self.turns):
            if t["from"] != "merchant":
                continue
            if t["message"].strip() == message.strip():
                count += 1
            else:
                break
        return count


# ---------------------------------------------------------------------------
# Core respond()
# ---------------------------------------------------------------------------

def respond(
    state: ConversationState,
    merchant_message: str,
    category: dict,
    merchant: dict,
    trigger: Optional[dict] = None,
    customer: Optional[dict] = None,
) -> dict:
    """Given the conversation so far + the latest inbound message, decide the
    next move: send / wait / end. Mutates `state` in place (appends turns,
    tracks sent bodies) so the caller can persist it.
    """
    # Count repeats against history BEFORE appending this turn (otherwise the
    # just-appended message always matches itself and the count is off-by-one
    # even on a first-ever reply).
    prior_repeat_count = state.repeated_auto_reply_count(merchant_message)
    state.turns.append({"from": "merchant", "message": merchant_message})

    # 1. Hostile → graceful, brief exit. Never argue, never re-pitch.
    if is_hostile(merchant_message):
        if state.ended:
            return {"action": "end", "rationale": "Already ended; hostile message confirms it."}
        state.ended = True
        return {
            "action": "send",
            "body": "Apologies — I won't message again. If anything changes, just say 'Hi Vera' anytime. 🙏",
            "cta": "none",
            "rationale": "Merchant expressed frustration/hostility; one brief apology + opt-out, then closing the conversation.",
        }

    # 2. Explicit not-interested → graceful exit, no further pitching.
    if is_not_interested(merchant_message):
        state.ended = True
        return {
            "action": "end",
            "rationale": "Merchant explicitly signaled not interested; exiting without further nudges.",
        }

    # 3. Auto-reply detection — same canned text repeating.
    #    prior_repeat_count == 0 the first time this exact text is seen;
    #    only treat it as an *auto-reply pattern* on that first occurrence
    #    if it structurally looks like a canned WA Business reply. Any
    #    verbatim repeat of a merchant message (2nd+ occurrence) is always
    #    treated as auto-reply behavior regardless of wording.
    looks_like_auto_reply = is_auto_reply(merchant_message)
    if prior_repeat_count >= 1 or looks_like_auto_reply:
        state.consecutive_auto_replies = prior_repeat_count + 1
        if state.consecutive_auto_replies == 1:
            return {
                "action": "send",
                "body": "Looks like an auto-reply 😊 When the owner sees this, a quick 'Yes' is all it takes.",
                "cta": "binary_yes_no",
                "rationale": "Detected likely auto-reply; one gentle explicit prompt so a human sees it, without repeating the full pitch.",
            }
        elif state.consecutive_auto_replies == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Same auto-reply twice in a row — owner likely not at the phone. Backing off 24h before retrying.",
            }
        else:
            state.ended = True
            return {
                "action": "end",
                "rationale": "Auto-reply 3+ times with zero real engagement signal; closing to avoid wasting turns.",
            }

    # 4. Intent commitment → switch straight to action, never re-qualify.
    if is_intent_commit(merchant_message):
        next_step = _action_next_step(trigger, merchant, customer)
        body = composer._dedup_against_history(next_step, state.sent_bodies)
        state.sent_bodies.append(body)
        state.turns.append({"from": "vera", "message": body})
        return {
            "action": "send",
            "body": body,
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant explicitly committed ('yes/let's do it') — switching from pitch to concrete action immediately, not re-qualifying.",
        }

    # 5. Off-topic / unrelated ask → briefly acknowledge, redirect politely, stay on mission.
    if is_off_topic(merchant_message):
        body = "That's outside what I can help with here, but happy to keep going on your Google/marketing side whenever you're ready."
        state.sent_bodies.append(body)
        state.turns.append({"from": "vera", "message": body})
        return {
            "action": "send",
            "body": body,
            "cta": "none",
            "rationale": "Off-topic ask (e.g. GST/tax/loan); politely declined and redirected back to Vera's mission without being curt.",
        }

    # 6. Otherwise — normal engaged reply. Compose a contextual continuation.
    pseudo_trigger = trigger or {
        "kind": "conversation_continuation",
        "payload": {},
        "urgency": 2,
        "suppression_key": f"cont:{state.conversation_id}",
    }
    composed = composer.compose(
        category=category,
        merchant=merchant,
        trigger={**pseudo_trigger, "payload": {**pseudo_trigger.get("payload", {}), "merchant_last_message": merchant_message}},
        customer=customer,
        prior_bodies=state.sent_bodies,
    )
    state.sent_bodies.append(composed["body"])
    state.turns.append({"from": "vera", "message": composed["body"]})
    return {
        "action": "send",
        "body": composed["body"],
        "cta": composed["cta"],
        "rationale": composed["rationale"],
    }


def _action_next_step(trigger: Optional[dict], merchant: dict, customer: Optional[dict]) -> str:
    name = merchant.get("identity", {}).get("owner_first_name") or merchant.get("identity", {}).get("name", "there")
    kind = (trigger or {}).get("kind", "")
    if customer:
        return f"Great — booking it now for {customer.get('identity', {}).get('name', 'them')}. I'll confirm the slot shortly."
    if kind in ("active_planning_intent",):
        topic = (trigger or {}).get("payload", {}).get("intent_topic", "this")
        return f"Great, {name} — drafting the {topic.replace('_', ' ')} plan now. I'll share it here in a moment; reply CONFIRM once you've seen it to publish."
    return f"Great, {name} — doing this now. I'll confirm here once it's done; reply CONFIRM to proceed."
