#!/usr/bin/env python3
"""
generate_submission.py — produce submission.jsonl (30 lines, one per
canonical test pair in dataset/test_pairs.json) using the same `compose()`
that bot.py exposes over HTTP.

Usage:
    export LLM_PROVIDER=gemini
    export GEMINI_API_KEY=AIza...     # optional, falls back to templates without it
    python generate_submission.py
"""
import json
import os
import time
from pathlib import Path

from bot import compose

DATASET = Path(__file__).parent / "dataset"

# Small pause between calls so we don't burst past a free-tier LLM's
# requests-per-minute limit (Gemini/Anthropic free tiers are typically low
# single-digit-to-low-double-digit RPM). Override with SUBMISSION_DELAY_SECONDS
# if your provider/plan allows faster calls.
DELAY_SECONDS = float(os.environ.get("SUBMISSION_DELAY_SECONDS", "4"))


def load(scope_dir: str, filename_no_ext: str) -> dict:
    p = DATASET / scope_dir / f"{filename_no_ext}.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    with open(DATASET / "test_pairs.json") as f:
        pairs = json.load(f)["pairs"]

    out_lines = []
    for pair in pairs:
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        trigger = load("triggers", trigger_id)
        merchant = load("merchants", merchant_id)
        category = load("categories", merchant["category_slug"])
        customer = load("customers", customer_id) if customer_id else None

        result = compose(category, merchant, trigger, customer)
        line = {
            "test_id": test_id,
            "body": result["body"],
            "cta": result["cta"],
            "send_as": result["send_as"],
            "suppression_key": result["suppression_key"],
            "rationale": result["rationale"],
        }
        out_lines.append(line)
        print(f"{test_id}: {result['body'][:90]}...")
        time.sleep(DELAY_SECONDS)

    out_path = Path(__file__).parent / "submission.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out_lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
