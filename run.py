#!/usr/bin/env python
"""CLI entry point.

    python run.py samples/msa.pdf --playbook playbook/default_playbook.yaml -o out/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from summarizer.config import SETTINGS
from summarizer.llm import missing_keys
from summarizer.pipeline import render_review_packet, run


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic contract summarizer (LiteLLM -> Anthropic).")
    parser.add_argument("document", help="PDF, .txt or .md to analyse")
    parser.add_argument("--playbook", default="playbook/default_playbook.yaml")
    parser.add_argument("-o", "--out", default="out", help="output directory")
    parser.add_argument("--whole-doc", action="store_true", help="skip extraction, send the PDF to Claude directly")
    parser.add_argument("--no-ocr", action="store_true", help="never OCR; fall back to whole-doc instead")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every agent tool call")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Provider-agnostic: the key needed depends on SUMMARIZER_MODEL / SUMMARIZER_JUDGE_MODEL.
    needed = sorted(set(missing_keys(SETTINGS.model)) | set(missing_keys(SETTINGS.judge_model)))
    if needed:
        print(f"Missing environment variable(s): {', '.join(needed)}", file=sys.stderr)
        print(f"  model:       {SETTINGS.model}", file=sys.stderr)
        print(f"  judge model: {SETTINGS.judge_model}", file=sys.stderr)
        return 2

    result = run(
        args.document,
        args.playbook,
        force_whole_doc=args.whole_doc,
        allow_ocr=not args.no_ocr,
        verbose=args.verbose,
        on_progress=lambda msg: print(msg, flush=True),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.document).stem

    json_path = out_dir / f"{stem}.result.json"
    md_path = out_dir / f"{stem}.review.md"
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_review_packet(result), encoding="utf-8")

    print(f"\njudge {result.judge.score:.2f} ({result.judge.verdict}) after {result.attempts} attempt(s)")
    print(f"grounding {result.grounding_rate:.0%} | overall {result.validation.overall} | ${result.usd_cost:.3f}")
    print(f"\n  {md_path}\n  {json_path}")

    # Non-zero when a human must intervene before this can move forward.
    return 1 if result.validation.overall == "fail" or result.judge.verdict == "revise" else 0


if __name__ == "__main__":
    raise SystemExit(main())
