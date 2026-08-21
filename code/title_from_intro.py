#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
title_from_intro.py

Generate better titles for normalized transcript files by calling the OpenAI API
on only the first ~2 paragraphs of the transcript.

Input format (normalized):
TITLE: ...
URL: ...
VIDEO_ID: ...

TRANSCRIPT:
...

Outputs:
- By default writes updated files to --out
- Optional --in-place to overwrite originals
- Optional --only-blank-or-weak to avoid rewriting good YouTube titles
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

load_dotenv("config.env")


try:
    from openai import OpenAI
except Exception:
    OpenAI = None

HEADER_RE = re.compile(
    r"^TITLE:\s*(?P<title>.*)\n"
    r"URL:\s*(?P<url>.*)\n"
    r"VIDEO_ID:\s*(?P<vid>.*)\n"
    r"\nTRANSCRIPT:\n(?P<body>.*)$",
    re.DOTALL,
)


def parse_normalized(text: str) -> Tuple[str, str, str, str]:
    m = HEADER_RE.match(text.strip())
    if not m:
        raise ValueError(
            "Normalized file missing required headers (TITLE/URL/VIDEO_ID/TRANSCRIPT)."
        )
    return (
        m.group("title").strip(),
        m.group("url").strip(),
        m.group("vid").strip(),
        m.group("body").strip(),
    )


def write_normalized(path: Path, title: str, url: str, vid: str, body: str) -> None:
    out = f"TITLE: {title}\nURL: {url}\nVIDEO_ID: {vid}\n\nTRANSCRIPT:\n{body}\n"
    path.write_text(out, encoding="utf-8")


def looks_weak(title: str) -> bool:
    if not title:
        return True
    t = title.strip()
    if len(t) < 14:
        return True
    if re.search(r"\b(mixdown|audio|final|v\d+)\b", t, re.IGNORECASE):
        return True
    if "_" in t and " " not in t:
        return True
    return False


def first_two_paragraphs(body: str, max_chars: int = 1600) -> str:
    """
    Grab top two paragraphs, where paragraphs are separated by blank lines.
    If transcript has no blank lines, use first ~max_chars.
    """
    raw = body.strip()
    if not raw:
        return ""

    # Normalize newlines
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Split by blank lines
    paras = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
    if len(paras) >= 2:
        snippet = paras[0] + "\n\n" + paras[1]
    else:
        # Fall back: first ~30 lines
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        snippet = "\n".join(lines[:30])

    snippet = snippet.strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0].strip() + "…"
    return snippet


SYSTEM = (
    "You generate accurate, topic-first episode titles for crypto/finance audio.\n"
    "Constraints:\n"
    "- Output a single line title only. No quotes. No extra text.\n"
    "- Prefer: <Topic/subject> (<Guest>) [<Show>] when evidence exists.\n"
    "- Do NOT lead with the show name unless there is no topic.\n"
    "- If the guest is unclear, omit them.\n"
    "- Keep it under 120 characters when possible.\n"
    "- Use American English.\n"
)

USER_TMPL = """You are given the first ~2 paragraphs of a transcript intro.

Task: Propose a concise, specific, topic-first title that captures what this episode is about.
If a guest name is clearly stated, include it in parentheses.
If the show name is clearly stated, include it at the end in brackets.

Intro:
{intro}
"""


def call_title(
    client: "OpenAI", intro: str, model: str, max_output_tokens: int, timeout: int
) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(intro=intro)},
        ],
        max_output_tokens=max_output_tokens,
        timeout=timeout,
    )
    # Responses API returns output_text convenience
    title = (resp.output_text or "").strip()
    title = re.sub(r"\s+", " ", title).strip()

    # Defensive cleanup: remove wrapping quotes if model adds them
    title = title.strip("\"'")

    # Hard guard: single line
    title = title.splitlines()[0].strip() if title else title
    return title


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="indir", required=True, help="Folder with normalized *.txt"
    )
    ap.add_argument(
        "--out",
        dest="outdir",
        default="",
        help="Output folder (default: <in>_titled_api)",
    )
    ap.add_argument("--in-place", action="store_true", help="Overwrite files in --in")
    ap.add_argument(
        "--only-blank-or-weak",
        action="store_true",
        help="Rewrite only if TITLE is blank or weak",
    )
    ap.add_argument(
        "--model", default="gpt-4.1-mini", help="OpenAI model (default: gpt-4.1-mini)"
    )
    ap.add_argument("--max-output-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--base-sleep", type=float, default=1.5)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files, just print proposed titles",
    )
    args = ap.parse_args()

    if OpenAI is None:
        print("ERROR: openai package not installed. Run: pip install -U openai")
        return 2

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Export it first, e.g.:")
        print("  export OPENAI_API_KEY='...'\n")
        return 2

    indir = Path(args.indir)
    if not indir.exists():
        print(f"ERROR: input folder not found: {indir}")
        return 2

    outdir = Path(args.outdir) if args.outdir else Path(str(indir) + "_titled_api")
    if not args.in_place:
        outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(indir.glob("*.txt"))
    if not files:
        print(f"No *.txt files found in: {indir}")
        return 0

    client = OpenAI()

    updated = 0
    skipped = 0
    errors = 0

    for i, p in enumerate(files, 1):
        raw = p.read_text(encoding="utf-8", errors="replace")
        try:
            title, url, vid, body = parse_normalized(raw)
        except Exception as e:
            print(f"[{i}/{len(files)}] SKIP (not normalized): {p.name} :: {e}")
            skipped += 1
            continue

        if args.only_blank_or_weak and not looks_weak(title):
            # keep as-is (useful for YouTube rips)
            if not args.in_place and not args.dry_run:
                write_normalized(outdir / p.name, title, url, vid, body)
            skipped += 1
            continue

        intro = first_two_paragraphs(body)
        if not intro:
            print(f"[{i}/{len(files)}] SKIP (empty transcript): {p.name}")
            skipped += 1
            continue

        # Retry with backoff on transient errors
        new_title: Optional[str] = None
        last_err: Optional[str] = None

        for attempt in range(args.retries + 1):
            try:
                new_title = call_title(
                    client=client,
                    intro=intro,
                    model=args.model,
                    max_output_tokens=args.max_output_tokens,
                    timeout=args.timeout,
                )
                break
            except Exception as e:
                last_err = str(e)
                sleep_s = args.base_sleep * (2**attempt)
                # cap backoff a bit
                sleep_s = min(sleep_s, 30)
                time.sleep(sleep_s)

        if not new_title:
            print(f"[{i}/{len(files)}] ERROR: {p.name}: {last_err}")
            errors += 1
            continue

        if args.dry_run:
            print(f"[{i}/{len(files)}] {p.name}\n  OLD: {title}\n  NEW: {new_title}\n")
            continue

        out_path = p if args.in_place else (outdir / p.name)
        write_normalized(out_path, new_title, url, vid, body)
        updated += 1

        if i % 10 == 0:
            print(f"Progress: {i}/{len(files)}")

    print(
        f"Done. Files: {len(files)} | Updated: {updated} | Skipped: {skipped} | Errors: {errors}"
    )
    print(f"Output: {indir if args.in_place else outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
