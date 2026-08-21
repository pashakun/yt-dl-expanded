#!/usr/bin/env python3
"""
make_abstracts.py

Generate (1) ABSTRACT and (2) KEY IDEAS for each normalized transcript file.

Input:  normalized/*.txt
Output: abstracts/*.abstract.txt

Resumable + rate-limit aware:
- Skips outputs that already exist
- Retries 429/5xx with backoff + jitter
- Parses "Please try again in Xs" and sleeps X seconds
- If "Request too large for ... tokens per min" occurs, falls back to chunk->notes->final

Requires:
  pip install openai

Env:
  export OPENAI_API_KEY="..."

Run:
  python3 make_abstracts.py
  python3 make_abstracts.py --model gpt-4.1 --sleep 1.0
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv("config.env")

from openai import OpenAI
from openai._exceptions import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

# ----------------------------
# Config / prompts
# ----------------------------

TASK_INSTRUCTIONS = """You are analyzing a transcript of a long-form crypto / financial infrastructure discussion.

Your task has two parts:

PART 1 — ABSTRACT
Write a single, tight paragraph (4–6 sentences) that explains:
- What this conversation is fundamentally about
- Why it matters for the future of crypto, finance, or institutions
- What questions or tensions remain unresolved

Do not quote speakers.
Do not mention timestamps.
Do not use promotional language.

PART 2 — KEY IDEAS
Extract 5–10 concise bullet points capturing:
- Claims or arguments made
- Predictions about the future
- Risks, frictions, or tradeoffs discussed
- Institutional implications (explicit or implicit)

Do not summarize the episode chronologically.
Do not repeat the abstract in bullet form.

Be precise. Favor substance over completeness.
"""

NOTES_INSTRUCTIONS = """You are extracting compact analytical notes from a transcript.

Produce:
- 12–20 bullets of key claims, tensions, risks, predictions, and institutional implications.
- No quotes. No timestamps. No promo language.
- Keep it dense and information-rich.
"""

FINAL_FROM_NOTES_INSTRUCTIONS = """You are given compact notes from a long-form crypto / financial infrastructure discussion.

Write:

PART 1 — ABSTRACT
A single paragraph (4–6 sentences) covering what it's fundamentally about, why it matters for crypto/finance/institutions, and unresolved tensions.

PART 2 — KEY IDEAS
5–10 bullets: claims, predictions, risks/frictions/tradeoffs, institutional implications.

No quotes. No timestamps. No promo language. Not chronological. Do not repeat abstract as bullets.
"""

# ----------------------------
# Helpers
# ----------------------------


@dataclass
class ParsedTranscript:
    title: str
    url: str
    video_id: str
    transcript: str


TITLE_RE = re.compile(r"^TITLE:\s*(.*)\s*$", re.MULTILINE)
URL_RE = re.compile(r"^URL:\s*(.*)\s*$", re.MULTILINE)
VID_RE = re.compile(r"^VIDEO_ID:\s*([A-Za-z0-9_-]+)\s*$", re.MULTILINE)
TRANSCRIPT_RE = re.compile(r"^TRANSCRIPT:\s*(.*)\s*$", re.MULTILINE | re.DOTALL)

RETRY_AFTER_RE = re.compile(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
TPM_TOO_LARGE_RE = re.compile(r"Request too large.*tokens per min", re.IGNORECASE)


def parse_normalized_file(text: str) -> ParsedTranscript:
    title_m = TITLE_RE.search(text)
    url_m = URL_RE.search(text)
    vid_m = VID_RE.search(text)
    tr_m = TRANSCRIPT_RE.search(text)

    if not (title_m and url_m and vid_m and tr_m):
        raise ValueError(
            "Normalized file missing required headers (TITLE/URL/VIDEO_ID/TRANSCRIPT)."
        )

    return ParsedTranscript(
        title=title_m.group(1).strip(),
        url = (url_m.group(1).strip() if url_m else ""),
        video_id=vid_m.group(1).strip(),
        transcript=tr_m.group(1).strip(),
    )


def approx_tokens(s: str) -> int:
    # Rough heuristic: ~4 chars/token in English-ish text.
    return max(1, len(s) // 4)


def chunk_text(text: str, target_tokens: int = 3500) -> List[str]:
    # Simple paragraph-ish chunker with token heuristic.
    parts = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    buf: List[str] = []
    buf_tok = 0

    for p in parts:
        p = p.strip()
        if not p:
            continue
        t = approx_tokens(p) + 1
        if buf_tok + t > target_tokens and buf:
            chunks.append("\n\n".join(buf))
            buf = [p]
            buf_tok = t
        else:
            buf.append(p)
            buf_tok += t

    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def extract_sleep_from_error(msg: str) -> Optional[float]:
    m = RETRY_AFTER_RE.search(msg)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# ----------------------------
# OpenAI call wrappers
# ----------------------------


def call_responses(
    client: OpenAI,
    model: str,
    instructions: str,
    input_text: str,
    max_output_tokens: int,
    timeout_s: float,
) -> str:
    # Using Responses API (current standard in docs).
    resp = client.with_options(timeout=timeout_s).responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
        max_output_tokens=max_output_tokens,
    )
    return resp.output_text


def call_with_retries(
    fn,
    *,
    max_retries: int,
    base_sleep: float,
    jitter: float,
    sleep_floor: float,
) -> str:
    attempt = 0
    while True:
        try:
            return fn()
        except RateLimitError as e:
            msg = str(e)
            # If API tells us exactly how long to wait, do that.
            exact = extract_sleep_from_error(msg)
            if exact is None:
                # Exponential backoff with jitter
                exact = max(
                    sleep_floor, base_sleep * (2**attempt) + random.uniform(0, jitter)
                )
            if attempt >= max_retries:
                raise
            time.sleep(exact)
            attempt += 1
        except (APITimeoutError, APIConnectionError, InternalServerError) as e:
            # Transient: retry with backoff
            if attempt >= max_retries:
                raise
            delay = max(
                sleep_floor, base_sleep * (2**attempt) + random.uniform(0, jitter)
            )
            time.sleep(delay)
            attempt += 1


# ----------------------------
# Core processing
# ----------------------------


def build_input_block(p: ParsedTranscript) -> str:
    return f"""TITLE: {p.title}
URL: {p.url}
VIDEO_ID: {p.video_id}

TRANSCRIPT:
{p.transcript}
"""


def build_notes_input_block(
    p: ParsedTranscript, chunk: str, chunk_idx: int, total: int
) -> str:
    return f"""TITLE: {p.title}
URL: {p.url}
VIDEO_ID: {p.video_id}
CHUNK: {chunk_idx}/{total}

TRANSCRIPT_CHUNK:
{chunk}
"""


def make_notes(
    client: OpenAI,
    model: str,
    p: ParsedTranscript,
    max_output_tokens: int,
    timeout_s: float,
    max_retries: int,
    base_sleep: float,
) -> str:
    chunks = chunk_text(p.transcript, target_tokens=3500)
    notes_parts: List[str] = []

    for i, ch in enumerate(chunks, start=1):
        input_text = build_notes_input_block(p, ch, i, len(chunks))

        def _do():
            return call_responses(
                client=client,
                model=model,
                instructions=NOTES_INSTRUCTIONS,
                input_text=input_text,
                max_output_tokens=max_output_tokens,
                timeout_s=timeout_s,
            )

        out = call_with_retries(
            _do,
            max_retries=max_retries,
            base_sleep=base_sleep,
            jitter=1.5,
            sleep_floor=0.8,
        )
        notes_parts.append(out.strip())

    # Merge notes (still compact compared to full transcript)
    return "\n\n".join(notes_parts).strip()


def make_final_from_notes(
    client: OpenAI,
    model: str,
    p: ParsedTranscript,
    notes: str,
    max_output_tokens: int,
    timeout_s: float,
    max_retries: int,
    base_sleep: float,
) -> str:
    input_text = f"""TITLE: {p.title}
URL: {p.url}
VIDEO_ID: {p.video_id}

NOTES:
{notes}
"""

    def _do():
        return call_responses(
            client=client,
            model=model,
            instructions=FINAL_FROM_NOTES_INSTRUCTIONS,
            input_text=input_text,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
        )

    return call_with_retries(
        _do,
        max_retries=max_retries,
        base_sleep=base_sleep,
        jitter=1.5,
        sleep_floor=0.8,
    ).strip()


def process_one(
    client: OpenAI,
    model: str,
    infile: Path,
    outdir: Path,
    *,
    sleep_between: float,
    max_output_tokens: int,
    timeout_s: float,
    max_retries: int,
    base_sleep: float,
) -> Tuple[bool, Optional[str]]:
    """
    Returns: (ok, error_message)
    """
    raw = infile.read_text(encoding="utf-8", errors="replace")
    p = parse_normalized_file(raw)

    # Output naming: preserve base name, swap suffix
    base = infile.name
    # Your inputs are like "...[id].wav.txt" so keep that and append .abstract.txt
    outfile = outdir / base.replace(".txt", ".abstract.txt")
    errfile = outdir / base.replace(".txt", ".error.txt")

    if outfile.exists():
        return True, None

    input_block = build_input_block(p)

    def _direct():
        return call_responses(
            client=client,
            model=model,
            instructions=TASK_INSTRUCTIONS,
            input_text=input_block,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
        )

    try:
        out = call_with_retries(
            _direct,
            max_retries=max_retries,
            base_sleep=base_sleep,
            jitter=1.5,
            sleep_floor=0.8,
        ).strip()

        outfile.write_text(out + "\n", encoding="utf-8")
        if errfile.exists():
            errfile.unlink(missing_ok=True)
        if sleep_between > 0:
            time.sleep(sleep_between)
        return True, None

    except RateLimitError as e:
        # If it's the special "Request too large for TPM" case, do chunk->notes->final.
        msg = str(e)
        if TPM_TOO_LARGE_RE.search(msg):
            try:
                notes = make_notes(
                    client,
                    model,
                    p,
                    max_output_tokens=max(700, max_output_tokens // 2),
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                    base_sleep=base_sleep,
                )
                final = make_final_from_notes(
                    client,
                    model,
                    p,
                    notes,
                    max_output_tokens=max_output_tokens,
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                    base_sleep=base_sleep,
                )
                outfile.write_text(final + "\n", encoding="utf-8")
                if errfile.exists():
                    errfile.unlink(missing_ok=True)
                if sleep_between > 0:
                    time.sleep(sleep_between)
                return True, None
            except Exception as e2:
                errfile.write_text(
                    f"ERROR: {type(e2).__name__}: {e2}\n", encoding="utf-8"
                )
                return False, f"{type(e2).__name__}: {e2}"

        errfile.write_text(f"ERROR: {type(e).__name__}: {e}\n", encoding="utf-8")
        return False, f"{type(e).__name__}: {e}"

    except (
        BadRequestError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    ) as e:
        errfile.write_text(f"ERROR: {type(e).__name__}: {e}\n", encoding="utf-8")
        return False, f"{type(e).__name__}: {e}"

    except Exception as e:
        errfile.write_text(f"ERROR: {type(e).__name__}: {e}\n", encoding="utf-8")
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="indir",
        default="normalized",
        help="Input folder (default: normalized)",
    )
    ap.add_argument(
        "--out",
        dest="outdir",
        default="abstracts",
        help="Output folder (default: abstracts)",
    )
    ap.add_argument("--model", default="gpt-4.1", help="Model name (default: gpt-4.1)")
    ap.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Sleep between successful requests (seconds)",
    )
    ap.add_argument(
        "--max-output-tokens", type=int, default=700, help="Max output tokens per file"
    )
    ap.add_argument(
        "--timeout", type=float, default=120.0, help="Per-request timeout in seconds"
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Max retries for transient/rate-limit errors",
    )
    ap.add_argument(
        "--base-sleep", type=float, default=1.5, help="Base backoff sleep seconds"
    )

    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in indir.glob("*.txt") if p.is_file()])
    if not files:
        print(f"No .txt files found in {indir.resolve()}", file=sys.stderr)
        return 2

    client = OpenAI()

    total = len(files)
    ok_count = 0
    err_count = 0

    for idx, f in enumerate(files, start=1):
        print(f"[{idx}/{total}] Processing: {f.name}")
        ok, err = process_one(
            client=client,
            model=args.model,
            infile=f,
            outdir=outdir,
            sleep_between=args.sleep,
            max_output_tokens=args.max_output_tokens,
            timeout_s=args.timeout,
            max_retries=args.retries,
            base_sleep=args.base_sleep,
        )
        if ok:
            ok_count += 1
        else:
            err_count += 1
            print(f"  ERROR: {err}")

    print(f"\nDone. Outputs in: {outdir}")
    print(f"Success: {ok_count}")
    print(f"Errors:  {err_count} (see *.error.txt files)")
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
