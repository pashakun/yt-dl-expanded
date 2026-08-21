#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_titles.py

Post-processing step to improve TITLE: in normalized transcript files
by extracting topic/guest/org from the intro portion of TRANSCRIPT:.

Design goals:
- Avoid overfitting to YouTube formats.
- Prefer "topic-first" titles (what it's about), then guest, then show tag.
- Only change titles that look weak/filename-ish unless --force.

Examples targeted:
- "Hello and welcome back to On The Brink... sitting down with Chris Maurice...
   founder of Yellowcard... stablecoin payments... Africa"
  -> "Yellowcard and Stablecoin Payments in Africa (Chris Maurice) [On The Brink]"

- "So I'm sitting here with Mike Lampras, chairman of Silvergate..."
  -> "Silvergate and Crypto Banking (Mike Lampras)"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

HEADER_RE = re.compile(
    r"^TITLE:\s*(?P<title>.*)\n"
    r"URL:\s*(?P<url>.*)\n"
    r"VIDEO_ID:\s*(?P<vid>.*)\n"
    r"\nTRANSCRIPT:\n(?P<body>.*)$",
    re.DOTALL,
)

# "Hello and welcome back to On The Brink."
SHOW_RE = re.compile(
    r"^\s*(?:hello and welcome(?: back)? to)\s+(.+?)[\.\!]\s*$",
    re.IGNORECASE,
)

# "This is Nick Carter."
HOST_RE = re.compile(
    r"^\s*(?:this is|i['’]m)\s+([A-Z][A-Za-z\.\- ]{2,60})[\.\!]\s*$",
    re.IGNORECASE,
)

# Guest detection (broad)
GUEST_PATTERNS = [
    re.compile(
        r"\b(?:today[, ]+)?i['’]m sitting down with\s+([^,.\n]+)", re.IGNORECASE
    ),
    re.compile(r"\bso i['’]m sitting here with\s+([^,.\n]+)", re.IGNORECASE),
    re.compile(r"\bmy guest today is\s+([^,.\n]+)", re.IGNORECASE),
    re.compile(r"\bjoined by\s+([^,.\n]+)", re.IGNORECASE),
    re.compile(r"\bi['’]m here with\s+([^,.\n]+)", re.IGNORECASE),
]

# Topic detection (explicit)
TOPIC_PATTERNS = [
    re.compile(r"\bto talk about\s+(.+?)[\.\!]", re.IGNORECASE),
    re.compile(r"\bwe talk about\s+(.+?)[\.\!]", re.IGNORECASE),
    re.compile(r"\bto discuss\s+(.+?)[\.\!]", re.IGNORECASE),
    re.compile(r"\bto unpack\s+(.+?)[\.\!]", re.IGNORECASE),
    re.compile(r"\bto dig into\s+(.+?)[\.\!]", re.IGNORECASE),
    re.compile(r"\bto break down\s+(.+?)[\.\!]", re.IGNORECASE),
]

# Org/role/topic inference from guest intro lines (implicit)
ROLE_ORG_PATTERNS = [
    # "founder of Yellowcard"
    re.compile(
        r"\b(founder|co-?founder|ceo|chief executive|president|chairman|chair|cto|cfo|cio|coo|head)\s+of\s+([^,.\n]+)",
        re.IGNORECASE,
    ),
    # "the founder, Chris Maurice, of Yellowcard"
    re.compile(
        r"\b(?:the\s+)?(founder|co-?founder|ceo|chairman|chair|cto|cfo|cio|coo|president|head)\b[^,\n]{0,40}\bof\s+([^,.\n]+)",
        re.IGNORECASE,
    ),
    # "Chris Maurice, the founder of Yellowcard, Africa's premier..."
    re.compile(
        r"\b([^,\n]+),\s+the\s+(founder|co-?founder|ceo|chairman|chair|cto|cfo|cio|coo|president|head)\s+of\s+([^,.\n]+)",
        re.IGNORECASE,
    ),
]

# Useful topical keywords to harvest if present in the intro
KEYPHRASE_PATTERNS = [
    re.compile(r"\bstablecoin(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bpayments?\b", re.IGNORECASE),
    re.compile(r"\bexchange\b", re.IGNORECASE),
    re.compile(r"\btreasury\b", re.IGNORECASE),
    re.compile(r"\bcompliance\b", re.IGNORECASE),
    re.compile(r"\bregulation\b", re.IGNORECASE),
    re.compile(r"\bcustody\b", re.IGNORECASE),
    re.compile(r"\bbanking\b", re.IGNORECASE),
    re.compile(r"\bcrypto\b", re.IGNORECASE),
    re.compile(r"\bafrica\b", re.IGNORECASE),
    re.compile(r"\bemerging markets?\b", re.IGNORECASE),
]


def read_normalized(path: Path) -> Tuple[str, str, str, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    m = HEADER_RE.match(raw.strip())
    if not m:
        raise ValueError(f"Not a normalized file (missing headers): {path.name}")
    return (
        m.group("title").strip(),
        m.group("url").strip(),
        m.group("vid").strip(),
        m.group("body").strip(),
    )


def write_normalized(path: Path, title: str, url: str, vid: str, body: str) -> None:
    txt = f"TITLE: {title}\nURL: {url}\nVIDEO_ID: {vid}\n\nTRANSCRIPT:\n{body}\n"
    path.write_text(txt, encoding="utf-8")


def clip(s: str, n: int = 160) -> str:
    s = " ".join(s.strip().split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def looks_like_filename_title(title: str) -> bool:
    if not title:
        return True
    t = title.strip()
    if len(t) < 12:
        return True
    if re.search(r"\b(AUDIO|FINAL|V\d+|mixdown)\b", t, re.IGNORECASE):
        return True
    if "_" in t and " " not in t:
        return True
    if re.search(r"[_\-]{2,}", t):
        return True
    return False


def fallback_from_filename(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"\s*\[[^\]]+\]\s*$", "", stem)
    stem = stem.replace("_", " ").strip()
    return clip(stem, 140) if stem else "Untitled"


def clean_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^(a|an|the)\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_show_host(lines: list[str]) -> Tuple[str, str]:
    show = ""
    host = ""
    for ln in lines[:12]:
        m = SHOW_RE.match(ln)
        if m:
            show = clean_name(m.group(1))
            break
    for ln in lines[:20]:
        m = HOST_RE.match(ln)
        if m:
            host = clean_name(m.group(1))
            break
    return show, host


def extract_guest(head_text: str) -> str:
    for pat in GUEST_PATTERNS:
        m = pat.search(head_text)
        if m:
            return clean_name(m.group(1))
    return ""


def extract_explicit_topic(head_text: str) -> str:
    for pat in TOPIC_PATTERNS:
        m = pat.search(head_text)
        if m:
            return clean_name(m.group(1).rstrip(" .!"))
    return ""


def extract_role_org(head_text: str) -> Tuple[str, str]:
    # Returns (role, org)
    for pat in ROLE_ORG_PATTERNS:
        m = pat.search(head_text)
        if not m:
            continue
        if len(m.groups()) == 2:
            role = clean_name(m.group(1))
            org = clean_name(m.group(2))
            return role, org
        if len(m.groups()) == 3:
            # pattern includes guest name then role then org
            role = clean_name(m.group(2))
            org = clean_name(m.group(3))
            return role, org
    return "", ""


def extract_keyphrases(head_text: str) -> list[str]:
    found = []
    for pat in KEYPHRASE_PATTERNS:
        if pat.search(head_text):
            # Use canonical-ish label
            token = (
                pat.pattern.strip("\\b").strip("()").replace("?:", "").replace("\\", "")
            )
            token = re.sub(r"\(\?:.*\)", "", token)
            token = token.replace("(?:s)?", "s").replace("?", "")
            token = token.replace("payments?", "payments")
            token = token.replace("stablecoin(?:s)?", "stablecoins")
            token = token.replace("emerging markets?", "emerging markets")
            token = token.replace("markets?", "markets")
            token = token.replace("africa", "Africa")
            token = token.replace("crypto", "crypto")
            token = token.replace("banking", "banking")
            token = token.replace("exchange", "exchange")
            token = token.replace("compliance", "compliance")
            token = token.replace("regulation", "regulation")
            token = token.replace("custody", "custody")
            token = token.replace("treasury", "treasury")
            token = token.strip()
            found.append(token)
    # Deduplicate while preserving order
    out = []
    for x in found:
        if x not in out:
            out.append(x)
    return out


def title_from_components(
    show: str,
    guest: str,
    role: str,
    org: str,
    explicit_topic: str,
    keyphrases: list[str],
) -> str:
    """
    Prefer topic-first:
      1) explicit topic if decent
      2) org + inferred topical cues
      3) org only
      4) guest only
    Then append guest, then [show] if present.
    """
    topic = ""

    # If explicit topic exists and isn't just vague filler
    if explicit_topic and len(explicit_topic.split()) >= 4:
        topic = explicit_topic

    # Otherwise infer from org + keyphrases
    if not topic and org:
        # Build a compact topic like "Yellowcard and Stablecoin Payments in Africa"
        # If both stablecoins and payments appear, combine.
        has_stable = any(k.lower() == "stablecoins" for k in keyphrases)
        has_pay = any(k.lower() == "payments" for k in keyphrases)
        has_africa = any(k == "Africa" for k in keyphrases)

        parts = []
        if org:
            parts.append(org)

        # Add one topical clause based on keyphrase signals
        clause = ""
        if has_stable and has_pay and has_africa:
            clause = "Stablecoin Payments in Africa"
        elif has_stable and has_pay:
            clause = "Stablecoin Payments"
        elif has_stable and has_africa:
            clause = "Stablecoins in Africa"
        elif has_stable:
            clause = "Stablecoins"
        elif "banking" in [k.lower() for k in keyphrases]:
            clause = "Crypto Banking"
        elif "regulation" in [k.lower() for k in keyphrases] or "compliance" in [
            k.lower() for k in keyphrases
        ]:
            clause = "Regulation and Compliance"
        elif "custody" in [k.lower() for k in keyphrases]:
            clause = "Custody and Risk"
        elif "exchange" in [k.lower() for k in keyphrases]:
            clause = "Exchange Operations"
        elif "treasury" in [k.lower() for k in keyphrases]:
            clause = "Treasury Operations"

        if clause:
            topic = f"{parts[0]} and {clause}" if parts else clause
        else:
            topic = parts[0]

    # If still nothing, org from role/org extraction
    if not topic and org:
        topic = org

    # If still nothing, use role
    if not topic and role:
        topic = role.title()

    # Last resort: guest
    if not topic and guest:
        topic = guest

    # Compose final
    base = topic
    if guest and guest.lower() not in base.lower():
        base = f"{base} ({guest})"

    if show:
        base = f"{base} [{show}]"

    return clip(base, 160)


def infer_title_from_intro(transcript: str, fallback: str) -> str:
    if not transcript:
        return fallback

    lines = [ln.strip() for ln in transcript.splitlines() if ln.strip()]
    head = lines[:120]  # scan deeper; intros can be longer
    head_text = "\n".join(head)

    show, host = extract_show_host(head)
    guest = extract_guest(head_text)
    explicit_topic = extract_explicit_topic(head_text)
    role, org = extract_role_org(head_text)
    keyphrases = extract_keyphrases(head_text)

    new_title = title_from_components(
        show=show,
        guest=guest,
        role=role,
        org=org,
        explicit_topic=explicit_topic,
        keyphrases=keyphrases,
    )

    # If the new title is somehow worse than fallback (super short), keep fallback
    if len(new_title) < 12:
        return fallback
    return new_title


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="indir", required=True, help="Folder with normalized *.txt files"
    )
    ap.add_argument(
        "--out", dest="outdir", required=True, help="Output folder (recommended)"
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite files in --in instead of writing to --out",
    )
    ap.add_argument(
        "--only-weak",
        action="store_true",
        default=True,
        help="Only rewrite titles that look weak/filename-ish (default true).",
    )
    ap.add_argument(
        "--force", action="store_true", help="Rewrite titles even if they look fine."
    )
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)

    if not indir.exists():
        raise SystemExit(f"Input folder not found: {indir}")

    if not args.in_place:
        outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(indir.glob("*.txt"))
    if not files:
        print(f"No *.txt files found in: {indir}")
        return 0

    changed = 0
    skipped = 0

    for p in files:
        try:
            title, url, vid, body = read_normalized(p)
        except Exception as e:
            print(f"SKIP (not normalized): {p.name} :: {e}")
            skipped += 1
            continue

        should_rewrite = args.force or (
            args.only_weak and looks_like_filename_title(title)
        )
        out_path = p if args.in_place else (outdir / p.name)

        if not should_rewrite:
            if not args.in_place:
                write_normalized(out_path, title, url, vid, body)
            continue

        fallback = title if title else fallback_from_filename(p)
        new_title = infer_title_from_intro(body, fallback=fallback)

        write_normalized(out_path, new_title, url, vid, body)
        if new_title != title:
            changed += 1

    print(f"Done. Files: {len(files)} | Updated titles: {changed} | Skipped: {skipped}")
    print(f"Output: {indir if args.in_place else outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
