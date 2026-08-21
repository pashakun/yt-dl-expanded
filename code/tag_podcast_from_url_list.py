#!/usr/bin/env python3
# tag_podcast_from_url_list.py
#
# Adds a podcast prefix to TITLE: lines in normalized transcripts,
# but ONLY for files whose VIDEO_ID matches a provided URL list.
#
# Safe defaults:
# - dry-run prints what would change
# - in-place edits only with --in-place
# - creates a .bak copy per edited file unless --no-backup

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs

VIDEO_ID_RE = re.compile(r"\[([A-Za-z0-9_-]{6,})\]")  # fallback: titles like "... [abc123].wav.txt"
HEADER_RE = re.compile(r"^(TITLE|URL|VIDEO_ID|TRANSCRIPT):\s*(.*)$")

def extract_video_id(s: str) -> str | None:
    s = s.strip()
    if not s or s.startswith("#"):
        return None

    # If it's already an ID
    if re.fullmatch(r"[A-Za-z0-9_-]{6,}", s):
        return s

    # Try URL parse
    try:
        u = urlparse(s)
    except Exception:
        return None

    host = (u.netloc or "").lower()
    path = u.path or ""

    # youtu.be/<id>
    if "youtu.be" in host:
        vid = path.strip("/").split("/")[0]
        return vid or None

    # youtube.com/watch?v=<id>
    if "youtube.com" in host:
        qs = parse_qs(u.query or "")
        if "v" in qs and qs["v"]:
            return qs["v"][0]
        # /shorts/<id>
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed"}:
            return parts[1]

    return None

def load_video_ids(url_list_path: Path) -> set[str]:
    ids: set[str] = set()
    for line in url_list_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        vid = extract_video_id(line)
        if vid:
            ids.add(vid)
    return ids

def parse_headers(text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in text.splitlines():
        m = HEADER_RE.match(line)
        if m:
            headers[m.group(1)] = m.group(2).strip()
        # Stop early once transcript begins (optional)
        if line.startswith("TRANSCRIPT:"):
            # still keep TRANSCRIPT header value captured above
            break
    return headers

def rewrite_title(text: str, prefix: str) -> tuple[str, str, str] | None:
    """
    Returns (new_text, old_title, new_title) if changed, else None.
    """
    lines = text.splitlines()
    new_lines = []
    old_title = None
    new_title = None
    changed = False

    for line in lines:
        if line.startswith("TITLE:"):
            old_title = line[len("TITLE:"):].strip()
            if old_title.lower().startswith(prefix.lower() + ":"):
                new_lines.append(line)
            else:
                new_title = f"{prefix}: {old_title}" if old_title else f"{prefix}"
                new_lines.append(f"TITLE: {new_title}")
                changed = True
        else:
            new_lines.append(line)

    if changed and old_title is not None and new_title is not None:
        return ("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""), old_title, new_title)
    return None

def maybe_match_file(path: Path, target_ids: set[str]) -> str | None:
    """
    Returns matched video_id if file corresponds to target, else None.
    Matches in this order:
      1) VIDEO_ID header
      2) URL header -> extract id
      3) filename contains [id]
    """
    txt = path.read_text(encoding="utf-8", errors="ignore")
    h = parse_headers(txt)

    vid = h.get("VIDEO_ID")
    if vid and vid in target_ids:
        return vid

    url = h.get("URL")
    if url:
        extracted = extract_video_id(url)
        if extracted and extracted in target_ids:
            return extracted

    m = VIDEO_ID_RE.search(path.name)
    if m and m.group(1) in target_ids:
        return m.group(1)

    return None

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized", required=True, help="Path to normalized folder (e.g., normalized/)")
    ap.add_argument("--url-list", required=True, help="Path to txt file with URLs or video IDs (one per line)")
    ap.add_argument("--prefix", default="The Future of Money", help="Prefix to add to TITLE")
    ap.add_argument("--in-place", action="store_true", help="Actually edit files (otherwise dry-run)")
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak backups when editing")
    ap.add_argument("--only-if-missing", action="store_true",
                    help="Only add prefix if TITLE does not already contain the prefix")
    args = ap.parse_args()

    normalized_dir = Path(args.normalized).expanduser().resolve()
    url_list_path = Path(args.url_list).expanduser().resolve()

    if not normalized_dir.exists():
        raise SystemExit(f"Normalized dir not found: {normalized_dir}")
    if not url_list_path.exists():
        raise SystemExit(f"URL list not found: {url_list_path}")

    target_ids = load_video_ids(url_list_path)
    if not target_ids:
        raise SystemExit("No video IDs found in the URL list. Check formatting.")

    txt_files = sorted(p for p in normalized_dir.glob("*.txt") if p.is_file())

    hits = 0
    changes = 0

    for p in txt_files:
        matched = maybe_match_file(p, target_ids)
        if not matched:
            continue

        hits += 1
        raw = p.read_text(encoding="utf-8", errors="ignore")

        # Optional: if only-if-missing, skip if already tagged
        if args.only_if_missing:
            h = parse_headers(raw)
            t = (h.get("TITLE") or "").lower()
            if t.startswith(args.prefix.lower() + ":"):
                continue

        rewritten = rewrite_title(raw, args.prefix)
        if not rewritten:
            continue

        new_text, old_title, new_title = rewritten
        print(f"[MATCH {matched}] {p.name}")
        print(f"  TITLE: {old_title}")
        print(f"     -> {new_title}")

        if args.in_place:
            if not args.no_backup:
                shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
            p.write_text(new_text, encoding="utf-8")
            changes += 1

    print("\nSummary")
    print(f"  Target video IDs: {len(target_ids)}")
    print(f"  Matched files:    {hits}")
    print(f"  Updated titles:   {changes}" if args.in_place else f"  Would update:     {hits} (preview mode)")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())