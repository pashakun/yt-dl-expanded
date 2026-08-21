#!/usr/bin/env python3
"""
build_corpus.py

Build a single corpus markdown file from per-episode abstract files.

Features:
- Honors --in and --out (no hardcoded paths).
- Supports abstracts in .md and/or .json.
- Dedupes by VIDEO_ID (prefers .md over .json when both exist).
- Preserves TITLE verbatim (so prefixes like "The Future of Money:" survive).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Entry:
    video_id: str
    title: str
    url: str
    source_path: Path
    body_md: str  # abstract content (already markdown-ish)


MD_TITLE_RE = re.compile(r"^TITLE:\s*(.+?)\s*$", re.MULTILINE)
MD_URL_RE = re.compile(r"^URL:\s*(.*?)\s*$", re.MULTILINE)
MD_VIDEO_RE = re.compile(r"^VIDEO_ID:\s*(.+?)\s*$", re.MULTILINE)
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def parse_md(path: Path) -> Optional[Entry]:
    raw = read_text(path)

    # VIDEO_ID (required)
    m_vid = MD_VIDEO_RE.search(raw)
    if not m_vid:
        return None
    video_id = m_vid.group(1).strip()

    # TITLE: (preferred) else first H1
    m_title = MD_TITLE_RE.search(raw)
    if m_title:
        title = m_title.group(1).strip()
    else:
        m_h1 = H1_RE.search(raw)
        title = m_h1.group(1).strip() if m_h1 else path.stem

    # URL: (optional)
    m_url = MD_URL_RE.search(raw)
    url = m_url.group(1).strip() if m_url else ""

    # Body: keep everything after the TRANSCRIPT header if present,
    # otherwise keep the whole file but strip the front-matter-ish headers.
    # This tries to keep your abstract sections intact.
    body = raw

    # If file contains "PART 1 — ABSTRACT" style, keep from there.
    idx = body.find("PART 1")
    if idx != -1:
        body_md = body[idx:].strip()
    else:
        # Otherwise remove the TITLE/URL/VIDEO_ID header block and keep rest.
        body_md = re.sub(
            r"^(TITLE|URL|VIDEO_ID):.*\n", "", body, flags=re.MULTILINE
        ).strip()

    return Entry(
        video_id=video_id,
        title=title,
        url=url,
        source_path=path,
        body_md=body_md,
    )


def parse_json(path: Path) -> Optional[Entry]:
    try:
        obj = json.loads(read_text(path))
    except Exception:
        return None

    # Try common key names
    video_id = (
        obj.get("video_id") or obj.get("VIDEO_ID") or obj.get("id") or ""
    ).strip()
    if not video_id:
        return None

    title = (obj.get("title") or obj.get("TITLE") or "").strip() or path.stem
    url = (obj.get("url") or obj.get("URL") or "").strip()

    # Abstract content: prefer a markdown field if present, else stitch from parts.
    body_md = ""
    for k in ("abstract_md", "abstract_markdown", "markdown", "md"):
        if isinstance(obj.get(k), str) and obj.get(k).strip():
            body_md = obj[k].strip()
            break

    if not body_md:
        # Fall back to a generic dump of fields if no markdown body exists
        if isinstance(obj.get("abstract"), str):
            body_md = obj["abstract"].strip()
        elif isinstance(obj.get("parts"), list):
            body_md = "\n\n".join(str(x) for x in obj["parts"]).strip()
        else:
            # last resort: pretty json
            body_md = (
                "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"
            )

    return Entry(
        video_id=video_id,
        title=title,
        url=url,
        source_path=path,
        body_md=body_md,
    )


def find_abstract_files(indir: Path) -> List[Path]:
    # Only look in the given directory (not hardcoded), non-recursive by default.
    # If you want recursive later, change glob to rglob.
    files = []
    files.extend(sorted(indir.glob("*.md")))
    files.extend(sorted(indir.glob("*.json")))
    return files


def load_entries(indir: Path) -> List[Entry]:
    files = find_abstract_files(indir)

    # Deduplicate by VIDEO_ID, prefer .md over .json
    chosen: Dict[str, Entry] = {}

    for f in files:
        entry: Optional[Entry]
        if f.suffix.lower() == ".md":
            entry = parse_md(f)
        else:
            entry = parse_json(f)

        if not entry:
            continue

        prev = chosen.get(entry.video_id)
        if not prev:
            chosen[entry.video_id] = entry
            continue

        # Prefer .md; if same type, keep the newer file
        if (
            prev.source_path.suffix.lower() != ".md"
            and entry.source_path.suffix.lower() == ".md"
        ):
            chosen[entry.video_id] = entry
        else:
            try:
                if f.stat().st_mtime > prev.source_path.stat().st_mtime:
                    chosen[entry.video_id] = entry
            except Exception:
                pass

    # Stable ordering: by title then video_id
    out = list(chosen.values())
    out.sort(key=lambda e: (e.title.lower(), e.video_id.lower()))
    return out


def render_entry(e: Entry) -> str:
    lines = []
    lines.append(f"## {e.title} [{e.video_id}]")
    if e.url:
        lines.append(f"- URL: {e.url}")
    lines.append(f"- VIDEO_ID: {e.video_id}")
    lines.append("")
    lines.append(e.body_md.strip())
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="indir",
        required=True,
        help="Directory containing abstract .md/.json files",
    )
    ap.add_argument(
        "--out", dest="outpath", required=True, help="Output corpus markdown path"
    )
    args = ap.parse_args()

    indir = Path(args.indir).expanduser().resolve()
    outpath = Path(args.outpath).expanduser().resolve()

    if not indir.exists() or not indir.is_dir():
        print(f"ERROR: --in directory not found: {indir}")
        return 2

    entries = load_entries(indir)
    print(f"Found {len(entries)} abstract entries in {indir}")

    outpath.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "# Corpus: Episode Abstracts",
        "",
        f"- Source folder: `{indir}`",
        f"- Entries: {len(entries)}",
        "",
        "---",
        "",
    ]

    parts = ["\n".join(header)]
    for e in entries:
        parts.append(render_entry(e))
        parts.append("\n---\n")

    outpath.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
    print(f"Wrote corpus: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
