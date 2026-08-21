#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

REQ_HEADERS = ("TITLE:", "URL:", "VIDEO_ID:", "TRANSCRIPT:")

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def parse_video_id(text: str) -> str | None:
    m = re.search(r"^VIDEO_ID:\s*(.+)\s*$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None

def parse_title(text: str) -> str | None:
    m = re.search(r"^TITLE:\s*(.+)\s*$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None

def has_required_headers(text: str) -> bool:
    return all(h in text for h in REQ_HEADERS)

def replace_title(text: str, new_title: str) -> tuple[str, bool]:
    # Replace the first TITLE: line only
    pat = re.compile(r"^TITLE:\s*.*$", flags=re.MULTILINE)
    if not pat.search(text):
        return text, False
    out = pat.sub(f"TITLE: {new_title}", text, count=1)
    return out, True

def build_id_to_title(normalized_dir: Path) -> dict[str, str]:
    id_to_title: dict[str, str] = {}
    for p in sorted(normalized_dir.glob("*.txt")):
        raw = read_text(p)
        if not has_required_headers(raw):
            continue
        vid = parse_video_id(raw)
        title = parse_title(raw)
        if vid and title:
            id_to_title[vid] = title
    return id_to_title

def main() -> int:
    ap = argparse.ArgumentParser(description="Sync TITLE from normalized/*.txt into abstracts/*.txt by VIDEO_ID.")
    ap.add_argument("--normalized", required=True, help="Path to normalized folder (source of truth for TITLE)")
    ap.add_argument("--abstracts", required=True, help="Path to abstracts folder (where TITLE will be updated)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would change but do not write files")
    args = ap.parse_args()

    normalized_dir = Path(args.normalized).expanduser().resolve()
    abstracts_dir = Path(args.abstracts).expanduser().resolve()

    id_to_title = build_id_to_title(normalized_dir)
    if not id_to_title:
        print(f"No usable normalized files found in: {normalized_dir}")
        return 2

    updated = 0
    skipped = 0
    missing = 0

    for a in sorted(abstracts_dir.glob("*.txt")):
        t = read_text(a)
        vid = parse_video_id(t)
        if not vid:
            skipped += 1
            continue
        new_title = id_to_title.get(vid)
        if not new_title:
            missing += 1
            continue

        old_title = parse_title(t) or ""
        if old_title.strip() == new_title.strip():
            skipped += 1
            continue

        new_text, ok = replace_title(t, new_title)
        if not ok:
            skipped += 1
            continue

        updated += 1
        if args.dry_run:
            print(f"[DRY] {a.name}:")
            print(f"  - {old_title}")
            print(f"  + {new_title}")
        else:
            write_text(a, new_text)

    print(f"Done. Updated: {updated}, skipped: {skipped}, missing VIDEO_ID in normalized: {missing}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
