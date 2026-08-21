#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PREFIX = "The Future of Money:"

TITLE_RE = re.compile(r"^TITLE:\s*(.*?)\s*$", re.M)
VID_RE   = re.compile(r"^VIDEO_ID:\s*(.*?)\s*$", re.M)

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def parse_headers(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = TITLE_RE.search(text)
    v = VID_RE.search(text)
    title = t.group(1).strip() if t else None
    vid = v.group(1).strip() if v else None
    return title, vid

def normalize_title_for_match(s: str) -> str:
    # Fallback matching only (if VIDEO_ID is missing somewhere)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[“”\"']", "", s)
    s = re.sub(r"[:：]\s*", ": ", s)
    return s

def update_md_title(md_path: Path, dry_run: bool) -> bool:
    raw = read_text(md_path)
    m = TITLE_RE.search(raw)
    if not m:
        return False

    old = m.group(1).strip()
    if old.startswith(PREFIX):
        return False

    new = f"{PREFIX} {old}"
    updated = raw[:m.start(1)] + new + raw[m.end(1):]

    if dry_run:
        print(f"MD  UPDATE: {md_path.name}\n  TITLE: {old}\n   ->   {new}")
        return True

    md_path.write_text(updated, encoding="utf-8")
    print(f"MD  UPDATED: {md_path.name}")
    return True

def get_json_title_ref(obj: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (container_dict, key) where obj[ key ] holds title.
    Supports a few shapes we've seen in the wild.
    """
    if isinstance(obj, dict):
        if "title" in obj and isinstance(obj["title"], str):
            return obj, "title"
        if "TITLE" in obj and isinstance(obj["TITLE"], str):
            return obj, "TITLE"
        meta = obj.get("meta")
        if isinstance(meta, dict):
            if "title" in meta and isinstance(meta["title"], str):
                return meta, "title"
            if "TITLE" in meta and isinstance(meta["TITLE"], str):
                return meta, "TITLE"
    return None, None

def update_json_title(json_path: Path, dry_run: bool) -> bool:
    try:
        obj = json.loads(read_text(json_path))
    except Exception:
        return False

    container, key = get_json_title_ref(obj)
    if not container or not key:
        return False

    old = container[key].strip()
    if old.startswith(PREFIX):
        return False

    new = f"{PREFIX} {old}"
    container[key] = new

    if dry_run:
        print(f"JSON UPDATE: {json_path.name}\n  title: {old}\n   ->   {new}")
        return True

    json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"JSON UPDATED: {json_path.name}")
    return True

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized", required=True, help="Path to normalized/ folder")
    ap.add_argument("--abstracts", required=True, help="Path to abstracts/ folder")
    ap.add_argument("--dry-run", action="store_true", help="Print changes only")
    ap.add_argument("--fallback-title-match", action="store_true",
                    help="If VIDEO_ID matching fails, try matching by title (less reliable).")
    args = ap.parse_args()

    normalized_dir = Path(args.normalized).expanduser().resolve()
    abstracts_dir  = Path(args.abstracts).expanduser().resolve()

    if not normalized_dir.exists():
        raise SystemExit(f"Missing folder: {normalized_dir}")
    if not abstracts_dir.exists():
        raise SystemExit(f"Missing folder: {abstracts_dir}")

    # 1) Collect VIDEO_IDs (and remainder titles) from normalized files tagged as Future of Money.
    target_vids = set()
    remainder_titles = set()

    norm_files = sorted(normalized_dir.glob("*.txt"))
    for p in norm_files:
        txt = read_text(p)
        title, vid = parse_headers(txt)
        if not title or not title.startswith(PREFIX):
            continue
        if vid:
            target_vids.add(vid)
        remainder = title[len(PREFIX):].strip()
        if remainder:
            remainder_titles.add(normalize_title_for_match(remainder))

    if not target_vids and not (args.fallback_title_match and remainder_titles):
        print("No Future of Money items found in normalized/, or missing VIDEO_IDs.")
        return 0

    # 2) Index abstract files by VIDEO_ID (from content), plus optional title index.
    md_files = sorted(abstracts_dir.glob("*.md"))
    json_files = sorted(abstracts_dir.glob("*.json"))

    md_by_vid = {}
    json_by_vid = {}

    for p in md_files:
        t, v = parse_headers(read_text(p))
        if v:
            md_by_vid[v] = p

    for p in json_files:
        try:
            obj = json.loads(read_text(p))
        except Exception:
            continue
        # Try to locate VIDEO_ID
        vid = None
        if isinstance(obj, dict):
            for k in ("video_id", "VIDEO_ID", "id"):
                if isinstance(obj.get(k), str):
                    vid = obj[k].strip()
                    break
            meta = obj.get("meta")
            if not vid and isinstance(meta, dict) and isinstance(meta.get("video_id"), str):
                vid = meta["video_id"].strip()
        if vid:
            json_by_vid[vid] = p

    updated = 0
    skipped = 0
    missing = 0

    # 3) Update by VIDEO_ID
    for vid in sorted(target_vids):
        any_hit = False
        mdp = md_by_vid.get(vid)
        jsp = json_by_vid.get(vid)

        if mdp:
            any_hit = True
            if update_md_title(mdp, args.dry_run):
                updated += 1
            else:
                skipped += 1

        if jsp:
            any_hit = True
            if update_json_title(jsp, args.dry_run):
                updated += 1
            else:
                skipped += 1

        if not any_hit:
            missing += 1
            print(f"MISS: No abstract found for VIDEO_ID={vid}")

    # 4) Optional fallback: title matching (if some abstracts don't carry VIDEO_ID consistently)
    if args.fallback_title_match and remainder_titles:
        # Build a title->path map from md files (weak heuristic).
        title_to_md = {}
        for p in md_files:
            t, v = parse_headers(read_text(p))
            if not t:
                continue
            t_norm = normalize_title_for_match(t.replace(PREFIX, "").strip())
            title_to_md.setdefault(t_norm, []).append(p)

        for rem in sorted(remainder_titles):
            for p in title_to_md.get(rem, []):
                if update_md_title(p, args.dry_run):
                    updated += 1

    print(f"Done. Updated: {updated}, skipped (already tagged or no TITLE): {skipped}, missing abstracts: {missing}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())