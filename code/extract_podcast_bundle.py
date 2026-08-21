#!/usr/bin/env python3

import re
import shutil
import sys
from pathlib import Path

TITLE_PREFIX = "The Future of Money:"

NORMALIZED_DIR = Path("normalized")
ABSTRACTS_DIR = Path("abstracts")
SUBTITLES_DIR = Path("subtitles")

OUT_ROOT = Path("podcasts") / "The Future of Money"
OUT_NORMALIZED = OUT_ROOT / "normalized"
OUT_ABSTRACTS = OUT_ROOT / "abstracts"
OUT_SUBTITLES = OUT_ROOT / "subtitles"

TITLE_RE = re.compile(r"^TITLE:\s*(.+)$", re.MULTILINE)
VIDEO_ID_RE = re.compile(r"^VIDEO_ID:\s*(.+)$", re.MULTILINE)


def extract_title_and_id(norm_path: Path):
    text = norm_path.read_text(encoding="utf-8", errors="ignore")
    m_title = TITLE_RE.search(text)
    m_vid = VIDEO_ID_RE.search(text)
    if not m_title or not m_vid:
        return None
    return m_title.group(1).strip(), m_vid.group(1).strip()


def ensure_dirs():
    OUT_NORMALIZED.mkdir(parents=True, exist_ok=True)
    OUT_ABSTRACTS.mkdir(parents=True, exist_ok=True)
    OUT_SUBTITLES.mkdir(parents=True, exist_ok=True)


def copy_matches(video_id: str, dry_run: bool = False):
    copied = {"normalized": 0, "abstracts": 0, "subtitles": 0}

    # normalized: copy the canonical normalized file(s) that contain this video_id
    norm_matches = list(NORMALIZED_DIR.glob(f"*{video_id}*.txt"))
    for p in norm_matches:
        dst = OUT_NORMALIZED / p.name
        if dry_run:
            print(f"[DRY] normalized: {p} -> {dst}")
        else:
            shutil.copy2(p, dst)
        copied["normalized"] += 1

    # abstracts: copy both .md and .json that match
    abs_matches = list(ABSTRACTS_DIR.glob(f"*{video_id}*"))
    for p in abs_matches:
        if p.suffix not in {".md", ".json"}:
            continue
        dst = OUT_ABSTRACTS / p.name
        if dry_run:
            print(f"[DRY] abstracts: {p} -> {dst}")
        else:
            shutil.copy2(p, dst)
        copied["abstracts"] += 1

    # subtitles: copy .srt that match
    srt_matches = list(SUBTITLES_DIR.glob(f"*{video_id}*.srt"))
    for p in srt_matches:
        dst = OUT_SUBTITLES / p.name
        if dry_run:
            print(f"[DRY] subtitles: {p} -> {dst}")
        else:
            shutil.copy2(p, dst)
        copied["subtitles"] += 1

    return copied


def main():
    dry_run = "--dry-run" in sys.argv

    ensure_dirs()

    video_ids = []
    for norm in NORMALIZED_DIR.glob("*.txt"):
        parsed = extract_title_and_id(norm)
        if not parsed:
            continue
        title, vid = parsed
        if title.startswith(TITLE_PREFIX):
            video_ids.append(vid)

    video_ids = sorted(set(video_ids))

    if not video_ids:
        print(f"No normalized files found with TITLE prefix: {TITLE_PREFIX}")
        return 1

    total = {"normalized": 0, "abstracts": 0, "subtitles": 0}
    missing = {"normalized": 0, "abstracts": 0, "subtitles": 0}

    for vid in video_ids:
        copied = copy_matches(vid, dry_run=dry_run)
        for k in total:
            total[k] += copied[k]
        # Track missing per type (if nothing copied for that type)
        for k in missing:
            if copied[k] == 0:
                missing[k] += 1

    print("\n== Summary ==")
    print(f"Episodes matched: {len(video_ids)}")
    print(
        f"Copied normalized: {total['normalized']} (missing for {missing['normalized']} episodes)"
    )
    print(
        f"Copied abstracts:  {total['abstracts']} (missing for {missing['abstracts']} episodes)"
    )
    print(
        f"Copied subtitles:  {total['subtitles']} (missing for {missing['subtitles']} episodes)"
    )
    print(f"Output folder: {OUT_ROOT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
