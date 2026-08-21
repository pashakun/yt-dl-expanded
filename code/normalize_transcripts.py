#!/usr/bin/env python3
"""
Normalize Whisper transcript .txt files into a consistent format:

TITLE: ...
URL: ...
VIDEO_ID: ...

TRANSCRIPT:
...

Works for:
- YouTube-style filenames: "Some Title [VIDEO_ID].wav.txt" or "Some Title [VIDEO_ID].txt"
- Non-YouTube files (no [ID]): assigns VIDEO_ID based on the filename stem

Optionally enriches TITLE/URL using titles_and_urls.txt (same format as before).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("config.env")

# YouTube ID in brackets, e.g. "... [A7gSCjWy3qo].txt" or "... [A7gSCjWy3qo].wav.txt"
YT_ID_IN_BRACKETS_RE = re.compile(
    r"\[([A-Za-z0-9_-]{8,})\](?:\.wav)?\.txt$", re.IGNORECASE
)

# Extract ID from URL lines if needed (v= or youtu.be/)
ID_IN_URL_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{8,})")

# Skip common non-transcript files
DEFAULT_SKIP_NAMES = {
    ".DS_Store",
    "titles_and_urls.txt",
    "trawl.txt",
    "trawl.log",
    "archive.txt",
    "failed_inputs.txt",
    "missing_urls.txt",
}


def load_title_url_map(path: Path) -> dict[str, tuple[str, str]]:
    """
    Supports lines like:
    - VIDEO_ID | Title | URL
    - Title | URL
    """
    mapping: dict[str, tuple[str, str]] = {}
    if not path.exists():
        return mapping

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(" | ") if p.strip()]
        if len(parts) < 2:
            continue

        # Case 1: ID | Title | URL
        if (
            len(parts) >= 3
            and re.fullmatch(r"[A-Za-z0-9_-]{8,}", parts[0])
            and ("youtube" in parts[-1] or "youtu.be" in parts[-1])
        ):
            vid = parts[0]
            title = " | ".join(parts[1:-1])
            url = parts[-1]
            mapping[vid] = (title, url)
            continue

        # Case 2: Title | URL
        title = " | ".join(parts[:-1])
        url = parts[-1]
        m = ID_IN_URL_RE.search(url)
        if not m:
            continue
        vid = m.group(1)
        mapping[vid] = (title, url)

    return mapping


def infer_video_id_from_filename(p: Path) -> str:
    """
    1) If filename contains [YOUTUBE_ID], use it.
    2) Otherwise create a stable ID from the filename stem (sanitized).
    """
    m = YT_ID_IN_BRACKETS_RE.search(p.name)
    if m:
        return m.group(1)

    stem = p.stem
    # If it's "*.wav.txt", p.stem becomes "*.wav" so strip a trailing ".wav" stem too
    if stem.lower().endswith(".wav"):
        stem = stem[:-4]

    # Sanitize to something safe (letters, numbers, underscore, dash)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return safe or "unknown"


def infer_title_from_filename(p: Path) -> str:
    """
    If file name is like 'Some Title [ID].txt', strip the bracket part.
    Otherwise use stem.
    """
    name = p.name
    # Remove suffixes
    if name.lower().endswith(".wav.txt"):
        base = name[: -len(".wav.txt")]
    elif name.lower().endswith(".txt"):
        base = name[: -len(".txt")]
    else:
        base = p.stem

    # Strip trailing " [ID]" if present
    base = re.sub(r"\s*\[[A-Za-z0-9_-]{8,}\]\s*$", "", base).strip()
    return base


def find_transcript_files(indir: Path, skip_names: set[str]) -> list[Path]:
    files: list[Path] = []
    for p in indir.glob("*.txt"):
        if p.name in skip_names:
            continue
        # Heuristic: ignore obviously non-transcript tiny files if you want,
        # but here we keep it simple and accept any txt except skip list.
        files.append(p)
    return sorted(files)


def normalize_one(
    txt_path: Path,
    out_dir: Path,
    title_url_map: dict[str, tuple[str, str]],
) -> tuple[bool, str]:
    """
    Returns (ok, message). Writes normalized file into out_dir with same filename.
    """
    video_id = infer_video_id_from_filename(txt_path)

    # Enrich from mapping if possible
    title, url = title_url_map.get(video_id, ("", ""))

    # If no mapped title, use filename-based title as a fallback
    if not title:
        title = infer_title_from_filename(txt_path)

    content = txt_path.read_text(encoding="utf-8", errors="replace").strip()

    normalized_text = (
        f"TITLE: {title}\nURL: {url}\nVIDEO_ID: {video_id}\n\nTRANSCRIPT:\n{content}\n"
    )

    out_path = out_dir / txt_path.name
    out_path.write_text(normalized_text, encoding="utf-8")
    return True, f"Wrote: {out_path.name}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="indir",
        required=True,
        help="Input folder containing raw *.txt transcripts",
    )
    ap.add_argument(
        "--out",
        dest="outdir",
        required=True,
        help="Output folder for normalized transcripts",
    )
    ap.add_argument(
        "--map",
        dest="map_file",
        default="titles_and_urls.txt",
        help="Path to titles_and_urls.txt (optional)",
    )
    ap.add_argument(
        "--skip",
        dest="skip_file",
        default="",
        help="Optional path to a file listing extra filenames to skip (one per line)",
    )
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    map_file = Path(args.map_file)

    if not indir.exists() or not indir.is_dir():
        print(f"Input folder not found: {indir}", file=sys.stderr)
        return 2

    skip_names = set(DEFAULT_SKIP_NAMES)
    if args.skip_file:
        sf = Path(args.skip_file)
        if sf.exists():
            for line in sf.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line:
                    skip_names.add(line)

    title_url_map = load_title_url_map(map_file)

    outdir.mkdir(parents=True, exist_ok=True)
    txt_files = find_transcript_files(indir, skip_names)

    if not txt_files:
        print(f"No transcript .txt files found in: {indir}")
        return 1

    missing_url = 0
    written = 0

    for p in txt_files:
        ok, _ = normalize_one(p, outdir, title_url_map)
        if ok:
            written += 1
            vid = infer_video_id_from_filename(p)
            _, url = title_url_map.get(vid, ("", ""))
            if not url:
                missing_url += 1

    print(f"Normalized transcripts written: {written} -> {outdir}/")
    print(f"Files with missing URL metadata: {missing_url} (URL will be blank)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
