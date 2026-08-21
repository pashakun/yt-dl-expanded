#!/usr/bin/env python3

from pathlib import Path
import re

NORMALIZED_DIR = Path("normalized")
SUBTITLES_DIR = Path("subtitles")

TITLE_RE = re.compile(r"^TITLE:\s*(.+)$", re.MULTILINE)
VIDEO_ID_RE = re.compile(r"^VIDEO_ID:\s*(.+)$", re.MULTILINE)

def extract_title_and_id(norm_path: Path):
    text = norm_path.read_text(encoding="utf-8", errors="ignore")
    m_title = TITLE_RE.search(text)
    m_vid = VIDEO_ID_RE.search(text)
    if not m_title or not m_vid:
        return None
    return m_vid.group(1).strip(), m_title.group(1).strip()

def update_srt_title(srt_path: Path, new_title: str):
    lines = srt_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    out = []
    replaced = False
    i = 0

    while i < len(lines):
        line = lines[i]
        out.append(line)

        # Look for first subtitle text block
        if not replaced and i + 2 < len(lines):
            if lines[i].isdigit() and "-->" in lines[i + 1]:
                # Replace subtitle text line
                out.append(new_title)
                i += 3
                replaced = True
                continue

        i += 1

    if replaced:
        srt_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return replaced

def main():
    updated = 0
    skipped = 0

    for norm in NORMALIZED_DIR.glob("*.txt"):
        parsed = extract_title_and_id(norm)
        if not parsed:
            continue

        video_id, title = parsed
        if not title.startswith("The Future of Money"):
            continue

        matches = list(SUBTITLES_DIR.glob(f"*{video_id}*.srt"))
        if not matches:
            skipped += 1
            continue

        for srt in matches:
            if update_srt_title(srt, title):
                updated += 1

    print(f"Updated SRT files: {updated}")
    print(f"Skipped (no match): {skipped}")

if __name__ == "__main__":
    main()
    