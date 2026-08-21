#!/usr/bin/env python3
# ingest.py
#
# Ingest pipeline:
#   Inbox/  -> audio/*.wav + transcripts_raw/*.txt + subtitles/*.srt + videos/* (archived originals)
#
# Requirements:
#   - ffmpeg installed and on PATH
#   - whisper CLI installed (pip install -U openai-whisper)
#
# Example:
#   source .venv/bin/activate
#   python3 ingest.py --inbox Inbox --model medium --language en

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv("config.env")

MEDIA_EXTS = {
    ".mp3",
    ".mp4",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".mov",
    ".mkv",
    ".webm",
    ".ogg",
}


@dataclass
class Result:
    src: Path
    ok: bool
    reason: str = ""
    wav: Optional[Path] = None
    txt: Optional[Path] = None
    srt: Optional[Path] = None
    seconds: Optional[float] = None
    log_path: Optional[Path] = None


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def sh(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a command without deadlocking on huge stdout/stderr.
    If stdout_path/stderr_path provided, stream output to files.
    """
    stdout_f = open(stdout_path, "wb") if stdout_path else None
    stderr_f = open(stderr_path, "wb") if stderr_path else None
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=stdout_f if stdout_f else None,
            stderr=stderr_f if stderr_f else None,
            check=check,
        )
    finally:
        if stdout_f:
            stdout_f.close()
        if stderr_f:
            stderr_f.close()


def require_on_path(bin_name: str) -> None:
    if shutil.which(bin_name) is None:
        raise RuntimeError(f"'{bin_name}' not found on PATH")


def ffprobe_duration_seconds(media_path: Path) -> Optional[float]:
    try:
        # Duration can be missing for some files; this tries to pull it cleanly.
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if cp.returncode != 0:
            return None
        s = cp.stdout.strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def to_wav(src: Path, audio_dir: Path) -> Path:
    """
    Convert input media to WAV in audio_dir.
    Uses 16kHz mono PCM for Whisper speed and consistency.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"{src.stem}.wav"
    if wav_path.exists() and wav_path.stat().st_size > 0:
        return wav_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(wav_path),
    ]
    sh(cmd, check=True)
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise RuntimeError("WAV conversion failed (empty or missing output)")
    return wav_path


def run_whisper(wav_path, out_dir, model="small", language="en"):
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "whisper",
        str(wav_path),
        "--model",
        model,
        "--language",
        language,
        "--output_format",
        "txt",
        "--output_dir",
        str(out_dir),
        "--fp16",
        "False",
    ]

    subprocess.run(cmd, check=True)

    expected = out_dir / f"{wav_path.stem}.txt"
    if not expected.exists():
        raise RuntimeError(f"Whisper TXT output not found: {expected}")

    return expected


def safe_move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # avoid clobbering: append timestamp
        ts = time.strftime("%Y%m%d-%H%M%S")
        dst = dst.with_name(f"{dst.stem}.{ts}{dst.suffix}")
    shutil.move(str(src), str(dst))


def write_registry(registry_path: Path, payload: dict) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def process_one(
    src: Path,
    *,
    inbox_dir: Path,
    videos_dir: Path,
    audio_dir: Path,
    transcripts_raw_dir: Path,
    subtitles_dir: Path,
    logs_dir: Path,
    registry_path: Path,
    model: str,
    language: Optional[str],
) -> Result:
    r = Result(src=src, ok=False)
    r.seconds = ffprobe_duration_seconds(src)

    log_path = logs_dir / f"{src.name}.log.txt"
    logs_dir.mkdir(parents=True, exist_ok=True)
    r.log_path = log_path

    try:
        wav_path = to_wav(src, audio_dir)
        r.wav = wav_path

        txt_path, srt_path = run_whisper(
            wav_path,
            out_dir=transcripts_raw_dir,
            model=model,
            language=language,
            log_path=log_path,
        )
        r.txt = txt_path
        r.srt = srt_path

        # Move SRT into subtitles/, keep TXT in transcripts_raw/
        safe_move(srt_path, subtitles_dir / srt_path.name)
        r.srt = subtitles_dir / srt_path.name  # update pointer

        # Archive original media into videos/
        safe_move(src, videos_dir / src.name)

        r.ok = True
        r.reason = "ok"

    except Exception as ex:
        r.ok = False
        r.reason = str(ex)

        # Write an error sidecar next to intended outputs for quick grepping
        err_path = (
            Path("abstracts") if Path("abstracts").exists() else Path(".")
        ) / f"{src.stem}.ingest.error.txt"
        try:
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(r.reason + "\n")
                if r.log_path:
                    f.write(f"Log: {r.log_path}\n")
        except Exception:
            pass

    # Registry record (always)
    write_registry(
        registry_path,
        {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "src": str(src),
            "ok": r.ok,
            "reason": r.reason,
            "wav": str(r.wav) if r.wav else None,
            "txt": str(r.txt) if r.txt else None,
            "srt": str(r.srt) if r.srt else None,
            "seconds": r.seconds,
            "model": model,
            "language": language,
            "log": str(r.log_path) if r.log_path else None,
        },
    )

    return r


def iter_inbox(inbox_dir: Path) -> List[Path]:
    items: List[Path] = []
    for p in sorted(inbox_dir.iterdir()):
        if p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in MEDIA_EXTS:
            items.append(p)
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", default="Inbox", help="Inbox folder to process")
    ap.add_argument(
        "--model", default="medium", help="Whisper model (tiny/base/small/medium/large)"
    )
    ap.add_argument(
        "--language",
        default="en",
        help="Language code (e.g., en). Use '' to auto-detect.",
    )
    args = ap.parse_args()

    inbox_dir = Path(args.inbox).expanduser().resolve()
    if not inbox_dir.exists():
        eprint(f"Inbox not found: {inbox_dir}")
        return 2

    # Enforce dependencies early with a clear error
    try:
        require_on_path("ffmpeg")
        require_on_path("ffprobe")
        require_on_path("whisper")
    except Exception as ex:
        eprint(str(ex))
        eprint("If whisper is missing, activate your venv and install it:")
        eprint("  source .venv/bin/activate")
        eprint("  pip install -U openai-whisper")
        return 2

    # Project dirs (in situ)
    root = Path.cwd()
    videos_dir = root / "videos"
    audio_dir = root / "audio"
    transcripts_raw_dir = root / "transcripts_raw"
    subtitles_dir = root / "subtitles"
    logs_dir = root / "misc" / "logs"
    registry_path = root / "ingest_registry.jsonl"

    # Normalize language arg: allow empty string for auto
    language = args.language.strip() if args.language is not None else None
    if language == "":
        language = None

    items = iter_inbox(inbox_dir)
    if not items:
        print(f"No media files found in {inbox_dir}")
        return 0

    print(f"Found {len(items)} file(s) in {inbox_dir}")

    ok = 0
    fail = 0
    for i, src in enumerate(items, start=1):
        print(f"\n[{i}/{len(items)}]\nProcessing: {src.name}")
        r = process_one(
            src,
            inbox_dir=inbox_dir,
            videos_dir=videos_dir,
            audio_dir=audio_dir,
            transcripts_raw_dir=transcripts_raw_dir,
            subtitles_dir=subtitles_dir,
            logs_dir=logs_dir,
            registry_path=registry_path,
            model=args.model,
            language=language,
        )
        if r.ok:
            ok += 1
            print("  OK")
            if r.txt:
                print(f"  TXT: {Path(r.txt).name} -> {transcripts_raw_dir}")
            if r.srt:
                print(f"  SRT: {Path(r.srt).name} -> {subtitles_dir}")
            print(f"  Archived: {src.name} -> {videos_dir}")
        else:
            fail += 1
            print(f"  ERROR: {src.name}: {r.reason}")
            if r.log_path:
                print(f"  Log: {r.log_path}")

    print(f"\nDone. OK: {ok}, Errors: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
