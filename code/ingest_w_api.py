#!/usr/bin/env python3
# ingest.py
#
# Ingest pipeline:
# Inbox/*.(mp3|mp4|m4a|wav|webm|mpeg|mpga|mov|mkv)
#  -> compress/split audio (ffmpeg)
#  -> OpenAI Audio Transcriptions API (gpt-4o-mini-transcribe by default)
#  -> transcripts_raw/<basename>.txt
#  -> move originals to ./audio or ./videos on success
#  -> append registry rows to ingest_registry.jsonl to skip already-done inputs


from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv("config.env")

# -------- Config --------

SUPPORTED_INPUT_EXTS = {
    ".mp3",
    ".mp4",
    ".m4a",
    ".wav",
    ".webm",
    ".mpeg",
    ".mpga",
    ".mov",
    ".mkv",
}

DEFAULT_MODEL = "gpt-4o-mini-transcribe"
# API upload limit documented as 25MB
MAX_UPLOAD_BYTES = (
    25 * 1024 * 1024
)  # 25 MB  [oai_citation:2‡OpenAI Platform](https://platform.openai.com/docs/guides/speech-to-text)

# ffmpeg encoding choices to keep files small but intelligible
# mono + 32kbps is usually fine for speech; bump if you need higher fidelity
FFMPEG_AUDIO_BITRATE = "32k"
FFMPEG_AUDIO_SR = "16000"
FFMPEG_AUDIO_CH = "1"

REGISTRY_FILE = "ingest_registry.jsonl"


# -------- Utilities --------


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def sh(cmd: List[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    # Don't pipe huge stdout; let tools write normally unless we need stderr for errors.
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def which_or_die(bin_name: str):
    if shutil.which(bin_name) is None:
        raise SystemExit(f"{bin_name} not found. Install it and ensure it is on PATH.")


def load_env_file(env_path: Path):
    """
    Minimal .env loader for KEY=VALUE lines.
    Only sets vars if not already set.
    """
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def safe_stem(p: Path) -> str:
    # Keep original stem but normalize whitespace a bit
    s = p.stem
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_video(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}


def file_sig(p: Path) -> str:
    st = p.stat()
    # good-enough signature: path + size + mtime
    return f"{p.name}|{st.st_size}|{int(st.st_mtime)}"


def read_registry(registry_path: Path) -> set[str]:
    done = set()
    if not registry_path.exists():
        return done
    with registry_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sig = row.get("file_sig")
                status = row.get("status")
                if sig and status == "ok":
                    done.add(sig)
            except Exception:
                continue
    return done


def append_registry(registry_path: Path, row: dict):
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -------- Audio prep (ffmpeg) --------


def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def ffmpeg_to_small_mp3(src: Path, out_mp3: Path) -> None:
    """
    Convert any audio/video input to a compact MP3 optimized for speech.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        FFMPEG_AUDIO_CH,
        "-ar",
        FFMPEG_AUDIO_SR,
        "-b:a",
        FFMPEG_AUDIO_BITRATE,
        str(out_mp3),
    ]
    sh(cmd)


def split_mp3_into_chunks(
    src_mp3: Path, out_dir: Path, chunk_seconds: int
) -> List[Path]:
    """
    Split MP3 into N chunks using ffmpeg segmenter.
    """
    ensure_dirs(out_dir)
    pattern = out_dir / (src_mp3.stem + ".part%03d.mp3")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_mp3),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-c",
        "copy",
        str(pattern),
    ]
    sh(cmd)
    chunks = sorted(out_dir.glob(src_mp3.stem + ".part*.mp3"))
    return chunks


def pick_chunk_seconds(mp3_path: Path) -> int:
    """
    Heuristic: aim for chunks under 20MB to stay safely below 25MB after overhead.
    At 32kbps mono, size is ~ 4KB/sec => 20MB ~ 5000 sec ~ 83 min.
    We’ll default to 30 minutes for safety across weird bitrate cases.
    """
    return 30 * 60


# -------- OpenAI transcription --------


def get_openai_client():
    try:
        from openai import OpenAI
    except Exception as ex:
        raise SystemExit(
            "Python package 'openai' not found. Install it in your venv:\n"
            "  pip install -U openai\n"
        ) from ex

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it or put it in ./config.env.\n"
            "Example:\n"
            "  export OPENAI_API_KEY='...'\n"
        )
    return OpenAI()


def transcribe_file_with_retries(
    client,
    audio_path: Path,
    *,
    model: str,
    language: Optional[str],
    response_format: str = "text",
    max_retries: int = 8,
) -> str:
    """
    Calls /v1/audio/transcriptions. Retries on transient 429/5xx.
    """
    # Lazy import to avoid hard dependency if user only wants dry-run
    from openai import APIError, APITimeoutError, RateLimitError

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            with audio_path.open("rb") as f:
                kwargs = {
                    "model": model,
                    "file": f,
                    "response_format": response_format,
                }
                # language is optional; for many models it can help if known
                if language:
                    kwargs["language"] = language

                # Note: docs say 4o-transcribe models support json or text.  [oai_citation:3‡OpenAI Platform](https://platform.openai.com/docs/guides/speech-to-text)
                resp = client.audio.transcriptions.create(**kwargs)
                # openai SDK returns an object with .text for these endpoints
                text = getattr(resp, "text", None)
                if not text:
                    # fall back if SDK returns raw dict-like
                    if isinstance(resp, dict) and "text" in resp:
                        text = resp["text"]
                if not text:
                    raise RuntimeError("Transcription succeeded but no text returned.")
                return text

        except RateLimitError as ex:
            # 429: backoff and retry
            eprint(
                f"  Rate limited (attempt {attempt}/{max_retries}). Sleeping {backoff:.1f}s"
            )
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 30.0)
            last = ex
        except (APITimeoutError, APIError) as ex:
            eprint(
                f"  API error (attempt {attempt}/{max_retries}): {ex}. Sleeping {backoff:.1f}s"
            )
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 30.0)
            last = ex
        except Exception as ex:
            # Non-retryable
            raise

    raise last  # type: ignore


# -------- Core processing --------


@dataclass
class Result:
    input_file: str
    file_sig: str
    status: str  # ok | error | skipped
    transcript_txt: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    notes: Optional[str] = None
    error: Optional[str] = None


def process_one(
    src: Path,
    *,
    client,
    model: str,
    language: Optional[str],
    work_audio_dir: Path,
    transcripts_raw_dir: Path,
    videos_dir: Path,
    audio_dir: Path,
    tmp_dir: Path,
    keep_intermediate: bool,
) -> Result:
    sig = file_sig(src)
    base = safe_stem(src)

    ensure_dirs(work_audio_dir, transcripts_raw_dir, videos_dir, audio_dir, tmp_dir)

    out_txt = transcripts_raw_dir / f"{base}.txt"

    # 1) Convert to compact mp3 (or reuse if already exists)
    small_mp3 = work_audio_dir / f"{base}.ingest.mp3"
    if not small_mp3.exists():
        eprint(f"  Audio: {src.name} -> {small_mp3.name}")
        ffmpeg_to_small_mp3(src, small_mp3)
    else:
        eprint(f"  Audio: exists -> {small_mp3.name}")

    # 2) If still > 25MB, split
    parts: List[Path]
    if small_mp3.stat().st_size <= MAX_UPLOAD_BYTES:
        parts = [small_mp3]
    else:
        chunk_seconds = pick_chunk_seconds(small_mp3)
        parts_dir = tmp_dir / (base + ".parts")
        eprint(
            f"  Split: {small_mp3.name} is {small_mp3.stat().st_size / 1024 / 1024:.1f}MB, chunking..."
        )
        parts = split_mp3_into_chunks(small_mp3, parts_dir, chunk_seconds)
        if not parts:
            raise RuntimeError("Split produced zero chunks.")

        # sanity: filter any chunk still too big
        too_big = [p for p in parts if p.stat().st_size > MAX_UPLOAD_BYTES]
        if too_big:
            raise RuntimeError(
                "Some chunks still exceed 25MB. Increase compression (bitrate) or reduce chunk_seconds."
            )

    # 3) Transcribe parts, then join
    full_texts: List[str] = []
    for i, part in enumerate(parts, start=1):
        label = f"{i}/{len(parts)}" if len(parts) > 1 else "1/1"
        eprint(f"  Transcribe [{label}]: {part.name}")
        t = transcribe_file_with_retries(
            client,
            part,
            model=model,
            language=language,
            response_format="text",
        )
        full_texts.append(t.strip())

    merged = "\n\n".join(full_texts).strip() + "\n"
    out_txt.write_text(merged, encoding="utf-8")

    # 4) Move original to destination on success
    dest_dir = videos_dir if is_video(src) else audio_dir
    dest = dest_dir / src.name
    # avoid clobber
    if dest.exists():
        # add suffix
        dest = dest_dir / f"{src.stem}.dup{int(time.time())}{src.suffix}"
    shutil.move(str(src), str(dest))

    # 5) Cleanup intermediates
    if not keep_intermediate:
        try:
            if small_mp3.exists():
                small_mp3.unlink()
        except Exception:
            pass
        # delete chunk dir
        try:
            parts_dir = tmp_dir / (base + ".parts")
            if parts_dir.exists() and parts_dir.is_dir():
                shutil.rmtree(parts_dir)
        except Exception:
            pass

    return Result(
        input_file=str(src),
        file_sig=sig,
        status="ok",
        transcript_txt=str(out_txt),
        model=model,
        language=language or "",
        notes=f"moved_to={dest}",
    )


def iter_inbox_files(inbox: Path) -> List[Path]:
    files = []
    for p in sorted(inbox.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in SUPPORTED_INPUT_EXTS:
            files.append(p)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest audio/video from Inbox using OpenAI STT API."
    )
    parser.add_argument(
        "--inbox", default="Inbox", help="Folder containing new inputs (default: Inbox)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Transcription model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--language", default=None, help="Language code, e.g. en (optional)"
    )
    parser.add_argument(
        "--config", default="config.env", help="Optional env file with OPENAI_API_KEY"
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate compressed/split audio",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List what would be processed and exit"
    )

    args = parser.parse_args()

    root = Path.cwd()
    inbox = root / args.inbox
    if not inbox.exists():
        raise SystemExit(f"Inbox folder not found: {inbox}")

    load_env_file(root / args.config)

    which_or_die("ffmpeg")

    registry_path = root / REGISTRY_FILE
    done = read_registry(registry_path)

    files = iter_inbox_files(inbox)
    if not files:
        print("No supported inputs found in Inbox.")
        return 0

    if args.dry_run:
        print("Would process:")
        for p in files:
            print(f" - {p.name}")
        return 0

    client = get_openai_client()

    # output dirs (you already have these)
    work_audio_dir = root / "misc" / "ingest_audio"
    transcripts_raw_dir = root / "transcripts_raw"
    videos_dir = root / "videos"
    audio_dir = root / "audio"
    tmp_dir = root / "misc" / "tmp"

    ensure_dirs(work_audio_dir, transcripts_raw_dir, videos_dir, audio_dir, tmp_dir)

    total = len(files)
    ok = 0
    skipped = 0
    errors = 0

    for idx, src in enumerate(files, start=1):
        sig = file_sig(src)
        print(f"\n[{idx}/{total}]")
        print(f"Processing: {src.name}")

        if sig in done:
            print("  SKIP: already ingested (registry)")
            row = asdict(
                Result(
                    input_file=str(src),
                    file_sig=sig,
                    status="skipped",
                    notes="already_ok",
                )
            )
            append_registry(registry_path, row)
            skipped += 1
            continue

        try:
            r = process_one(
                src,
                client=client,
                model=args.model,
                language=args.language,
                work_audio_dir=work_audio_dir,
                transcripts_raw_dir=transcripts_raw_dir,
                videos_dir=videos_dir,
                audio_dir=audio_dir,
                tmp_dir=tmp_dir,
                keep_intermediate=args.keep_intermediate,
            )
            print(f"  OK: wrote {Path(r.transcript_txt).name}")
            append_registry(registry_path, asdict(r))
            ok += 1

        except Exception as ex:
            errors += 1
            msg = f"{type(ex).__name__}: {ex}"
            print(f"  ERROR: {src.name}: {msg}")

            # write error file next to where you expect to look
            err_path = root / "transcripts_raw" / f"{safe_stem(src)}.error.txt"
            try:
                err_path.write_text(msg + "\n", encoding="utf-8")
                print(f"  Wrote: {err_path}")
            except Exception:
                pass

            append_registry(
                registry_path,
                asdict(
                    Result(input_file=str(src), file_sig=sig, status="error", error=msg)
                ),
            )

    print("\nDone.")
    print(f"  OK: {ok}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print("Outputs in: transcripts_raw/")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
