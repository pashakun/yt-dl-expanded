# Podcast Corpus: Crypto & Financial Infrastructure

This repository contains a curated, machine-readable corpus of long-form crypto and financial-infrastructure conversations, designed for **idea mining and narrative analysis with LLMs**.

## Why this exists

Using LLMs directly on podcast transcripts runs into two structural limitations:

1. Single-episode prompts tend to produce shallow summaries
2. Feeding multiple full transcripts into an LLM quickly exceeds the **context window (token limit)**

This corpus introduces an intermediate abstraction layer that compresses long conversations into structured analytical summaries, enabling cross-episode reasoning without uploading raw transcripts.

## What’s inside

The corpus includes ~100 long-form podcast episodes across shows such as Bankless, Epicenter, a16z, Unchained, Future of Money, and The Defiant.

Each episode was processed through the following pipeline:

1. Audio downloaded from YouTube
2. Audio transcribed to text
3. Transcripts normalized for consistent structure and metadata
4. A structured abstract generated via an LLM
   - No direct quotes
   - No timestamps
   - No promotional language

The result is a compressed analytical representation of each conversation.

## Key file

### `corpus_abstracts.md`

This is the primary artifact.

It contains one structured abstract per episode and is intended to be **used directly as LLM input**. Example queries include:

- “What are the dominant narratives around stablecoins and institutional adoption?”
- “What tensions repeatedly surface around compliance, custody, and risk?”
- “Where do speakers diverge on the future role of banks in crypto?”

Because the underlying transcripts are abstracted, the file supports analysis across many episodes without hitting token limits.

## Folder structure (high level)

- `videos/` — original video files
- `wav/` — extracted audio
- `transcripts/` — raw transcript text
- `normalized/` — cleaned transcripts with consistent metadata
- `abstracts/` — per-episode structured abstracts
- `corpus_abstracts.md` — merged abstract corpus (recommended entry point)
- `subtitles/` — `.srt` files
- `*.py` — scripts used to build the processing pipeline

## How to use this corpus

For analysis and research:
- Open `corpus_abstracts.md`
- Paste it into an LLM
- Ask thematic, comparative, or synthesis-level questions

The same pipeline can be applied to other long-form content such as:
- Webinars
- Conference talks
- Panels
- Internal recordings

Any format where **themes, arguments, and institutional implications** matter more than verbatim quotes.
