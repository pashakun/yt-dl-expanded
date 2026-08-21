.PHONY: ingest status build-corpus retry-failed

ingest:
	python3 ingest.py --inbox

status:
	python3 ingest.py --status

build-corpus:
	python3 build_corpus.py

retry-failed:
	python3 ingest.py --retry-failed