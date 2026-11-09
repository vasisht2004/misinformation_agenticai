This repository is an agentic misinformation detector. It retrieves similar fact-checks from a corpus of classified Indian fact-check articles and claim embeddings, then uses a language-model agent to judge whether a new claim is supported, misleading, or unverified.

The article corpus and embeddings are not stored in git. See DATA.md for the file manifest, checksums, and where to get the shared copies.

## Getting started

```bash
git clone https://github.com/vasisht2004/misinformation_agenticai.git
cd misinformation_agenticai
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Get `data/raw/classified_v2.parquet` and `data/raw/bge_m3_primary_claim.parquet` from Vash. They are not in git. Check them against DATA.md:

```bash
shasum -a 256 data/raw/classified_v2.parquet data/raw/bge_m3_primary_claim.parquet
```

Build the memory (this writes `data/memory/` and skips the work when the manifest already matches):

```bash
python -c "from misinfo_agent.memory import build_memory; build_memory()"
```

```bash
cp .env.example .env
pytest
```

Never commit `.env`. Work on `feat/<area>`. Open a pull request into `main`; Vash reviews it. Never commit `data/` or `.env`.
