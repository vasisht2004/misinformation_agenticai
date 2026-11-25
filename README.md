This repository is an agentic misinformation detector for Indian content: English text, a link, or an image. The model reads and writes. Evidence and explicit rules decide the verdict. The card says Unverifiable when two independent credible sources cannot be found.

The app title is two lines: AGENTIC AI SYSTEM / FOR MISINFORMATION DETECTION. How to run each input, and the card it should show, is `docs/runbook.md`. The design is `docs/ARCHITECTURE.md`. The map of the repo is `docs/PROJECT_CONTEXT.md`. Eval write-ups are `docs/eval_prior_match.md`, `docs/eval_investigate.md`, `docs/eval_end_to_end.md`, `docs/eval_diagnosis.md`, and `docs/system_check.md`.

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

Wikipedia's perennial sources list adds credibility tiers the curated lists do not name. The checker still runs without the file and logs a warning once. Build it with:

```bash
python scripts/build_source_tiers.py
```

```bash
cp .env.example .env
pytest
```

Never commit `.env`. Work on `feat/<area>`. Open a pull request into `main`; Vash reviews it. Never commit `data/` or `.env`.
