The parquet files below were copied on 2026-11-04. They are not stored in git. The files are shared via the Google Drive folder "Misinformation project".

| File | Source path | Copy date | Rows | Size (bytes) | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `data/raw/classified_v2.parquet` | `/Users/payas/Downloads/codes_misinformation/data/classified/classified_v2.parquet` | 2026-11-04 | 52,678 (52,099 with `is_fact_check == true`) | 88,155,193 | `04b934ec67059bceb2e22812341aaa9edf0fb94e636e9eaf2bb444ee8a038949` |
| `data/raw/bge_m3_primary_claim.parquet` | `/Users/payas/Downloads/codes_misinformation/data/embeddings/bge_m3_primary_claim.parquet` | 2026-11-04 | 52,099 | 328,644,857 | `0f9e0e9423607dd5d8f7a5c82e2cb34b8bf81c0c6317a70e0b6de28aaf9c8266` |

Every embedding `article_id` is present in `classified_v2.parquet`. `misinfo_agent/prompts/prompt_v4a.txt` is committed in git.
