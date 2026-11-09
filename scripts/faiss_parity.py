"""FAISS cross-check for the claim memory. Not used by search.

NumPy is the path ClaimMemory.search actually runs. A date filter is a
boolean mask, and a flat index does not apply one. This script is the only
place FAISS is imported: loading it in the same process as the MPS encoder
segfaults, so the search module must not import it at all.

The query is a stored row, not a freshly encoded string. Both sides see the
same vector, so a mismatch is the index arithmetic disagreeing, not MPS noise.
Unfiltered, because the mask is the part FAISS is not being asked to do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misinfo_agent.config import EMB_NPY_PATH, FAISS_INDEX_PATH
from misinfo_agent.memory import _topk_indices


def main() -> None:
    matrix = np.load(EMB_NPY_PATH)
    # Rows are unit length, so inner product is cosine.
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(np.ascontiguousarray(matrix))
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    rng = np.random.default_rng(42)
    picks = rng.choice(matrix.shape[0], size=20, replace=False)
    for row in picks:
        query = np.ascontiguousarray(matrix[int(row)])
        numpy_ids = _topk_indices(matrix @ query, 10)
        _scores, faiss_ids = index.search(query.reshape(1, -1), 10)
        if not np.array_equal(numpy_ids, faiss_ids[0]):
            raise SystemExit(
                "FAISS and NumPy disagreed on top-10 ids for memory row "
                f"{int(row)}: numpy={numpy_ids.tolist()} faiss={faiss_ids[0].tolist()}"
            )
    print("faiss_parity_ok 20")


if __name__ == "__main__":
    main()
