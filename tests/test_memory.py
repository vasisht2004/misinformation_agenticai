"""Box 5 checks that need the real matrix, plus the tokeniser, which does not.

The rank-1 check is the encoder contract: a stored primary_claim, encoded
again at max length 256 with normalised vectors, has to come back as itself.
A toy matrix would pass for the wrong reason. The session fixture builds the
memory once; later tests reuse the loaded matrices and the loaded model.
"""

import pandas as pd
import pytest

from misinfo_agent.memory import bm25_tokens, get_memory


@pytest.fixture(scope="session")
def memory():
    return get_memory()


def _unique_claim(memory):
    """One stored claim whose text is not shared with another row.

    A duplicated string can tie at cosine 1. Rank 1 would then depend on
    sort order, and the test would not be telling us about the encoder.
    """
    counts = memory.meta["primary_claim"].value_counts()
    unique = set(counts[counts == 1].index)
    subset = memory.meta[memory.meta["primary_claim"].isin(unique)].dropna(subset=["date_parsed"])
    subset = subset[subset["primary_claim"].str.len() > 40]
    return subset.sort_values("article_id").iloc[0]


def test_exact_memory_claim_is_rank_one(memory):
    row = _unique_claim(memory)
    hits = memory.search(row["primary_claim"], k=5)
    assert hits, "exact claim returned nothing"
    assert hits[0]["article_id"] == str(row["article_id"])
    # Re-encoding the stored text should land on top of its own row, not merely
    # in the neighbourhood. A wrong window or a missing normalisation fails here.
    assert hits[0]["dense_cos"] > 0.99
    assert hits[0]["dense_rank"] == 1


def test_date_before_removes_later_rows(memory):
    row = _unique_claim(memory)
    cutoff = pd.Timestamp(row["date_parsed"]).strftime("%Y-%m-%d")
    plain = memory.search(row["primary_claim"], k=5)
    assert plain[0]["article_id"] == str(row["article_id"])
    # The row's own day is not before the cutoff. Strict < has to drop it,
    # along with anything published later. Same-day fact-checks are excluded
    # so evaluation cannot "remember" an article from the day of the claim.
    filtered = memory.search(row["primary_claim"], k=10, date_before=cutoff)
    assert all(hit["article_id"] != str(row["article_id"]) for hit in filtered)
    assert all(hit["date"] is not None and hit["date"] < cutoff for hit in filtered)
    later = [hit for hit in plain if hit["date"] and hit["date"] >= cutoff]
    assert later
    assert {hit["article_id"] for hit in later}.isdisjoint(hit["article_id"] for hit in filtered)


def test_domain_filter_returns_only_that_domain(memory):
    sports = memory.meta[memory.meta["domain"] == "Sports"].iloc[0]
    hits = memory.search(sports["primary_claim"], k=10, domains=["Health"])
    assert hits, "health slice of the memory should still return hits"
    assert {hit["domain"] for hit in hits} == {"Health"}
    assert all(hit["article_id"] != str(sports["article_id"]) for hit in hits)


def test_unknown_domain_raises(memory):
    # "Disasters" is not a label. An empty hit list would look like a claim
    # the archive had never seen, so a wrong name has to be an error.
    with pytest.raises(ValueError) as caught:
        memory.search("flood in Chennai", k=1, domains=["Disasters"])
    message = str(caught.value)
    assert "Disasters & accidents" in message
    assert "Health" in message
    assert "Finance & scams" in message


def test_importing_memory_does_not_import_faiss():
    import ast
    import subprocess
    import sys
    from pathlib import Path

    # A lazy `import faiss` inside search() would not run during a bare
    # import, so the syntax tree is checked as well as a fresh interpreter.
    # This process may already have imported faiss for an unrelated reason.
    source = Path("misinfo_agent/memory.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
            assert "faiss" not in names
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("faiss")
    script = "import sys\nimport misinfo_agent.memory\nassert 'faiss' not in sys.modules\n"
    completed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_faiss_and_numpy_return_the_same_ids():
    # The parity script is a separate interpreter on purpose. Search must
    # not import FAISS, and this process may already have the MPS encoder
    # loaded, which segfaults if FAISS is imported afterwards.
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "scripts/faiss_parity.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "faiss_parity_ok 20" in completed.stdout


def test_rupee_spellings_share_bm25_tokens():
    assert bm25_tokens("Rs 500") == bm25_tokens("₹500")
    assert bm25_tokens("rs.500") == bm25_tokens("₹500")
    assert bm25_tokens("Rs 500") == ["inr500"]
    # The participle and the verb have to be one term, or "withdrawn" never
    # meets the fact-check that says "RBI will withdraw".
    assert bm25_tokens("withdrawn") == bm25_tokens("withdraw")
    # "-ed" alone would leave "announc", which would miss the verb.
    assert bm25_tokens("announced") == bm25_tokens("announce")
