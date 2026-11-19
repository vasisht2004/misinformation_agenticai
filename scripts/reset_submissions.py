"""Empty data/submissions.db before an evaluation.

Reach is the number of stored submissions. A demo that recorded the
garlic forward would otherwise look like a crowd when the eval scores
the same text. This deletes the file. The next ingest creates it again.

It asks before deleting. A bare run, or an end-of-file where a person
was supposed to answer, leaves the file alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import misinfo_agent.config as config  # noqa: E402


def main() -> int:
    path = config.SUBMISSIONS_DB
    print(f"This will delete {path}.")
    print("Reach counts come from this file. Empty it only before an evaluation.")
    try:
        answer = input("Type yes to delete it: ")
    except EOFError:
        print("No confirmation. Left the database as it is.")
        return 1
    if answer.strip() != "yes":
        print("Left the database as it is.")
        return 1
    if not path.exists():
        print("There was no database file.")
        return 0
    path.unlink()
    print("Deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
