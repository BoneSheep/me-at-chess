"""Download every game a chess.com user has played, one PGN file per month.

Uses the public chess.com API (no key needed):
  GET https://api.chess.com/pub/player/{user}/games/archives
  GET https://api.chess.com/pub/player/{user}/games/{YYYY}/{MM}/pgn

Months that are already on disk are skipped, except the two most recent
ones, which may still be growing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

API = "https://api.chess.com/pub/player/{user}/games/archives"
HEADERS = {"User-Agent": "me-at-chess (github.com/BoneSheep/me-at-chess)"}


def _get(session: requests.Session, url: str, retries: int = 5) -> requests.Response:
    for attempt in range(retries):
        resp = session.get(url, headers=HEADERS, timeout=60)
        if resp.status_code == 429:  # rate limited: back off and retry
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def archive_months(session: requests.Session, user: str) -> list[str]:
    """Return archive URLs like https://api.chess.com/pub/player/x/games/2024/05."""
    resp = _get(session, API.format(user=user.lower()))
    return resp.json().get("archives", [])


def month_filename(archive_url: str) -> str:
    year, month = archive_url.rstrip("/").split("/")[-2:]
    return f"{year}-{month}.pgn"


def fetch(user: str, out_dir: Path, session: requests.Session | None = None, refresh_last: int = 2) -> list[Path]:
    session = session or requests.Session()
    out_dir.mkdir(parents=True, exist_ok=True)
    archives = archive_months(session, user)
    written: list[Path] = []
    for i, url in enumerate(archives):
        path = out_dir / month_filename(url)
        is_recent = i >= len(archives) - refresh_last
        if path.exists() and not is_recent:
            continue
        pgn = _get(session, url + "/pgn").text
        path.write_text(pgn, encoding="utf-8")
        written.append(path)
        time.sleep(0.2)  # be polite to the API
    print(f"{user}: {len(archives)} months on chess.com, {len(written)} written to {out_dir}")
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--user", default="BoneSheep")
    p.add_argument("--out", type=Path, default=Path("data/raw/chesscom"))
    args = p.parse_args(argv)
    fetch(args.user, args.out)


if __name__ == "__main__":
    main()
