# Over-the-board games

Drop PGN files of OTB / FIDE games here (any filename ending in `.pgn` or `.pgn.gz`).
Games are matched to Ruben by player name (`Ruben Thienpont`, `Thienpont, Ruben`, …,
see `PLAYER_ALIASES` in `src/me_at_chess/__init__.py`).

Games without ratings get `--default-elo` (1500). FIDE doesn't publish game scores,
so these usually come from a club or tournament database (e.g. a KBSB/FRBE
interclub PGN, or a ChessBase / 365chess export).
