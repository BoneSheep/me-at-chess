from me_at_chess.fetch_chesscom import fetch, month_filename


class FakeResp:
    def __init__(self, status=200, json_data=None, text=""):
        self.status_code, self._json, self.text = status, json_data, text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        if url.endswith("/archives"):
            base = "https://api.chess.com/pub/player/bonesheep/games"
            return FakeResp(json_data={"archives": [f"{base}/2023/12", f"{base}/2024/01", f"{base}/2024/02"]})
        return FakeResp(text=f"[Event \"{url}\"]\n\n1. e4 *\n")


def test_month_filename():
    assert month_filename("https://api.chess.com/pub/player/x/games/2024/05") == "2024-05.pgn"


def test_fetch_downloads_and_skips_old_existing_months(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    (tmp_path / "2023-12.pgn").write_text("old")
    (tmp_path / "2024-02.pgn").write_text("stale")
    s = FakeSession()
    written = fetch("BoneSheep", tmp_path, session=s)
    names = sorted(p.name for p in written)
    # 2023-12 exists and is not recent -> skipped; the last two months are (re)fetched
    assert names == ["2024-01.pgn", "2024-02.pgn"]
    assert (tmp_path / "2023-12.pgn").read_text() == "old"
    assert "2024/02/pgn" in (tmp_path / "2024-02.pgn").read_text()
    assert s.calls[0].endswith("/player/bonesheep/games/archives")
