"""Building a training manifest out of GOL.

GOL is five times MoeSpeech and shaped differently: the download unit is a
whole work rather than one character, and its transcripts come from a single
ASR pass, so the mutual-CER filter that carried MoeSpeech has no counterpart.
"""

import csv

from training.scripts.prepare_gol import select_games


def _metadata(tmp_path, rows, text: str = "あ"):
    """A stand-in for GOL's metadata.tsv: game_id, speaker, text, path, duration."""
    p = tmp_path / "metadata.tsv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["game_id", "speaker", "text", "file_path", "duration"])
        for g, s, d in rows:
            w.writerow([g, s, text, f"{g}/{s}/x.wav", d])
    return p


def test_largest_first_reaches_the_target_with_fewest_tars(tmp_path):
    """Measured on the real metadata: 1,000 hours is 18 tars taken largest-first.
    A tar is 11 GB on average, so the ordering decides hours of download."""
    md = _metadata(tmp_path, [("big", "s1", 7200.0), ("big", "s2", 7200.0), ("small", "s3", 60.0)])
    (chosen,) = select_games(md, hours=1.0)
    assert chosen["game_id"] == "big"


def test_selection_stops_once_the_target_is_reached(tmp_path):
    md = _metadata(tmp_path, [(f"g{i}", "s", 3600.0) for i in range(5)])
    assert len(select_games(md, hours=1.0)) == 1


def test_a_game_reports_its_speakers_and_utterances(tmp_path):
    """Both decide what the next stage can use: the speaker filter needs the
    counts, and a tar of one speaker is worth less than a tar of thirty."""
    md = _metadata(tmp_path, [("g", "a", 60.0), ("g", "a", 60.0), ("g", "b", 60.0)])
    (g,) = select_games(md, hours=100.0)
    assert (g["speakers"], g["utterances"]) == (2, 3)


def test_selection_is_deterministic(tmp_path):
    md = _metadata(tmp_path, [(f"g{i}", "s", 3600.0) for i in range(20)])
    a = [g["game_id"] for g in select_games(md, hours=5.0)]
    b = [g["game_id"] for g in select_games(md, hours=5.0)]
    assert a == b


def test_the_hours_column_is_seconds(tmp_path):
    """metadata.tsv's duration is in seconds and --hours is in hours. The same
    unit confusion cost a test in the MoeSpeech pipeline."""
    md = _metadata(tmp_path, [("g", "s", 3600.0)])
    assert len(select_games(md, hours=0.5)) == 1
    assert select_games(md, hours=100.0)[0]["hours"] == 1.0
