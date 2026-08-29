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
    A tar is 11 GB on average, so the ordering decides hours of download.

    Every signal here except duration points at the other game: `a-many-short`
    has more utterances, more speakers, sorts first by id and comes first in
    the file. Only summed duration picks `z-one-long`. A fixture where the
    biggest game also happened to be the first or the busiest would pass under
    a sort on any of those -- or under no sort at all -- and a size-blind order
    needs about 57 tars for the 1,000 hours that cost 18 here, some 600 GB of
    download that buys nothing.
    """
    rows = [("a-many-short", f"s{i}", 60.0) for i in range(30)]
    rows += [("z-one-long", "s", 7200.0)] * 2
    (chosen,) = select_games(_metadata(tmp_path, rows), hours=1.0)
    assert chosen["game_id"] == "z-one-long"


def test_selection_stops_once_the_target_is_reached(tmp_path):
    md = _metadata(tmp_path, [(f"g{i}", "s", 3600.0) for i in range(5)])
    assert len(select_games(md, hours=1.0)) == 1


def test_a_game_reports_its_speakers_and_utterances(tmp_path):
    """Both decide what the next stage can use: the speaker filter needs the
    counts, and a tar of one speaker is worth less than a tar of thirty."""
    md = _metadata(tmp_path, [("g", "a", 60.0), ("g", "a", 60.0), ("g", "b", 60.0)])
    (g,) = select_games(md, hours=100.0)
    assert (g["speakers"], g["utterances"]) == (2, 3)


def test_ties_break_on_game_id_so_a_resumed_run_asks_for_the_same_tars(tmp_path):
    """Games of equal length must come out in a stated order, not in whatever
    order the accumulating dict happens to hold them.

    Asserting instead that two calls agree would pass for any pure function,
    including one that leaves equal games in hash order: within one process the
    two calls agree, and across processes they do not. A resumed run would then
    re-download tars it already has and skip ones it does not, at 11 GB each.
    """
    md = _metadata(tmp_path, [(g, "s", 3600.0) for g in ["gc", "ga", "gb"]])
    assert [g["game_id"] for g in select_games(md, hours=2.0)] == ["ga", "gb"]


def test_the_hours_column_is_seconds(tmp_path):
    """metadata.tsv's duration is in seconds and --hours is in hours. The same
    unit confusion cost a test in the MoeSpeech pipeline."""
    md = _metadata(tmp_path, [("g", "s", 3600.0)])
    assert len(select_games(md, hours=0.5)) == 1
    assert select_games(md, hours=100.0)[0]["hours"] == 1.0
