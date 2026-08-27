"""Building a training manifest out of MoeSpeech.

Everything here guards one property: the manifest must describe the audio
truthfully. A wrong `start` or `duration` does not raise -- it trains the model
on speech that does not match its text, and the only symptom is a model that
never quite becomes intelligible.
"""

import csv

import pytest
import typer

from training.scripts.prepare_moespeech import select_characters


def _info_csv(tmp_path, rows):
    """A stand-in for the dataset's info.csv: name, num_files, minutes, f0."""
    path = tmp_path / "info.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "num_files", "total_duration_min", "f0_mean"])
        w.writerows(rows)
    return path


def test_selection_stops_once_the_target_is_reached(tmp_path):
    info = _info_csv(
        tmp_path, [("a", 100, 60.0, 300.0), ("b", 100, 60.0, 300.0), ("c", 100, 60.0, 300.0)]
    )
    chosen = select_characters(info, hours=1.0, order="largest")
    assert len(chosen) == 1, chosen


def test_the_target_is_hours_and_the_column_is_minutes(tmp_path):
    """total_duration_min is minutes; --hours is hours. Two hours of 30-minute
    speakers is four of them, not one. Every other case here is satisfied by a
    single speaker, so this is the only test that walks the accumulator far
    enough to notice a lost `* 60` or a summed `num_files`."""
    rows = [(f"c{i}", 100, 30.0, 300.0) for i in range(10)]
    chosen = select_characters(_info_csv(tmp_path, rows), hours=2.0, order="largest")
    assert len(chosen) == 4, chosen
    assert sum(c["total_duration_min"] for c in chosen) == 120.0


def test_a_japanese_speaker_name_survives_the_read(tmp_path):
    """Default encoding here is cp932, and every name in this dataset is
    Japanese. Read without an explicit utf-8 the name comes back as mojibake
    rather than raising, and Task 2 turns it straight into a zip URL."""
    info = _info_csv(tmp_path, [("ずんだもん", 100, 600.0, 300.0)])
    (chosen,) = select_characters(info, hours=1.0, order="largest")
    assert chosen["name"] == "ずんだもん"


def test_largest_first_takes_the_biggest(tmp_path):
    info = _info_csv(tmp_path, [("small", 10, 6.0, 300.0), ("big", 100, 600.0, 300.0)])
    (chosen,) = select_characters(info, hours=1.0, order="largest")
    assert chosen["name"] == "big"


def test_selection_is_deterministic(tmp_path):
    """A re-run after an interrupted download must ask for the same zips."""
    rows = [(f"c{i}", 100, 30.0, 300.0) for i in range(20)]
    info = _info_csv(tmp_path, rows)
    first = select_characters(info, hours=5.0, order="largest")
    second = select_characters(info, hours=5.0, order="largest")
    assert [c["name"] for c in first] == [c["name"] for c in second]


def test_random_order_is_also_deterministic(tmp_path):
    """Reproducibility does not depend on which ordering was chosen."""
    rows = [(f"c{i}", 100, 30.0, 300.0) for i in range(20)]
    info = _info_csv(tmp_path, rows)
    a = select_characters(info, hours=5.0, order="random")
    b = select_characters(info, hours=5.0, order="random")
    assert [c["name"] for c in a] == [c["name"] for c in b]


def test_asking_for_more_than_exists_returns_everything(tmp_path):
    """Rather than raising: the caller asked for an upper bound, not a promise."""
    info = _info_csv(tmp_path, [("a", 10, 6.0, 300.0)])
    assert len(select_characters(info, hours=1000.0, order="largest")) == 1


def test_an_unknown_order_is_refused(tmp_path):
    """A typo in --order must not silently fall through to info.csv's own
    order, which would look just as deterministic on re-run."""
    info = _info_csv(tmp_path, [("a", 10, 6.0, 300.0)])
    with pytest.raises(typer.BadParameter):
        select_characters(info, hours=1.0, order="larget")
