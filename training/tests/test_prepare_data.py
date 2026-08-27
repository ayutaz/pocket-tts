"""prepare_data.py keeps chapter audio files whole: utterances sharing a
chapter share one manifest `path`, distinguished only by `start`."""

import gzip
import json
from pathlib import Path

import huggingface_hub

from training.scripts import prepare_data


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_utterances_in_one_chapter_share_the_manifest_path(tmp_path, monkeypatch):
    chapters_json = tmp_path / "chapters.json"
    manifest_json = tmp_path / "manifest.json"
    _write_jsonl(
        chapters_json,
        [
            {
                "chapter_filepath": "book1/ch1",
                "url": "http://example.invalid/ch1.mp3",
                "duration": 9.0,
                "utterances": [
                    {"audio_filepath": "book1/ch1_utt0", "offset": 0.0, "duration": 5.0},
                    {"audio_filepath": "book1/ch1_utt1", "offset": 5.0, "duration": 4.0},
                ],
            }
        ],
    )
    _write_jsonl(
        manifest_json,
        [
            {
                "audio_filepath": "book1/ch1_utt0",
                "duration": 5.0,
                "normalized_text": "hello world",
                "speaker": "spk1",
                "set": "train",
            },
            {
                "audio_filepath": "book1/ch1_utt1",
                "duration": 4.0,
                "normalized_text": "goodbye",
                "speaker": "spk1",
                "set": "dev",
            },
        ],
    )

    def fake_hf_hub_download(repo, filename, repo_type):
        return str(chapters_json) if "chapters" in filename else str(manifest_json)

    def fake_download(url, dest, **kwargs):
        # Simulate a successful download: touch the destination.
        open(dest, "w").close()

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(prepare_data, "download", fake_download)

    audio_out = tmp_path / "downloads"
    out_dir = tmp_path / "manifests"
    audio_out.mkdir()
    out_dir.mkdir()

    train_m, valid_m = prepare_data.prepare_hifitts2(audio_out, out_dir, None, 1)

    train_recs = [json.loads(line) for line in train_m.open(encoding="utf-8")]
    valid_recs = [json.loads(line) for line in valid_m.open(encoding="utf-8")]
    assert len(train_recs) == 1
    assert len(valid_recs) == 1

    # No per-utterance splitting: both utterances point at the same chapter file.
    assert train_recs[0]["path"] == valid_recs[0]["path"]
    assert Path(train_recs[0]["path"]) == audio_out / "hifitts2_audio" / "ch1.mp3"

    assert train_recs[0]["start"] == 0.0
    assert train_recs[0]["duration"] == 5.0
    assert train_recs[0]["transcript"] == "hello world"

    assert valid_recs[0]["start"] == 5.0
    assert valid_recs[0]["duration"] == 4.0
    assert valid_recs[0]["transcript"] == "goodbye"

    # Exactly one download for the whole chapter, not one per utterance.
    assert list(audio_out.rglob("*.mp3")) == [audio_out / "hifitts2_audio" / "ch1.mp3"]


def test_hf_alignments_join_on_utterance_id_not_path(tmp_path, monkeypatch):
    """Rows that share a chapter file's `path` still join correctly, because
    the join key is each row's `audio_filepath`, not its `path`."""
    # published alignments: two utterances, utterance-relative timestamps
    snap = tmp_path / "snap"
    (snap / "train").mkdir(parents=True)
    with gzip.open(snap / "train" / "train_aligned-000-of-001.jsonl.gz", "wt") as w:
        w.write(
            json.dumps(
                {
                    "audio_filepath": "book1/utt0.flac",
                    "words": [{"word": "hi", "start": 0.1, "end": 0.4}],
                }
            )
            + "\n"
        )
    with gzip.open(snap / "eval_aligned.jsonl.gz", "wt") as w:
        pass
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))

    # raw manifest: both rows point at the SAME chapter file, distinct ids
    manifest = tmp_path / "raw.jsonl"
    rows = [
        {
            "path": "/audio/book1/chapter.flac",
            "start": 0.0,
            "duration": 2.0,
            "transcript": "hi",
            "audio_filepath": "book1/utt0.flac",
        },
        {
            "path": "/audio/book1/chapter.flac",
            "start": 2.0,
            "duration": 2.0,
            "transcript": "yo",
            "audio_filepath": "book1/utt9.flac",
        },
    ]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))

    out = tmp_path / "aligned.jsonl"
    prepare_data.attach_hf_alignments(manifest, out, "hf://x/y", tmp_path)
    got = [json.loads(line) for line in out.read_text().splitlines()]
    assert got[0]["words"][0]["word"] == "hi"  # matched by id
    assert "words" not in got[1]  # unmatched row kept, no words


def _one_line_manifest(tmp_path, lines=1):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("".join(json.dumps({"duration": 1.0}) + "\n" for _ in range(lines)))
    return manifest


def test_align_forwards_the_segmenter_to_the_aligner(tmp_path, monkeypatch):
    """A language written without spaces must not reach the aligner under the
    whitespace segmenter: it returns one word per utterance, so the aligner
    emits a single span, the loader finds no cut point, and the voice prompt
    silently comes from the utterance being predicted. Nothing raises. The
    caller is the only one that knows which language its manifest is in, so it
    has to be able to say."""
    manifest = _one_line_manifest(tmp_path)
    out = tmp_path / "train_aligned.jsonl"
    cmds = []

    def fake_run(cmd, **kwargs):
        cmds.append(cmd)
        Path(cmd[4]).write_text("")  # the .partial the aligner streams into

    monkeypatch.setattr(prepare_data.subprocess, "run", fake_run)
    prepare_data.align(manifest, out, 1, "some/model", "manifest", segmenter="japanese")

    assert cmds[-1][cmds[-1].index("--segmenter") + 1] == "japanese", cmds[-1]


def test_align_defaults_to_the_whitespace_segmenter(tmp_path, monkeypatch):
    """The corpus this script prepares is English, and the two existing call
    sites pass their arguments positionally. The default is align_data's own
    default too, so what those two send is what they sent before."""
    manifest = _one_line_manifest(tmp_path)
    out = tmp_path / "train_aligned.jsonl"
    cmds = []

    def fake_run(cmd, **kwargs):
        cmds.append(cmd)
        Path(cmd[4]).write_text("")

    monkeypatch.setattr(prepare_data.subprocess, "run", fake_run)
    prepare_data.align(manifest, out, 1, "some/model", "manifest")

    assert cmds[-1][cmds[-1].index("--segmenter") + 1] == "whitespace", cmds[-1]


def test_the_sharded_branch_forwards_the_segmenter_too(tmp_path, monkeypatch):
    """Each branch builds its own command line, so an option that arrives on
    one GPU can still be missing on eight -- and eight GPUs is what a corpus
    large enough to matter is aligned on."""
    manifest = _one_line_manifest(tmp_path, lines=2)
    out = tmp_path / "train_aligned.jsonl"
    cmds = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            cmds.append(cmd)
            # what the shard leaves behind, which align() merges and unlinks
            out.with_suffix(f".shard{cmd[cmd.index('--shard') + 1]}").write_text("")

        def wait(self):
            return 0

    monkeypatch.setattr(prepare_data.subprocess, "Popen", FakeProc)
    prepare_data.align(manifest, out, 2, "some/model", "manifest", segmenter="japanese")

    assert len(cmds) == 2, cmds
    for cmd in cmds:
        assert cmd[cmd.index("--segmenter") + 1] == "japanese", cmd
