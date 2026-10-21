import json
from pathlib import Path

import pytest

from kurdish_ins.instruction_gen import (BAD_RESPONSE, KEPT, NO_QUESTION, SPAN_NOT_FOUND, TOO_SHORT,
                                         InstructionGenStage, check, chunks)


STAGE_KWARGS = dict(allow_any={"IN": 0.5, "HI": 0.5}, forbid_any={"MT": 0.2, "NA": 0.5},
                    keep_sources=["theses"], mt_exempt_domains=["wiki.org"], min_words=3,
                    chunk_words=1500, model="m", temperature=0.3,
                    min_answer_words=3, workers=2, retries=1, limit=None, seed=0)


def stage(**kwargs):
    return InstructionGenStage(**{**STAGE_KWARGS, **kwargs})


def record(i, scores=None, source="web", text="one two three four five", domain=None):
    scores = {"IN": 0.9, "en": 0.8} if scores is None else scores
    return {"source": source, "id": f"id{i}", "url": f"http://s{i}.org", "domain": domain or f"s{i}.org",
            "text": text, "register_labels": sorted(label for label, p in scores.items() if p > 0.5),
            "register_scores": scores}


class TestKeeps:
    def test_any_allowed_label_above_its_threshold_keeps_it(self):
        assert stage().keeps(record(1, {"HI": 0.6, "re": 0.9}))

    def test_no_allowed_label_above_its_threshold(self):
        assert not stage().keeps(record(1, {"IN": 0.5, "LY": 0.9}))

    def test_a_forbidden_label_above_its_threshold_drops_it(self):
        assert not stage().keeps(record(1, {"IN": 0.9, "NA": 0.6}))

    def test_each_forbidden_label_has_its_own_threshold(self):
        assert not stage().keeps(record(1, {"IN": 0.9, "MT": 0.3}))
        assert stage().keeps(record(1, {"IN": 0.9, "MT": 0.2, "NA": 0.4}))

    def test_an_exempt_domain_skips_the_mt_check(self):
        assert stage().keeps(record(1, {"IN": 0.9, "MT": 0.3}, domain="wiki.org"))

    def test_an_exempt_domain_still_needs_the_other_rules(self):
        assert not stage().keeps(record(1, {"IN": 0.9, "NA": 0.6}, domain="wiki.org"))
        assert not stage().keeps(record(1, {"MT": 0.3}, domain="wiki.org"))

    def test_a_kept_source_needs_no_label(self):
        assert stage().keeps(record(1, {"MT": 0.9}, source="theses"))


class TestChunks:
    def test_a_short_text_is_one_chunk(self):
        assert chunks("a b\nc d", 10) == ["a b\nc d"]

    def test_cut_at_line_breaks(self):
        assert chunks("a b\nc d\ne f", 4) == ["a b\nc d", "e f"]

    def test_a_long_line_is_a_chunk_of_its_own(self):
        assert chunks("a\nb c d e\nf", 2) == ["a", "b c d e", "f"]


class TestCheck:
    TEXT = "First sentence here. Second sentence follows. Third one."

    def test_a_verbatim_span_is_kept(self):
        assert check([{"question": " Q? ", "answer_span": "Second sentence follows."}], self.TEXT, 3) == \
            (KEPT, ("Q?", "Second sentence follows.", 21))

    def test_a_rephrased_span_is_not_found(self):
        assert check([{"question": "Q?", "answer_span": "Second sentence is next."}],
                     self.TEXT, 3) == (SPAN_NOT_FOUND, None)

    def test_a_short_answer(self):
        assert check([{"question": "Q?", "answer_span": "Third one."}], self.TEXT, 3) == (TOO_SHORT, None)

    def test_an_empty_list(self):
        assert check([], self.TEXT, 3) == (NO_QUESTION, None)

    @pytest.mark.parametrize("items", [
        {"unparsed": "x"},
        ["x"],
        [{"answer_span": "First sentence here."}],
        [{"question": "Q?", "answer_span": ["First sentence here."]}],  # a list is the multi-span schema
        [{"question": "Q?", "answer_span": " "}],
    ])
    def test_a_bad_response(self, items):
        assert check(items, self.TEXT, 3) == (BAD_RESPONSE, None)


class FakeStage(InstructionGenStage):
    """Answers each chunk with its first three words, and records the texts it was asked about."""

    def __init__(self, **kwargs):
        super().__init__(**{**STAGE_KWARGS, **kwargs})
        self.asked = []

    def ask(self, text):
        self.asked.append(text)
        return [{"question": "Q?", "answer_span": " ".join(text.split()[:3])}]

    @staticmethod
    def _make_client():
        return None


def write_manifest(input_dir, records):
    input_dir.mkdir()
    with open(input_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


class TestRun:
    def test_writes_pairs_responses_and_stats(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1), record(2, {"NA": 0.9, "ne": 0.9})])
        FakeStage().run(tmp_path / "in", tmp_path / "out")
        [pair] = [json.loads(l) for l in open(tmp_path / "out" / "manifest.jsonl")]
        assert (pair["id"], pair["question"], pair["answer"], pair["answer_start"]) == \
            ("id1", "Q?", "one two three", 0)
        stats = json.load(open(tmp_path / "out" / "stats.json"))
        assert stats["outcomes"] == {KEPT: 1}

    def test_a_second_run_skips_answered_chunks(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1), record(2)])
        FakeStage(limit=1).run(tmp_path / "in", tmp_path / "out")
        second = FakeStage()
        second.run(tmp_path / "in", tmp_path / "out")
        assert len(second.asked) == 1
        assert len(open(tmp_path / "out" / "manifest.jsonl").readlines()) == 2

    def test_the_same_seed_picks_the_same_chunks(self):
        chunks_ = [{"key": str(i)} for i in range(50)]
        assert FakeStage(limit=5).select(chunks_) == FakeStage(limit=5).select(chunks_)


class TestManifest:
    def test_the_classify_manifest_of_the_run(self, tmp_path):
        assert stage().manifest(tmp_path) == tmp_path / "manifest.jsonl"

    def test_input_manifest_replaces_it(self, tmp_path):
        manifest = "data/modal/full/manifest.jsonl"
        assert stage(input_manifest=manifest).manifest(tmp_path) == Path(manifest)
