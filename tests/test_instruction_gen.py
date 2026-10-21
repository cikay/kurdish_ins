import json
from pathlib import Path

import pytest

from types import SimpleNamespace

from kurdish_ins.instruction_gen import (BAD_RESPONSE, FAILED, KEPT, NO_QUESTION, SPAN_NOT_FOUND, TOO_SHORT,
                                         InstructionGenStage, batch_reply, check, chunks)


STAGE_KWARGS = dict(allow_any={"IN": 0.5, "HI": 0.5}, forbid_any={"MT": 0.2, "NA": 0.5},
                    keep_sources=["theses"], mt_exempt_domains=["wiki.org"], min_words=3,
                    chunk_words=1500, model="m", temperature=0.3,
                    min_answer_words=3, limit=None, seed=0, batch_chunks=10)


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
        assert check({"question": " Q? ", "answer_spans": ["Second sentence follows."]}, self.TEXT, 3) == \
            (KEPT, ("Q?", "Second sentence follows.", [(21, 45)]))

    def test_spans_are_joined_in_order(self):
        outcome, (_, answer, offsets) = check(
            {"question": "Q?", "answer_spans": ["First sentence here.", "Third one."]}, self.TEXT, 3)
        assert (outcome, answer, offsets) == (KEPT, "First sentence here. Third one.", [(0, 20), (46, 56)])

    def test_a_skipped_line_break_is_a_blank_line(self):
        outcome, (_, answer, _) = check(
            {"question": "Q?", "answer_spans": ["One two three.", "Seven eight nine."]},
            "One two three.\nFour five six.\nSeven eight nine.", 3)
        assert (outcome, answer) == (KEPT, "One two three.\n\nSeven eight nine.")

    def test_any_number_of_spans(self):
        sentences = [f"Sentence {i} here." for i in range(10)]
        outcome, (_, answer, _) = check({"question": "Q?", "answer_spans": sentences[::2]}, " ".join(sentences), 3)
        assert (outcome, answer) == (KEPT, " ".join(sentences[::2]))

    def test_spans_out_of_order_are_not_found(self):
        assert check({"question": "Q?", "answer_spans": ["Third one.", "First sentence here."]},
                     self.TEXT, 3) == (SPAN_NOT_FOUND, None)

    def test_overlapping_spans_are_not_found(self):
        assert check({"question": "Q?", "answer_spans": ["First sentence here.", "here. Second"]},
                     self.TEXT, 3) == (SPAN_NOT_FOUND, None)

    def test_a_rephrased_span_is_not_found(self):
        assert check({"question": "Q?", "answer_spans": ["Second sentence is next."]},
                     self.TEXT, 3) == (SPAN_NOT_FOUND, None)

    def test_a_short_answer(self):
        assert check({"question": "Q?", "answer_spans": ["Third one."]}, self.TEXT, 3) == (TOO_SHORT, None)

    def test_null_is_no_question(self):
        assert check(None, self.TEXT, 3) == (NO_QUESTION, None)

    @pytest.mark.parametrize("reply", [
        {"unparsed": "x"},
        [{"question": "Q?", "answer_spans": ["First sentence here."]}],  # a list, not one object
        {"answer_spans": ["First sentence here."]},
        {"question": "Q?", "answer_span": "First sentence here."},  # the single-span schema
        {"question": "Q?", "answer_spans": []},
        {"question": "Q?", "answer_spans": ["First sentence here.", " "]},
    ])
    def test_a_bad_response(self, reply):
        assert check(reply, self.TEXT, 3) == (BAD_RESPONSE, None)


# Where a run given tmp_path / "out" writes: the prompt and thinking level are added to its name
OUT = "out-m-multi-span-thinking-low"


def write_manifest(input_dir, records):
    input_dir.mkdir()
    with open(input_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


class TestRun:
    def test_writes_pairs_responses_and_stats(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1), record(2, {"NA": 0.9, "ne": 0.9})])
        BatchStage(FakeBatchClient()).run(tmp_path / "in", tmp_path / "out")
        [pair] = [json.loads(l) for l in open(tmp_path / OUT / "manifest.jsonl")]
        assert (pair["id"], pair["question"], pair["answer"], pair["answer_spans"]) == \
            ("id1", "Q?", "one two three", [{"start": 0, "end": 13}])
        stats = json.load(open(tmp_path / OUT / "stats.json"))
        assert stats["outcomes"] == {KEPT: 1}
        assert (stats["question_words_avg"], stats["answer_words_avg"]) == (1, 3)

    def test_saves_each_chunks_tokens_and_adds_them_up(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1), record(2)])
        BatchStage(FakeBatchClient()).run(tmp_path / "in", tmp_path / "out")
        responses = [json.loads(l) for l in open(tmp_path / OUT / "responses.jsonl")]
        assert [r["tokens"] for r in responses] == [{"input": 5, "output": 2, "thinking": 1}] * 2
        stats = json.load(open(tmp_path / OUT / "stats.json"))
        assert stats["tokens"] == {"input": 10, "output": 4, "thinking": 2}
        assert stats["tokens_avg"] == {"input": 5, "output": 2, "thinking": 1}

    def test_rows_saved_without_tokens_are_left_out_of_the_token_average(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1), record(2)])
        (tmp_path / OUT).mkdir()
        with open(tmp_path / OUT / "responses.jsonl", "w") as rf:
            rf.write(json.dumps({"key": "id1#0", "outcome": NO_QUESTION, "register_labels": []}) + "\n")
        BatchStage(FakeBatchClient()).run(tmp_path / "in", tmp_path / "out")
        stats = json.load(open(tmp_path / OUT / "stats.json"))
        assert stats["tokens_avg"] == {"input": 5, "output": 2, "thinking": 1}

    def test_a_second_run_skips_answered_chunks(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1), record(2)])
        (tmp_path / OUT).mkdir()
        with open(tmp_path / OUT / "responses.jsonl", "w") as rf:
            rf.write(json.dumps({"key": "id1#0", "outcome": NO_QUESTION, "register_labels": []}) + "\n")
        client = FakeBatchClient()
        BatchStage(client).run(tmp_path / "in", tmp_path / "out")
        assert [r["key"] for r in client.uploaded[0]] == ["id2#0"]
        assert len(open(tmp_path / OUT / "responses.jsonl").readlines()) == 2

    def test_a_limited_run_writes_to_a_limit_folder_beside_the_runs(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1), record(2)])
        BatchStage(FakeBatchClient(), limit=1).run(tmp_path / "in", tmp_path / "instruction-gen" / "full")
        assert [p.name for p in (tmp_path / "instruction-gen").iterdir()] == ["limit-1-m-multi-span-thinking-low"]
        assert len(open(tmp_path / "instruction-gen" / "limit-1-m-multi-span-thinking-low" / "manifest.jsonl").readlines()) == 1

    def test_the_same_seed_picks_the_same_chunks(self):
        chunks_ = [{"key": str(i)} for i in range(50)]
        assert stage(limit=5).select(chunks_) == stage(limit=5).select(chunks_)


class TestManifest:
    def test_the_classify_manifest_of_the_run(self, tmp_path):
        assert stage().manifest(tmp_path) == tmp_path / "manifest.jsonl"

    def test_input_manifest_replaces_it(self, tmp_path):
        manifest = "data/modal/full/manifest.jsonl"
        assert stage(input_manifest=manifest).manifest(tmp_path) == Path(manifest)


class TestBatchReply:
    def test_the_text_parts_are_the_reply(self):
        line = {"key": "k", "response": {
            "candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True},
                                                  {"text": '{"question": "Q?", '}, {"text": '"answer_spans": ["a"]}'}]}}],
            "usageMetadata": {"promptTokenCount": 10}}}
        assert batch_reply(line) == ({"question": "Q?", "answer_spans": ["a"]}, {"promptTokenCount": 10})

    def test_an_error_line_failed(self):
        assert batch_reply({"key": "k", "error": {"code": 500}}) == (FAILED, {})


class QuotaFull(Exception):
    code = 429


class FakeBatchClient:
    """A Batch API that answers each request with the first three words of its passage, or
    with an error for the keys in `errors`. Job n is batches/n, made from the n-th upload;
    the jobs in `expired` end JOB_STATE_EXPIRED. Every job has ended the first time it is
    checked. With `quota`, a job is refused with 429 while that many have not ended."""

    def __init__(self, errors=(), expired=(), quota=None):
        self.errors = set(errors)
        self.expired = set(expired)
        self.quota = quota
        self.uploaded = []
        self.created = 0
        self.active = set()
        self.most_active = 0
        self.deleted = []
        self.files = SimpleNamespace(upload=self.upload, download=self.download, delete=self.delete)
        self.batches = SimpleNamespace(create=self.create, get=self.get)

    def upload(self, file, config):
        self.uploaded.append([json.loads(line) for line in open(file, encoding="utf-8")])
        return SimpleNamespace(name=f"files/in-{len(self.uploaded)}")

    def create(self, model, src, config):
        if self.quota is not None and len(self.active) >= self.quota:
            raise QuotaFull()
        self.created += 1
        name = f"batches/{src.removeprefix('files/in-')}"
        self.active.add(name)
        self.most_active = max(self.most_active, len(self.active))
        return SimpleNamespace(name=name)

    def delete(self, name):
        self.deleted.append(name)

    def get(self, name):
        self.active.discard(name)
        n = name.removeprefix("batches/")
        state = "JOB_STATE_EXPIRED" if name in self.expired else "JOB_STATE_SUCCEEDED"
        return SimpleNamespace(name=name, state=SimpleNamespace(name=state),
                               dest=SimpleNamespace(file_name=f"files/out-{n}"), error=None)

    def download(self, file):
        lines = []
        for request in self.uploaded[int(file.removeprefix("files/out-")) - 1]:
            if request["key"] in self.errors:
                lines.append({"key": request["key"], "error": {"code": 500}})
                continue
            passage = request["request"]["contents"][0]["parts"][0]["text"].split("Passage:\n")[1]
            reply = {"question": "Q?", "answer_spans": [" ".join(passage.split()[:3])]}
            lines.append({"key": request["key"], "response": {
                "candidates": [{"content": {"parts": [{"text": json.dumps(reply)}]}}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "thoughtsTokenCount": 1}}})
        return "\n".join(json.dumps(line) for line in lines).encode("utf-8")


class BatchStage(InstructionGenStage):
    def __init__(self, client, **kwargs):
        super().__init__(**{**STAGE_KWARGS, **kwargs})
        self.client = client

    def _make_client(self):
        return self.client


class TestBatch:
    def test_writes_pairs_and_forgets_the_job(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1), record(2)])
        client = FakeBatchClient()
        BatchStage(client).run(tmp_path / "in", tmp_path / "out")
        pairs = [json.loads(l) for l in open(tmp_path / OUT / "manifest.jsonl")]
        assert sorted(p["id"] for p in pairs) == ["id1", "id2"]
        assert {p["answer"] for p in pairs} == {"one two three"}
        assert not (tmp_path / OUT / "batch_job.json").exists()
        assert not (tmp_path / OUT / "batch_requests.jsonl").exists()

    def test_a_request_is_the_prompt_with_the_generation_config(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1)])
        client = FakeBatchClient()
        BatchStage(client).run(tmp_path / "in", tmp_path / "out")
        [[request]] = client.uploaded
        assert request["key"] == "id1#0"
        assert request["request"]["generation_config"] == {
            "response_mime_type": "application/json", "temperature": 0.3,
            "thinking_config": {"thinking_level": "LOW"}}

    def test_a_failed_request_is_sent_again_by_the_next_run(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1), record(2)])
        BatchStage(FakeBatchClient(errors={"id1#0"})).run(tmp_path / "in", tmp_path / "out")
        assert [json.loads(l)["key"] for l in open(tmp_path / OUT / "responses.jsonl")] == ["id2#0"]
        client = FakeBatchClient()
        BatchStage(client).run(tmp_path / "in", tmp_path / "out")
        assert [r["key"] for r in client.uploaded[0]] == ["id1#0"]

    def test_a_job_of_an_earlier_run_is_waited_for_not_submitted_again(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1)])
        (tmp_path / OUT).mkdir()
        (tmp_path / OUT / "batch_job.json").write_text(json.dumps({"jobs": {"batches/1": ["id1#0"]}}))
        client = FakeBatchClient()
        client.uploaded.append([BatchStage(client).batch_request(
            {"key": "id1#0", "text": "one two three four five"})])
        BatchStage(client).run(tmp_path / "in", tmp_path / "out")
        assert client.created == 0
        assert len(open(tmp_path / OUT / "manifest.jsonl").readlines()) == 1

    def test_the_chunks_are_split_into_jobs_of_batch_chunks(self, tmp_path):
        write_manifest(tmp_path / "in", [record(i) for i in range(5)])
        client = FakeBatchClient()
        BatchStage(client, batch_chunks=2).run(tmp_path / "in", tmp_path / "out")
        assert [len(requests) for requests in client.uploaded] == [2, 2, 1]
        assert len(open(tmp_path / OUT / "manifest.jsonl").readlines()) == 5

    def test_a_resumed_run_sends_only_the_chunks_no_waiting_job_holds(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1), record(2)])
        (tmp_path / OUT).mkdir()
        (tmp_path / OUT / "batch_job.json").write_text(json.dumps({"jobs": {"batches/1": ["id1#0"]}}))
        client = FakeBatchClient()
        client.uploaded.append([BatchStage(client).batch_request(
            {"key": "id1#0", "text": "one two three four five"})])
        BatchStage(client).run(tmp_path / "in", tmp_path / "out")
        assert [r["key"] for r in client.uploaded[1]] == ["id2#0"]
        assert len(open(tmp_path / OUT / "manifest.jsonl").readlines()) == 2

    def test_a_full_quota_holds_jobs_back_until_others_end(self, tmp_path):
        write_manifest(tmp_path / "in", [record(i) for i in range(5)])
        client = FakeBatchClient(quota=2)
        BatchStage(client, batch_chunks=1).run(tmp_path / "in", tmp_path / "out")
        assert (client.created, client.most_active) == (5, 2)
        assert client.deleted  # the uploads of the refused jobs
        assert len(open(tmp_path / OUT / "manifest.jsonl").readlines()) == 5
        assert not (tmp_path / OUT / "batch_job.json").exists()

    def test_a_job_over_the_quota_on_its_own_stops_the_run(self, tmp_path):
        write_manifest(tmp_path / "in", [record(1)])
        with pytest.raises(SystemExit, match="Lower batch_chunks"):
            BatchStage(FakeBatchClient(quota=0)).run(tmp_path / "in", tmp_path / "out")

    def test_an_expired_job_loses_only_its_own_chunks(self, tmp_path):
        write_manifest(tmp_path / "in", [record(i) for i in range(4)])
        client = FakeBatchClient(expired={"batches/1"})
        with pytest.raises(SystemExit, match="batches/1 ended JOB_STATE_EXPIRED"):
            BatchStage(client, batch_chunks=2).run(tmp_path / "in", tmp_path / "out")
        expired_keys = [r["key"] for r in client.uploaded[0]]
        saved = {json.loads(l)["key"] for l in open(tmp_path / OUT / "responses.jsonl")}
        assert saved == {r["key"] for r in client.uploaded[1]}
        assert not (tmp_path / OUT / "batch_job.json").exists()
        client = FakeBatchClient()
        BatchStage(client, batch_chunks=2).run(tmp_path / "in", tmp_path / "out")
        assert [r["key"] for r in client.uploaded[0]] == expired_keys


class TestWait:
    def test_the_wait_doubles_up_to_ten_minutes(self, monkeypatch):
        states = ["JOB_STATE_PENDING"] + ["JOB_STATE_RUNNING"] * 8 + ["JOB_STATE_SUCCEEDED"]
        client = SimpleNamespace(batches=SimpleNamespace(
            get=lambda name: SimpleNamespace(name=name, state=SimpleNamespace(name=states.pop(0)))))
        waits = []
        monkeypatch.setattr("kurdish_ins.instruction_gen.time.sleep", waits.append)
        stage_ = stage()
        stage_._client = client
        assert [job.state.name for job in stage_._wait(["batches/1"])] == ["JOB_STATE_SUCCEEDED"]
        assert waits == [10, 20, 40, 80, 160, 320, 600, 600, 600]
