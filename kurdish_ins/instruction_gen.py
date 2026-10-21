"""Stage — Ask Gemini for one question per chosen entry, answered by the entry's own text.

Reads the classify manifest and keeps the entries worth an instruction: those with a
score above its threshold for any label of `allow_any` and for no label of `forbid_any`,
plus every entry of `keep_sources` whatever its scores. Entries of `mt_exempt_domains`
skip the MT check of `forbid_any`, and only that one. A long entry is cut into chunks
at line breaks and each chunk is asked about on its own.

The model must answer with one span copied from the chunk (see `prompts.py`). A pair is
kept only when the span appears verbatim in the chunk and is long enough, so the answer is
always the source's own Kurmanji, never the model's.

With `batch` set, the chunks go to Gemini's Batch API as one job, at half the price, and
the stage waits for it (up to 24 hours). Otherwise it sends them one call at a time.

Each chunk's outcome is appended to `responses.jsonl` as soon as it comes back, so a run
that stops is resumed by running it again: chunks already answered are skipped. A batch
job's name is kept in `batch_job.json` until its results are saved, so a run stopped while
waiting waits for the same job again rather than paying for a second one.
"""

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kurdish_ins.prompts import SINGLE_SPAN_PROMPT
from kurdish_ins.stage import Stage, run_name, stage_config

# What became of a chunk. Only `kept` writes a pair to the manifest.
KEPT = "kept"
NO_QUESTION = "no_question"        # the model returned null
BAD_RESPONSE = "bad_response"      # not JSON, or not the shape the prompt asks for
SPAN_NOT_FOUND = "span_not_found"  # the span is not in the chunk verbatim
TOO_SHORT = "too_short"            # the answer has fewer than min_answer_words words

# What ask() returns when every retry failed, and a batch result line holds an error. Not
# None, which is the model's "no question".
FAILED = object()

# Thinking tokens are billed as output, and gemini-3.8-flash cannot turn thinking off: LOW is
# its lowest level
THINKING_LEVEL = "LOW"

# A batch job is checked this often while it runs
BATCH_POLL_SECONDS = 60
BATCH_DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def chunks(text: str, max_words: int) -> list[str]:
    """Split a text at line breaks into pieces of at most `max_words` words.

    A single line longer than that becomes a piece of its own rather than being cut
    mid-sentence.
    """
    pieces, lines, words = [], [], 0
    for line in text.split("\n"):
        n = len(line.split())
        if lines and words + n > max_words:
            pieces.append("\n".join(lines).strip())
            lines, words = [], 0
        lines.append(line)
        words += n
    if lines:
        pieces.append("\n".join(lines).strip())
    return pieces


def prompt(text: str) -> str:
    return f"{SINGLE_SPAN_PROMPT}\n\nPassage:\n{text}"


def parse_reply(text: str | None):
    """The JSON the model wrote, or a dict check() calls a bad response when it is not JSON."""
    try:
        return json.loads(text or "null")
    except json.JSONDecodeError:
        return {"unparsed": text}


def batch_reply(line: dict):
    """(reply, usage) of one line of a batch job's results. The reply is FAILED when the
    line holds an error; `usage` is the response's usageMetadata."""
    response = line.get("response")
    if not response:
        return FAILED, {}
    candidates = response.get("candidates") or [{}]
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    return parse_reply(text), response.get("usageMetadata") or {}


def check(reply, text: str, min_answer_words: int):
    """(outcome, pair) for what the model returned. `pair` is (question, answer, start)
    when the outcome is `kept`, else None."""
    if reply is None:
        return NO_QUESTION, None
    if not isinstance(reply, dict):
        return BAD_RESPONSE, None
    question, answer = reply.get("question"), reply.get("answer_span")
    if not isinstance(question, str) or not isinstance(answer, str) or not question.strip() or not answer.strip():
        return BAD_RESPONSE, None
    start = text.find(answer)
    if start == -1:
        return SPAN_NOT_FOUND, None
    if len(answer.split()) < min_answer_words:
        return TOO_SHORT, None
    return KEPT, (question.strip(), answer, start)


class InstructionGenStage(Stage):
    name = "instruction-gen"

    def __init__(self, allow_any: dict[str, float], forbid_any: dict[str, float], keep_sources: list[str],
                 mt_exempt_domains: list[str],
                 min_words: int, chunk_words: int, model: str, temperature: float,
                 min_answer_words: int, batch: bool, workers: int, retries: int, limit: int | None, seed: int,
                 input_manifest: str | None = None):
        self._allow_any = allow_any
        self._forbid_any = forbid_any
        self._keep_sources = set(keep_sources)
        self._mt_exempt_domains = set(mt_exempt_domains)
        self._min_words = min_words
        self._chunk_words = chunk_words
        self._model = model
        self._temperature = temperature
        self._min_answer_words = min_answer_words
        self._batch = batch
        self._workers = workers
        self._retries = retries
        self._limit = limit
        self._seed = seed
        self._input_manifest = input_manifest
        self._client = None

    def run(self, input_dir: Path | None, output_dir: Path) -> None:
        # A sample run writes to data/instruction-gen/limit-<limit>-... in place of the run's
        # folder, so it never mixes with the full run. The thinking level is in the name too,
        # so runs with different levels can be compared.
        name = f"limit-{self._limit}" if self._limit else output_dir.name
        output_dir = output_dir.parent / f"{name}-thinking-{THINKING_LEVEL.lower()}"
        output_dir.mkdir(parents=True, exist_ok=True)
        responses_path = output_dir / "responses.jsonl"
        answered = self._answered(responses_path)

        manifest = self.manifest(input_dir)
        print(f"Reading {manifest}")
        todo = self.select(self.wanted(manifest))
        print(f"{len(todo):,} chunks chosen, {len(answered):,} of them answered by an earlier run "
              f"(model={self._model})")
        todo = [chunk for chunk in todo if chunk["key"] not in answered]

        if self._batch and (todo or (output_dir / "batch_job.json").exists()):
            self._client = self._make_client()
            self._generate_batch(todo, output_dir)
        elif todo:
            self._client = self._make_client()
            self._generate_all(todo, responses_path, output_dir / "manifest.jsonl")

        stats = self.stats(responses_path)
        with open(output_dir / "stats.json", "w", encoding="utf-8") as sf:
            json.dump(stats, sf, ensure_ascii=False, indent=2)
        print(f"\nDone — written to {output_dir}/manifest.jsonl, responses.jsonl and stats.json")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    def manifest(self, input_dir: Path | None) -> Path:
        """`input_manifest` when it is set, else the classify manifest of this run."""
        if self._input_manifest:
            return Path(self._input_manifest)
        return input_dir / "manifest.jsonl"

    def wanted(self, manifest_path: Path):
        """Yield every chunk of the chosen entries that has at least `min_words` words.

        Reads the manifest a line at a time: the full one is over 1.5 GB.
        """
        with open(manifest_path, encoding="utf-8") as mf:
            for line in mf:
                record = json.loads(line)
                if not self.keeps(record):
                    continue
                for n, text in enumerate(chunks(record["text"], self._chunk_words)):
                    if len(text.split()) < self._min_words:
                        continue
                    yield {
                        "key": f"{record['id']}#{n}",
                        "source": record["source"],
                        "id": record["id"],
                        "url": record["url"],
                        "domain": record["domain"],
                        "chunk": n,
                        "text": text,
                        "register_labels": record["register_labels"],
                        "register_scores": record["register_scores"],
                    }

    def keeps(self, record: dict) -> bool:
        if record["source"] in self._keep_sources:
            return True
        scores = record["register_scores"]
        return self.above(scores, self._allow_any) and not self.above(scores, self.forbid_any(record))

    def forbid_any(self, record: dict) -> dict[str, float]:
        """The forbidden labels for a record: all of `forbid_any`, but MT for an exempt domain."""
        if record["domain"] in self._mt_exempt_domains:
            return {label: threshold for label, threshold in self._forbid_any.items() if label != "MT"}
        return self._forbid_any

    @staticmethod
    def above(scores: dict[str, float], thresholds: dict[str, float]) -> bool:
        """Whether any label scores above its threshold, like classify's `threshold`."""
        return any(scores.get(label, 0) > threshold for label, threshold in thresholds.items())

    def select(self, chunks) -> list[dict]:
        """The chunks in a shuffled order fixed by `seed`, cut to `limit`.

        The same seed picks the same chunks every time, so sample runs with different prompts
        can be compared pair by pair.
        """
        chunks = list(chunks)
        random.Random(self._seed).shuffle(chunks)
        return chunks[:self._limit] if self._limit else chunks

    def _answered(self, responses_path: Path) -> set[str]:
        """The keys of the chunks an earlier run already answered."""
        if not responses_path.exists():
            return set()
        with open(responses_path, encoding="utf-8") as rf:
            return {json.loads(line)["key"] for line in rf}

    def _generate_all(self, todo: list[dict], responses_path: Path, manifest_path: Path) -> None:
        outcomes = Counter()
        with (ThreadPoolExecutor(self._workers) as pool,
              open(responses_path, "a", encoding="utf-8") as rf,
              open(manifest_path, "a", encoding="utf-8") as mf):
            futures = {pool.submit(self.ask, chunk["text"]): chunk for chunk in todo}
            try:
                for count, future in enumerate(as_completed(futures), 1):
                    chunk, reply = futures[future], future.result()
                    outcomes[self.save(chunk, reply, rf, mf)] += 1
                    print(f"  [{count:,}/{len(todo):,}] " + ", ".join(f"{k} {n:,}" for k, n in outcomes.items()),
                          end="\r")
            except BaseException:
                # Do not wait for the queued chunks: a fatal API error or Ctrl-C ends the run now
                pool.shutdown(cancel_futures=True)
                raise

    def _generate_batch(self, todo: list[dict], output_dir: Path) -> None:
        """Send the chunks as one Batch API job, wait for it and save its results.

        A job submitted by an earlier run, named in `batch_job.json`, is waited for instead
        of submitting a new one. Its results are matched to the chunks by key; chunks it
        failed, or did not include, are sent by the next run.
        """
        job_path = output_dir / "batch_job.json"
        requests_path = output_dir / "batch_requests.jsonl"
        if job_path.exists():
            name = json.loads(job_path.read_text(encoding="utf-8"))["name"]
            print(f"Waiting for {name}, submitted by an earlier run")
        else:
            name = self._submit(todo, requests_path, display_name=f"instruction-gen-{output_dir.name}")
            job_path.write_text(json.dumps({"name": name, "chunks": len(todo)}), encoding="utf-8")
            print(f"Submitted {name} with {len(todo):,} chunks. If this run is stopped, running it "
                  f"again waits for the same job.")

        job = self._wait(name)
        if job.state.name != "JOB_STATE_SUCCEEDED":
            job_path.unlink()
            sys.exit(f"{name} ended {job.state.name} ({job.error}). Run again to submit the chunks again.")

        self._save_results(job, todo, output_dir)
        # Only now: a run stopped while saving downloads the same results again
        job_path.unlink()
        requests_path.unlink(missing_ok=True)

    def _submit(self, todo: list[dict], requests_path: Path, display_name: str) -> str:
        """Write the chunks' requests to `requests_path`, upload it and create a batch job
        from it. Returns the job's name."""
        with open(requests_path, "w", encoding="utf-8") as f:
            for chunk in todo:
                f.write(json.dumps(self.batch_request(chunk), ensure_ascii=False) + "\n")
        uploaded = self._client.files.upload(file=str(requests_path),
                                             config={"display_name": display_name, "mime_type": "jsonl"})
        job = self._client.batches.create(model=self._model, src=uploaded.name,
                                          config={"display_name": display_name})
        return job.name

    def _save_results(self, job, todo: list[dict], output_dir: Path) -> None:
        """Download a finished job's results and save each chunk's reply, as single calls do,
        then print the outcomes and the tokens the job used."""
        results = self._client.files.download(file=job.dest.file_name).decode("utf-8")
        by_key = {chunk["key"]: chunk for chunk in todo}
        outcomes, tokens = Counter(), Counter()
        with (open(output_dir / "responses.jsonl", "a", encoding="utf-8") as rf,
              open(output_dir / "manifest.jsonl", "a", encoding="utf-8") as mf):
            for line in results.splitlines():
                if not line.strip():
                    continue
                result = json.loads(line)
                chunk = by_key.get(result.get("key"))
                if chunk is None:  # saved by an earlier run that stopped while saving
                    outcomes["already_saved"] += 1
                    continue
                reply, usage = batch_reply(result)
                tokens.update({"input": usage.get("promptTokenCount", 0),
                               "output": usage.get("candidatesTokenCount", 0),
                               "thinking": usage.get("thoughtsTokenCount", 0)})
                outcomes[self.save(chunk, reply, rf, mf)] += 1
        print(f"  {job.name}: " + ", ".join(f"{k} {n:,}" for k, n in outcomes.items()))
        print(f"  tokens: input {tokens['input']:,}, output {tokens['output']:,}, thinking {tokens['thinking']:,}")

    def _wait(self, name: str):
        """The batch job once it has ended, checking it every BATCH_POLL_SECONDS."""
        started = time.monotonic()
        while True:
            job = self._client.batches.get(name=name)
            if job.state.name in BATCH_DONE_STATES:
                print()
                return job
            minutes = (time.monotonic() - started) / 60
            print(f"  {job.state.name} ({minutes:.0f} min)", end="\r", flush=True)
            time.sleep(BATCH_POLL_SECONDS)

    def batch_request(self, chunk: dict) -> dict:
        """One line of a Batch API input file: the same request ask() sends for the chunk."""
        return {
            "key": chunk["key"],
            "request": {
                "contents": [{"role": "user", "parts": [{"text": prompt(chunk["text"])}]}],
                "generation_config": {
                    "response_mime_type": "application/json",
                    "temperature": self._temperature,
                    "thinking_config": {"thinking_level": THINKING_LEVEL},
                },
            },
        }

    def save(self, chunk: dict, reply, rf, mf) -> str:
        """Check the model's reply for a chunk and append it to `responses.jsonl`, and the pair
        to `manifest.jsonl` when it is kept. Returns the outcome. A FAILED reply is not
        written, so a later run asks again."""
        if reply is FAILED:
            return "api_error"
        outcome, pair = check(reply, chunk["text"], self._min_answer_words)
        rf.write(json.dumps({"key": chunk["key"], "outcome": outcome,
                             "register_labels": chunk["register_labels"], "reply": reply},
                            ensure_ascii=False) + "\n")
        rf.flush()
        if pair:
            mf.write(json.dumps(self.record(chunk, *pair), ensure_ascii=False) + "\n")
            mf.flush()
        return outcome

    def record(self, chunk: dict, question: str, answer: str, start: int) -> dict:
        return {
            "source": chunk["source"],
            "id": chunk["id"],
            "url": chunk["url"],
            "domain": chunk["domain"],
            "chunk": chunk["chunk"],
            "context": chunk["text"],
            "question": question,
            "answer": answer,
            "answer_start": start,
            "model": self._model,
            "register_labels": chunk["register_labels"],
            "register_scores": chunk["register_scores"],
        }

    def ask(self, text: str):
        """The parsed JSON the model returned for a chunk, or FAILED when every retry failed."""
        from google.genai import errors, types

        for attempt in range(self._retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt(text),
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=self._temperature,
                        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
                    ),
                )
            except errors.ClientError as e:
                # 4xx errors (bad model, bad key, ...) won't fix themselves; only retry rate limits
                if e.code != 429:
                    raise
                error = "rate limited"
            except Exception as e:
                error = str(e)
            else:
                return parse_reply(response.text)
            wait = 2 ** attempt * 5
            print(f"\n  API error ({error}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
        return FAILED

    @staticmethod
    def _make_client():
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit("GEMINI_API_KEY is not set (export it or put it in .env)")
        return genai.Client(api_key=api_key)

    @staticmethod
    def stats(responses_path: Path) -> dict:
        """Outcomes over every answered chunk, in total and per label set, and the average
        question and answer word counts of the kept pairs, from `responses.jsonl` (so a
        resumed run counts the earlier runs' chunks too)."""
        total, per_label_set = Counter(), defaultdict(Counter)
        question_words = answer_words = 0
        if responses_path.exists():
            with open(responses_path, encoding="utf-8") as rf:
                for line in rf:
                    response = json.loads(line)
                    total[response["outcome"]] += 1
                    per_label_set[" ".join(response["register_labels"]) or "(none)"][response["outcome"]] += 1
                    if response["outcome"] == KEPT:
                        question_words += len(response["reply"]["question"].split())
                        answer_words += len(response["reply"]["answer_span"].split())
        pairs = total[KEPT]
        return {
            "chunks": sum(total.values()),
            "outcomes": dict(total.most_common()),
            "question_words_avg": round(question_words / pairs, 1) if pairs else 0,
            "answer_words_avg": round(answer_words / pairs, 1) if pairs else 0,
            "label_sets": {labels: dict(outcomes.most_common()) for labels, outcomes
                           in sorted(per_label_set.items(), key=lambda kv: -sum(kv[1].values()))},
        }


if __name__ == "__main__":
    run = run_name()
    stage = InstructionGenStage(**stage_config("instruction-gen"))
    stage.run(Path("data/classify") / run, Path("data/instruction-gen") / run)
