#!/usr/bin/env python3
"""Push instruction-gen's pairs to the Hugging Face Hub, after counting the questions and
answers GlotLID does not identify as Kurmanji.

Reads an instruction-gen manifest, asks GlotLID for the language of every question and
answer, and prints how many of each are not kmr_Latn and what GlotLID called them
instead. Every pair goes up, with its question as `instruction`, its answer as `output`,
and its context (the chunk the answer was copied from), id, domain, source and url, as
the `kmr_latn` subset of a private dataset: on the first full run GlotLID called 764 of
33,073 questions kiu_Latn, and the ones read were plain Kurmanji, so its count is printed
but nothing is dropped on it.

A pair is dropped when its answer or context has a phone number or an email address,
other than the placeholders FineWeb puts in place of emails. On the first full run that
was 82 pairs: contact lines of publishers and offices, mobiles in notices, and the
contact pages at the end of theses, some with the author's birth date and place.

Run it with --dry-run first to see the counts without pushing.

The Hugging Face token is HF_TOKEN in .env, so it never lands in the shell's history. It
is read from the file only: a HF_TOKEN already in the shell (a read token, say) would
otherwise win and the push fail with 403.

Usage:
    python -m scripts.push_hf data/instruction-gen/<folder>/manifest.jsonl --dataset user/name --dry-run
    python -m scripts.push_hf data/instruction-gen/<folder>/manifest.jsonl --dataset user/name
"""

import argparse
import json
import re
import sys
from collections import Counter

# GlotLID v3, a fastText model that tells apart Kurmanji (kmr_Latn), Sorani and Turkish
GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_FILE = "model.bin"

# The language questions and answers are counted against, and the subset the pairs go up as
LANGUAGE = "kmr_Latn"
SUBSET = "kmr_latn"

# The manifest fields that go up to the Hub, and their names there. `question` and `answer`
# take the names instruction-tuning datasets use. `id` leads back to the source document,
# and `context` makes the pairs usable for reading comprehension too.
COLUMNS = {"question": "instruction", "answer": "output", "context": "context", "id": "id",
           "domain": "domain", "source": "source", "url": "url"}

# The dataset is created private, to be looked over on the Hub before it is made public
PRIVATE = True

# Words that come before a phone number, in Kurmanji, Turkish and English
PHONE_WORDS = r"(?:tel|telefon|telefûn|telîfon|tlf|phone|mobile|mobîl|whatsapp|faks|fax|gsm|hejmara telefon\w*|pêwendî|têkilî)"
# A phone number after one of PHONE_WORDS; one with a country code (+ or 00); and a
# Turkish or Iraqi number written with its leading 0, which often stands alone
PHONE_PATTERNS = [
    re.compile(PHONE_WORDS + r"\W{0,6}((?:\+|00)?\(?\d[\d ()-]{6,20}\d)", re.IGNORECASE),
    re.compile(r"(?<![\w./-])((?:\+|00)\d{1,3}[ (-]*\d[\d ()-]{6,18}\d)(?![\w/.-])"),
    re.compile(r"(?<![\w./-])(0[1-9]\d{2}[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}|07\d{9}|05\d{9})(?![\w/.-])"),
]
# A phone number has 7 to 15 digits (E.164 allows 15); fewer is a year or a page range
PHONE_DIGITS = range(7, 16)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# FineWeb's anonymiser puts these in place of the emails it finds
EMAIL_PLACEHOLDER_DOMAINS = ("example.org", "example.com")
# The fields searched for phone numbers and emails; the answer is a copy of context spans,
# but both are searched so the check does not depend on that
CONTACT_FIELDS = ("answer", "context")

# The other languages printed for questions and answers that are not LANGUAGE
TOP_OTHER_LANGUAGES = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="an instruction-gen manifest.jsonl")
    parser.add_argument("--dataset", required=True, help="the Hub dataset to push to, e.g. user/kurdish-instructions")
    parser.add_argument("--dry-run", action="store_true", help="print the counts and push nothing")
    return parser.parse_args()


def hf_token() -> str:
    from dotenv import dotenv_values

    token = dotenv_values(".env").get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not in .env")
    return token


def load_glotlid():
    import fasttext
    from huggingface_hub import hf_hub_download

    return fasttext.load_model(hf_hub_download(GLOTLID_REPO, GLOTLID_FILE))


def languages(model, texts: list[str]) -> list[str]:
    """GlotLID's top language for each text, e.g. kmr_Latn. fastText reads one line at a
    time, so line breaks become spaces."""
    labels, _ = model.predict([text.replace("\n", " ") for text in texts])
    return [label[0].removeprefix("__label__") for label in labels]


def count_languages(pairs: list[dict], model) -> dict:
    """How many questions and answers are not LANGUAGE: per field, both together, either
    one, and the languages found instead."""
    question_languages = languages(model, [pair["question"] for pair in pairs])
    answer_languages = languages(model, [pair["answer"] for pair in pairs])
    counts = {"pairs": len(pairs), "question_not": 0, "answer_not": 0, "both_not": 0, "either_not": 0,
              "question_languages": Counter(), "answer_languages": Counter()}
    for question_language, answer_language in zip(question_languages, answer_languages):
        question_ok, answer_ok = question_language == LANGUAGE, answer_language == LANGUAGE
        if not question_ok:
            counts["question_not"] += 1
            counts["question_languages"][question_language] += 1
        if not answer_ok:
            counts["answer_not"] += 1
            counts["answer_languages"][answer_language] += 1
        counts["both_not"] += not question_ok and not answer_ok
        counts["either_not"] += not question_ok or not answer_ok
    return counts


def has_phone(text: str) -> bool:
    for pattern in PHONE_PATTERNS:
        for match in pattern.finditer(text):
            # An ISBN without hyphens looks like a phone number written with its leading 0
            if re.search(r"ISBN\W*$", text[max(0, match.start() - 8):match.start()]):
                continue
            if len(re.sub(r"\D", "", match.group(1))) in PHONE_DIGITS:
                return True
    return False


def has_email(text: str) -> bool:
    return any(not email.rstrip(".").endswith(EMAIL_PLACEHOLDER_DOMAINS) for email in EMAIL.findall(text))


def drop_contacts(pairs: list[dict]) -> list[dict]:
    """The pairs with no phone number or email address in CONTACT_FIELDS."""
    return [pair for pair in pairs
            if not any(has_phone(pair[field]) or has_email(pair[field]) for field in CONTACT_FIELDS)]


def report(counts: dict) -> str:
    lines = [f"pairs                     {counts['pairs']:,}",
             f"question not {LANGUAGE}     {counts['question_not']:,}",
             f"answer not {LANGUAGE}       {counts['answer_not']:,}",
             f"both not {LANGUAGE}         {counts['both_not']:,}",
             f"either not {LANGUAGE}       {counts['either_not']:,}"]
    for field in ("question", "answer"):
        found = counts[f"{field}_languages"].most_common(TOP_OTHER_LANGUAGES)
        if found:
            lines.append(f"{field} languages other than {LANGUAGE}: "
                         + ", ".join(f"{language} {n:,}" for language, n in found))
    return "\n".join(lines)


def push(pairs: list[dict], dataset: str, token: str) -> None:
    from datasets import Dataset

    rows = [{name: pair[field] for field, name in COLUMNS.items()} for pair in pairs]
    Dataset.from_list(rows).push_to_hub(dataset, config_name=SUBSET, private=PRIVATE, token=token)


def main():
    args = parse_args()
    token = None if args.dry_run else hf_token()
    with open(args.manifest, encoding="utf-8") as mf:
        pairs = [json.loads(line) for line in mf]
    print(f"Read {len(pairs):,} pairs from {args.manifest}")

    print(report(count_languages(pairs, load_glotlid())))

    kept = drop_contacts(pairs)
    print(f"dropped for a phone number or email address  {len(pairs) - len(kept):,}")
    pairs = kept

    if args.dry_run:
        print("\nDry run: nothing pushed")
        return
    push(pairs, args.dataset, token)
    print(f"\nPushed {len(pairs):,} pairs to {args.dataset} as the {SUBSET} subset (private)")


if __name__ == "__main__":
    main()
