from scripts.push_hf import count_languages, languages


class FakeGlotLID:
    """Calls a text Turkish when it has "tr", English when it has "en", else Kurmanji, and
    records what it was given."""

    def __init__(self):
        self.seen = []

    def predict(self, texts):
        self.seen += texts
        labels = [["__label__tur_Latn"] if "tr" in t.split() else ["__label__eng_Latn"] if "en" in t.split()
                  else ["__label__kmr_Latn"] for t in texts]
        return labels, [[1.0]] * len(texts)


def pair(question, answer):
    return {"question": question, "answer": answer, "context": answer, "domain": "d", "source": "s", "url": "u"}


class TestLanguages:
    def test_line_breaks_become_spaces(self):
        model = FakeGlotLID()
        languages(model, ["yek\ndu"])
        assert model.seen == ["yek du"]


class TestCountLanguages:
    def test_counts_questions_and_answers_not_in_kurmanji(self):
        pairs = [pair("pirs", "bersiv"), pair("tr", "bersiv"), pair("pirs", "en"), pair("tr", "en")]
        counts = count_languages(pairs, FakeGlotLID())
        assert (counts["pairs"], counts["question_not"], counts["answer_not"], counts["both_not"],
                counts["either_not"]) == (4, 2, 2, 1, 3)

    def test_counts_the_languages_found_instead(self):
        counts = count_languages([pair("tr", "en"), pair("tr", "bersiv")], FakeGlotLID())
        assert counts["question_languages"] == {"tur_Latn": 2}
        assert counts["answer_languages"] == {"eng_Latn": 1}

