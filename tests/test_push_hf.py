from scripts.push_hf import count_languages, drop_contacts, languages


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


class TestDropContacts:
    def test_drops_a_pair_with_a_phone_number_after_a_phone_word(self):
        pairs = [pair("pirs", "bersiv"), {**pair("pirs", "bersiv"), "context": "Telefon : 0532 409 88 57"}]
        assert drop_contacts(pairs) == [pairs[0]]

    def test_drops_a_pair_with_an_international_number_in_its_answer(self):
        assert drop_contacts([pair("pirs", "li ser 0049 17631714013 yan jî")]) == []

    def test_drops_a_pair_with_an_email_address(self):
        assert drop_contacts([pair("pirs", "mirazroni@hotmail.com")]) == []

    def test_keeps_finewebs_email_placeholders(self):
        pairs = [pair("pirs", "firstname.lastname@example.org û email@example.com.")]
        assert drop_contacts(pairs) == pairs

    def test_keeps_years_page_ranges_and_isbns(self):
        pairs = [pair("pirs", "Xanî (1650/51-1707), rûpel (147-148-149), ISBN 0078664276")]
        assert drop_contacts(pairs) == pairs
