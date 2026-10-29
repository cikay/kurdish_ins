from scripts.clean import clean

REPLACEMENTS = {"þ": "ş", "ÅŸ": "ş"}
DROP_PATTERNS = {"garbled_g": "[Ģġ]", "dotless_i_word": r"\bjı\b"}


def pair(instruction, output, context=None):
    return {"instruction": instruction, "output": output, "context": context or output, "id": "1",
            "domain": "d", "source": "s", "url": "u"}


class TestClean:
    def test_replaces_garbled_letters_in_every_text_field_and_counts_them(self):
        kept, dropped, replaced = clean([pair("Piþtî çi bû?", "PiÅŸtî şer bû.", "Berê. PiÅŸtî şer bû.")],
                                        REPLACEMENTS, DROP_PATTERNS)
        assert kept == [pair("Piştî çi bû?", "Piştî şer bû.", "Berê. Piştî şer bû.")]
        assert dropped == {}
        assert replaced == {"þ": 1, "ÅŸ": 2}

    def test_keeps_the_answer_a_verbatim_copy_of_its_context(self):
        kept, _, _ = clean([pair("Çi bû?", "baþ bû", "Ew baþ bû.")], REPLACEMENTS, DROP_PATTERNS)
        assert kept[0]["output"] in kept[0]["context"]

    def test_drops_a_pair_whose_question_or_answer_has_garbling_with_no_single_correct_form(self):
        pairs = [pair("PiĢtî çi bû?", "Bersiv."), pair("Çi bû?", "Ew jı malê hat."), pair("Çi bû?", "Bersiv.")]
        kept, dropped, _ = clean(pairs, REPLACEMENTS, DROP_PATTERNS)
        assert kept == [pairs[2]]
        assert dropped == {"garbled_g": 1, "dotless_i_word": 1}

    def test_does_not_drop_a_pair_for_garbling_in_its_context_only(self):
        pairs = [pair("Çi bû?", "Bersiv.", "Bersiv. PiĢtî")]
        kept, dropped, _ = clean(pairs, REPLACEMENTS, DROP_PATTERNS)
        assert kept == pairs and dropped == {}

    def test_counts_replaced_letters_of_kept_pairs_only(self):
        _, _, replaced = clean([pair("Piþtî ġêx?", "Bersiv.")], REPLACEMENTS, DROP_PATTERNS)
        assert replaced == {}

    def test_puts_text_in_nfc_so_a_letter_with_a_combining_mark_matches(self):
        kept, _, _ = clean([pair("Çi bû?", "zimanȇ")], {"ȇ": "ê"}, DROP_PATTERNS)
        assert kept[0]["output"] == "zimanê"
