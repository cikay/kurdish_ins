import pytest
import yaml

from kurdish_ins.classify import ClassifyStage
from kurdish_ins.register_classifier import labels_from_scores
from kurdish_ins.stage import run_name


def stage():
    return ClassifyStage(dataset="d", subset="s", input_file=None, threshold=0.5,
                         batch_size=2, max_length=512, stride=0)


class FakeClassifier:
    """Labels every text `IN`/`en`, and records the texts it was given."""

    def __init__(self):
        self.texts = []

    def classify(self, text):
        self.texts.append(text)
        return ["IN", "en"], {"IN": 0.9, "en": 0.8}


def entry(i):
    return {"id": f"id{i}", "url": f"http://site{i}.org/page", "text": f"text {i}", "date": "x"}


class TestLabelsFromScores:
    def test_sub_register_brings_its_parent(self):
        assert labels_from_scores({"en": 0.9, "IN": 0.2}, 0.5) == ["IN", "en"]

    def test_nothing_above_threshold_is_no_label(self):
        assert labels_from_scores({"IN": 0.5, "NA": 0.1}, 0.5) == []


class TestClassify:
    def test_every_entry_is_classified_in_order(self):
        classifier = FakeClassifier()
        records = list(stage().classify([entry(i) for i in range(3)], classifier))
        assert len(records) == 3
        assert classifier.texts == ["text 0", "text 1", "text 2"]

    def test_record_keeps_text_and_adds_registers(self):
        [record] = stage().classify([entry(1)], FakeClassifier())
        assert record == {
            "id": "id1",
            "url": "http://site1.org/page",
            "domain": "site1.org",
            "text": "text 1",
            "register_labels": ["IN", "en"],
            "register_scores": {"IN": 0.9, "en": 0.8},
        }


def config(tmp_path, input_file):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump({"classify": {"description": "x", "input_file": input_file}}))
    return path


class TestRunName:
    def test_no_input_file_is_the_full_run(self, tmp_path):
        assert run_name(config(tmp_path, None)) == "full"

    def test_input_file_names_the_run(self, tmp_path):
        assert run_name(config(tmp_path, "data/sample.jsonl")) == "sample"

    def test_input_file_called_full_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            run_name(config(tmp_path, "data/full.jsonl"))
