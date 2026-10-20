"""Web register classification with TurkuNLP/web-register-classification-multilingual."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "TurkuNLP/web-register-classification-multilingual"
TOKENIZER_ID = "xlm-roberta-large"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def labels_from_scores(scores, threshold):
    """Sorted labels scoring above the threshold."""
    return sorted(label for label, p in scores.items() if p > threshold)


def average_by_text(probs, text_index_of_window, n_texts):
    """Turn per-window scores into per-text scores.

    `probs` has one row per window. The result has one row per text: the average of
    the rows of that text's windows. `text_index_of_window[i]` is the text window `i`
    came from, e.g. [0, 0, 0, 1, 2, 2] for three texts; every text has at least one
    window.
    """
    return torch.stack([probs[text_index_of_window == i].mean(dim=0) for i in range(n_texts)])


class RegisterClassifier:
    def __init__(self, threshold=0.5, max_length=512, stride=0, batch_size=8):
        self.threshold = threshold
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.device = pick_device()
        print(f"Loading {MODEL_ID} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, dtype=torch.float32).to(self.device).eval()
        self.id2label = self.model.config.id2label

    def classify(self, texts):
        """Return one (sorted labels, {label: score}) tuple per text.

        The model reads at most `max_length` tokens, so each text is cut into windows of
        that size (`stride` tokens shared by neighbouring windows), and a label's score is
        its average over the text's windows. A text that fits in one window scores as it
        would without windowing.

        The windows of all the texts go through the model together, `batch_size` at a
        time, so a batch is full even when most texts are a single window.
        """
        windows = self.tokenizer(texts, truncation=True, max_length=self.max_length,
                                 stride=self.stride, return_overflowing_tokens=True)
        text_index_of_window = torch.tensor(windows.pop("overflow_to_sample_mapping"))
        batch_probs = []
        for start in range(0, len(text_index_of_window), self.batch_size):
            batch = self.tokenizer.pad(
                {key: values[start:start + self.batch_size] for key, values in windows.items()},
                return_tensors="pt").to(self.device)
            with torch.no_grad():
                batch_probs.append(torch.sigmoid(self.model(**batch).logits).cpu())

        averages = average_by_text(torch.cat(batch_probs), text_index_of_window, len(texts))

        results = []
        for average in averages:
            scores = {self.id2label[i]: round(p.item(), 4) for i, p in enumerate(average)}
            results.append((labels_from_scores(scores, self.threshold), scores))
        return results
