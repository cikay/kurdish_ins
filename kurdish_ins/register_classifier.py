"""Web register classification with TurkuNLP/web-register-classification-multilingual."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "TurkuNLP/web-register-classification-multilingual"
TOKENIZER_ID = "xlm-roberta-large"

# Sub-register -> main register (a predicted sub-register implies its parent)
PARENT = {
    "it": "SP",
    "ne": "NA", "sr": "NA", "nb": "NA",
    "re": "HI",
    "en": "IN", "ra": "IN", "dtp": "IN", "fi": "IN", "lt": "IN",
    "rv": "OP", "ob": "OP", "rs": "OP", "av": "OP",
    "ds": "IP", "ed": "IP",
}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def labels_from_scores(scores, threshold):
    """Sorted labels scoring above the threshold, plus the parent of every sub-register."""
    labels = {label for label, p in scores.items() if p > threshold}
    labels |= {PARENT[label] for label in labels if label in PARENT}
    return sorted(labels)


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

    def classify(self, text):
        """Return (sorted labels, {label: score}) for one text.

        The model reads at most `max_length` tokens, so the text is cut into windows of
        that size (`stride` tokens shared by neighbouring windows), and a label's score is
        its average over the windows. A text that fits in one window scores as it would
        without windowing.
        """
        windows = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 stride=self.stride, return_overflowing_tokens=True)
        windows.pop("overflow_to_sample_mapping")
        probs = []
        for start in range(0, len(windows["input_ids"]), self.batch_size):
            batch = self.tokenizer.pad(
                {key: values[start:start + self.batch_size] for key, values in windows.items()},
                return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs.append(torch.sigmoid(self.model(**batch).logits).cpu())
        average = torch.cat(probs).mean(dim=0)
        scores = {self.id2label[i]: round(p.item(), 4) for i, p in enumerate(average)}
        return labels_from_scores(scores, self.threshold), scores
