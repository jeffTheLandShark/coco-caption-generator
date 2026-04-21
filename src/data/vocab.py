import json
import re
from collections import Counter


class Vocabulary:

    PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
    SPECIAL = [PAD, SOS, EOS, UNK]

    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}

    def build(self, captions: list[str]) -> None:
        counter: Counter = Counter()
        for cap in captions:
            counter.update(self._tokenize(cap))
        for token in self.SPECIAL:
            self._add(token)
        for word, freq in counter.items():
            if freq >= self.min_freq:
                self._add(word)
        print(f"[Vocab] size={len(self)} (min_freq={self.min_freq})")

    def encode(self, caption: str, add_special: bool = True) -> list[int]:
        tokens = self._tokenize(caption)
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]
        if add_special:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        skip = set(self.SPECIAL) if skip_special else set()
        words = [
            self.idx2word.get(i, self.UNK)
            for i in ids
            if self.idx2word.get(i, self.UNK) not in skip
        ]
        return " ".join(words)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx, "min_freq": self.min_freq}, f)
        print(f"[Vocab] saved → {path}")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path) as f:
            data = json.load(f)
        vocab = cls(min_freq=data["min_freq"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab

    def _add(self, word: str) -> None:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()

    def __len__(self) -> int:
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD]

    @property
    def sos_idx(self) -> int:
        return self.word2idx[self.SOS]

    @property
    def eos_idx(self) -> int:
        return self.word2idx[self.EOS]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[self.UNK]