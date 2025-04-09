import json
from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
# SEP_TOKEN = "<sep>"
MASK_TOKEN = "<mask>"


class SimpleVocab:
    def __init__(
        self, pad_token: str, unk_token: str, mask_token: str, bos_token: str, eos_token: str, tokens: list[str]
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.vocab = {}
        for id, token in enumerate([pad_token, unk_token, mask_token, bos_token, eos_token] + tokens):
            self.vocab[token] = id

        self.rev_vocab = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.vocab)

    def token2id(self, token: str) -> int:
        if token in self.vocab.keys():
            return self.vocab[token]
        else:
            return self.vocab[self.unk_token]

    def id2token(self, id: int) -> str:
        return self.rev_vocab[id]

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def mask_token_id(self) -> int:
        return self.vocab[self.mask_token]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.unk_token_id]

    @property
    def bos_token_id(self) -> int:
        return self.vocab[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.eos_token]

    def save(self, path: Path):
        d = {}
        d["pad_token"] = self.pad_token
        d["unk_token"] = self.unk_token
        d["mask_token"] = self.mask_token
        d["bos_token"] = self.bos_token
        d["eos_token"] = self.eos_token
        d["vocab"] = self.vocab
        with open(path, mode="w", encoding="utf-8") as f:
            json.dump(d, f)

    @staticmethod
    def load(path: Path):
        with open(path, mode="r", encoding="utf-8") as f:
            d = json.load(f)

        v = list(d["vocab"].keys())[5:]
        return SimpleVocab(d["pad_token"], d["unk_token"], d["mask_token"], d["bos_token"], d["eos_token"], v)
