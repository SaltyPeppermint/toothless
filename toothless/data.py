import json
from pathlib import Path
from dataclasses import dataclass
from functools import cmp_to_key

import torch
from torch import Tensor
from torch.utils.data import Dataset

import diskcache as dc
from tqdm.auto import tqdm


from eggshell import rise, TreeData  # type: ignore

from .args import DataArguments
from .vocab import BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, PAD_TOKEN, UNK_TOKEN, SimpleVocab


# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.normalizers import BertNormalizer
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
# from tokenizers.pre_tokenizers import Split
# from tokenizers.normalizers import Strip
# from tokenizers.normalizers import Sequence as NormalizerSequence
# import matplotlib.pyplot as plt


@dataclass
class Triple:
    l_ids: Tensor
    l_str: str
    tgt_ids: Tensor
    tgt_str: str
    r_ids: Tensor
    r_str: str


class TripleDataSet(Dataset[Triple]):
    def __init__(self, conf: DataArguments):
        """
        :param k represents the max relative distance
        """
        self.json_root = Path(conf.data_path)
        self.force_reload = conf.force_reload
        torch.manual_seed(conf.rng_seed)

        self.cache = Path(conf.cache_dir) / Path(*self.json_root.parts[-2:])
        self.cache.mkdir(parents=True, exist_ok=True)
        self.sample_cache = dc.Cache(self.cache / "sample_cache")

        self.index_cache = self.cache / "index_cache.pickle"
        self.vocab_path = self.cache / "vocab.json"

        self.vocab = self._build_vocab()
        self.len = self._iterate_samples()

    def _build_vocab(self) -> SimpleVocab:
        normal_tokens = rise.operators() + ["[constant]", "[variable]"]
        vocab = SimpleVocab(PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN, normal_tokens)
        vocab.save(self.vocab_path)
        return vocab

    def _iterate_samples(self) -> int:
        print(self.json_root)

        json_files = list(self.json_root.glob("*.json"))
        json_files.sort(key=cmp_to_key(path_cmp))

        with open(json_files[0], mode="r", encoding="utf-8") as f:
            file = json.load(f)
            start = file["start_expr"]

        i = 0

        for batch_file in tqdm(json_files, desc="Enumerating tripples"):
            with open(batch_file, mode="r", encoding="utf-8") as f:
                file = json.load(f)
                guide = file["midpoint"]["midpoint"]["expression"]
                for target in file["midpoint"]["goals"]:
                    self.sample_cache[i] = {"start": start, "guide": guide, "target": target}

        return i

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> Triple:
        s = self.sample_cache[idx]

        assert type(s) is dict[str, str]

        l_tree = rise.RecExpr(s["start"]).to_data()
        tgt_tree = rise.RecExpr(s["guide"]).to_data()
        r_tree = rise.RecExpr(s["target"]).to_data()

        l_ids = self._pyrec_to_tensor(l_tree)
        tgt_ids = self._pyrec_to_tensor(tgt_tree)
        r_ids = self._pyrec_to_tensor(r_tree)

        return Triple(l_ids, s["left"], tgt_ids, s["guide"], r_ids, s["target"])

    def _pyrec_to_tensor(self, tree_data: TreeData) -> Tensor:
        return torch.tensor(
            [self.vocab.bos_token_id]
            + [self.vocab.token2id(node.name) for node in tree_data.nodes()]
            + [self.vocab.eos_token_id],
            dtype=torch.long,
        )


def path_cmp(a: Path, b: Path) -> int:
    midpoint_id_a, batch_id_a = a.stem.split("-", 1)
    midpoint_id_b, batch_id_b = b.stem.split("-", 1)

    if int(midpoint_id_a) == int(midpoint_id_b):
        return int(batch_id_a) - int(batch_id_b)
    return int(midpoint_id_a) - int(midpoint_id_b)


def split_off_special(partial_tok: list[str], vocab: SimpleVocab) -> list[str]:
    partial_tok = partial_tok[1:]
    for i, j in enumerate(partial_tok):
        if j in vocab.special_tokens:
            return partial_tok[:i]
    return partial_tok
