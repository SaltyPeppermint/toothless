import json
from pathlib import Path
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

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

        self.range_to_file_cache = self.cache / "range_to_file.json"
        self.len, self.range_to_file = self._iterate_samples()

        self.vocab_path = self.cache / "vocab.json"
        self.vocab = self._build_vocab()

    def _build_vocab(self) -> SimpleVocab:
        normal_tokens = rise.operators() + ["[constant]", "[variable]"]
        vocab = SimpleVocab(PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN, normal_tokens)
        vocab.save(self.vocab_path)
        return vocab

    def _iterate_samples(self) -> tuple[int, dict[tuple[int, int], Path]]:
        if self.range_to_file_cache.is_file():
            with open(self.range_to_file_cache, mode="r", encoding="utf-8") as f:
                return json.load(f)

        json_files = list(self.json_root.glob("*.json"))
        json_files.sort()

        range_to_file = {}
        i = 0
        for batch_file in tqdm(json_files, desc="Enumerating tripples"):
            with open(batch_file, mode="r", encoding="utf-8") as f:
                file = json.load(f)
            j = i + len(file["midpoint"]["goals"])
            range_to_file[(i, j)] = batch_file
            i = j + 1

        with open(self.range_to_file_cache, mode="w", encoding="utf-8") as f:
            json.dump((i, range_to_file), f)

        return i, range_to_file

    def __len__(self):
        return self.len

    def load_str_triple(self, idx: int) -> tuple[str, str, str]:
        for (lower, upper), p in self.range_to_file.items():
            if lower <= idx < upper:
                with open(p, mode="r", encoding="utf-8") as f:
                    file = json.load(f)
                start = file["start_expr"]
                guide = file["midpoint"]["midpoint"]["expression"]
                target = file["midpoint"]["goals"][idx - lower]["expression"]
                return start, guide, target

        raise ValueError(f"{idx} Not found")

    def __getitem__(self, idx: int) -> Triple:
        left, middle, right = self.load_str_triple(idx)

        l_tree = rise.RecExpr(left).to_data()
        tgt_tree = rise.RecExpr(middle).to_data()
        r_tree = rise.RecExpr(right).to_data()

        l_ids = self._pyrec_to_tensor(l_tree)
        tgt_ids = self._pyrec_to_tensor(tgt_tree)
        r_ids = self._pyrec_to_tensor(r_tree)

        return Triple(l_ids, left, tgt_ids, middle, r_ids, right)

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
