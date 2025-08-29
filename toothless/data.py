import json
from pathlib import Path
from dataclasses import dataclass
from functools import cmp_to_key

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
CACHE_BATCH_SIZE = 10000


@dataclass
class SampleStore:
    starts: list[str]
    guides: list[str]
    targets: list[str]
    index_tripples: list[tuple[int, int, int]]

    def __len__(self):
        return len(self.index_tripples)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        s_idx, g_idx, t_idx = self.index_tripples[idx]
        return self.starts[s_idx], self.guides[g_idx], self.targets[t_idx]


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

        self.index_cache = self.cache / "index_cache.pickle"
        self.vocab_path = self.cache / "vocab.json"

        self.vocab = self._build_vocab()
        self.sample_store = self._iterate_samples()

    def _build_vocab(self) -> SimpleVocab:
        normal_tokens = rise.operators() + ["[constant]", "[variable]"]
        vocab = SimpleVocab(PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN, normal_tokens)
        vocab.save(self.vocab_path)
        return vocab

    def _iterate_samples(self) -> SampleStore:
        print(self.json_root)

        json_files = list(self.json_root.glob("*.json"))
        json_files.sort(key=cmp_to_key(path_cmp))

        with open(json_files[0], mode="r", encoding="utf-8") as f:
            file = json.load(f)
            starts = [file["start_expr"]]

        guides = []
        targets = []

        pointers = []

        for batch_file in tqdm(json_files, desc="Enumerating tripples"):
            midpoint_id, _ = batch_file.stem.split("-", 1)

            with open(batch_file, mode="r", encoding="utf-8") as f:
                file = json.load(f)
                if len(guides) < int(midpoint_id):
                    guides.append(file["midpoint"]["midpoint"]["expression"])
                for goal in file["midpoint"]["goals"]:
                    targets.append(goal)
                    pointers.append(
                        (len(starts) - 1, len(guides) - 1, len(targets) - 1),
                    )

        return SampleStore(starts, guides, targets, pointers)

    def __getitem__(self, idx: int) -> Triple:
        left, middle, right = self.sample_store[idx]

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

    def __len__(self) -> int:
        return len(self.sample_store)


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
