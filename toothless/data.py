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
CACHE_BATCH_SIZE = 10000


@dataclass
class TripleFilePointer:
    midpoint_id: int
    batch_id: int
    index_in_batch: int

    def get_path(self, cache_path: Path) -> Path:
        return cache_path / f"{self.midpoint_id}-{self.batch_id}.json"

    def read(self, cache_path: Path) -> dict[str, str]:
        with open(self.get_path(cache_path), mode="r", encoding="utf-8") as f:
            file = json.load(f)
        return {
            "left": file["start_expr"],
            "middle": file["midpoint"]["midpoint"]["expression"],
            "right": file["midpoint"]["goals"][self.index_in_batch]["expression"],
        }


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
        self.sample_pointers = self._iterate_samples()

    def _build_vocab(self) -> SimpleVocab:
        normal_tokens = rise.operators() + ["[constant]", "[variable]"]
        vocab = SimpleVocab(PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN, normal_tokens)
        vocab.save(self.vocab_path)
        return vocab

    def _iterate_samples(self) -> list[TripleFilePointer]:
        print(self.json_root)

        json_files = list(self.json_root.glob("*.json"))
        json_files.sort()

        pointers = []

        for batch_file in tqdm(json_files, desc="Enumerating tripples"):
            midpoint_id, batch_id = batch_file.stem.split("-", 1)
            with open(batch_file, mode="r", encoding="utf-8") as f:
                file = json.load(f)
                for i in range(len(file["midpoint"]["goals"])):
                    pointers.append(TripleFilePointer(int(midpoint_id), int(batch_id), i))

        return pointers

    def __getitem__(self, idx: int) -> Triple:
        s = self.sample_pointers[idx].read(self.json_root)

        l_tree = rise.RecExpr(s["left"]).to_data()
        tgt_tree = rise.RecExpr(s["middle"]).to_data()
        r_tree = rise.RecExpr(s["right"]).to_data()

        l_ids = self._pyrec_to_tensor(l_tree)
        tgt_ids = self._pyrec_to_tensor(tgt_tree)
        r_ids = self._pyrec_to_tensor(r_tree)

        return Triple(l_ids, s["left"], tgt_ids, s["middle"], r_ids, s["right"])

    def _pyrec_to_tensor(self, tree_data: TreeData) -> Tensor:
        return torch.tensor(
            [self.vocab.bos_token_id]
            + [self.vocab.token2id(node.name) for node in tree_data.nodes()]
            + [self.vocab.eos_token_id],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.sample_pointers)


def split_off_special(partial_tok: list[str], vocab: SimpleVocab) -> list[str]:
    partial_tok = partial_tok[1:]
    for i, j in enumerate(partial_tok):
        if j in vocab.special_tokens:
            return partial_tok[:i]
    return partial_tok
