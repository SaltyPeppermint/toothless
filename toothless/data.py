import json
from pathlib import Path
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from tqdm.auto import tqdm
import polars as pl

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
SCHEMA = {"from": pl.UInt64, "to": pl.UInt64, "path": pl.String}


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

        self.index_table_cache_path = self.cache / "index_table_cache.csv"
        self.index_table = self._iterate_samples()

        self.n_samples = conf.n_samples

        self.vocab_path = self.cache / "vocab.json"
        self.vocab = self._build_vocab()

    def _build_vocab(self) -> SimpleVocab:
        normal_tokens = rise.operators() + ["[constant]", "[variable]"]
        vocab = SimpleVocab(PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN, normal_tokens)
        vocab.save(self.vocab_path)
        return vocab

    def _iterate_samples(self) -> pl.DataFrame:
        if self.index_table_cache_path.is_file():
            return pl.read_csv(self.index_table_cache_path, schema=SCHEMA)

        json_files = list(self.json_root.glob("*.json"))
        json_files.sort()

        entries = []
        i = 0

        for batch_file in tqdm(json_files, desc="Enumerating tripples"):
            with open(batch_file, mode="r", encoding="utf-8") as f:
                file = json.load(f)
            j = i + len(file["midpoint"]["goals"])
            entries.append([i, j, str(batch_file)])

            i = j
        df = pl.DataFrame(entries, schema=SCHEMA, orient="row")
        df.write_csv(self.index_table_cache_path)

        return df

    def load_str_triple(self, idx: int) -> tuple[str, str, str]:
        result = self.index_table.filter((pl.col("from") <= idx) & (idx < pl.col("to"))).row(0, named=True)

        with open(str(result["path"]), mode="r", encoding="utf-8") as f:
            file = json.load(f)
        start = file["start_expr"]
        guide = file["midpoint"]["midpoint"]["expression"]
        target = file["midpoint"]["goals"][idx - result["from"]]["expression"]
        return start, guide, target

    def _pyrec_to_tensor(self, tree_data: TreeData) -> Tensor:
        return torch.tensor(
            [self.vocab.bos_token_id]
            + [self.vocab.token2id(node.name) for node in tree_data.nodes()]
            + [self.vocab.eos_token_id],
            dtype=torch.long,
        )

    def __getitem__(self, idx: int) -> Triple:
        left, middle, right = self.load_str_triple(idx)

        l_tree = rise.RecExpr(left).to_data()
        tgt_tree = rise.RecExpr(middle).to_data()
        r_tree = rise.RecExpr(right).to_data()

        l_ids = self._pyrec_to_tensor(l_tree)
        tgt_ids = self._pyrec_to_tensor(tgt_tree)
        r_ids = self._pyrec_to_tensor(r_tree)

        return Triple(l_ids, left, tgt_ids, middle, r_ids, right)

    def __len__(self):
        total_samples = self.index_table["to"].max()
        if self.n_samples is not None:
            return min(total_samples, self.n_samples)  # pyright: ignore[reportArgumentType]
        return total_samples


def split_off_special(partial_tok: list[str], vocab: SimpleVocab) -> list[str]:
    partial_tok = partial_tok[1:]
    for i, j in enumerate(partial_tok):
        if j in vocab.special_tokens:
            return partial_tok[:i]
    return partial_tok
