import json
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import polars as pl
from eggshell import rise  # type: ignore
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import BertNormalizer
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
from tokenizers.pre_tokenizers import Split
from tokenizers.normalizers import Strip
from tokenizers.normalizers import Sequence as NormalizerSequence

import torch
from torch import Tensor
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


# from eggshell import TreeData
from toothless.utils import loading

CHUNK_SIZE = 128

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
# BOS_TOKEN = "<s>"
# EOS_TOKEN = "</s>"
# SEP_TOKEN = "<sep>"
MASK_TOKEN = "<mask>"


class SimpleVocab:
    def __init__(self, pad_token: str, unk_token: str, mask_token: str, tokens: list[str]):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.vocab = {}
        for id, token in enumerate([pad_token, unk_token, mask_token] + tokens):
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
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def mask_token_id(self):
        return self.vocab[self.mask_token]

    @property
    def unk_token_id(self):
        return self.vocab[self.unk_token_id]

    def save(self, path: Path):
        d = {}
        d["pad_token"] = self.pad_token
        d["unk_token"] = self.unk_token
        d["mask_token"] = self.mask_token
        d["vocab"] = self.vocab
        with open(path, mode="w", encoding="utf-8") as f:
            json.dump(d, f)

    @staticmethod
    def load(path: Path):
        with open(path, mode="r", encoding="utf-8") as f:
            d = json.load(f)

        v = [d["vocab"][i] for i in range(3, len(d["vocab"]))]
        return SimpleVocab(d["pad_token"], d["unk_token"], d["mask_token"], v)


class CustomDataset(data.Dataset):
    def __init__(
        self,
        json_root: Path,
        pairs_per_expl: int,
        max_distance: int = 32,
        random_state: int = 42,
        force_reload: bool = False,
    ):
        self.json_root = Path(json_root)
        self.pairs_per_expl = pairs_per_expl
        self.max_distance = max_distance
        self.force_reload = force_reload
        torch.manual_seed(random_state)

        self.cache_dir = Path("cache") / Path(*self.json_root.parts[2:])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._process_raw()
        self.vocab = self._build_vocab()
        self.samples = self._process()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        # TODO Implement an iterator for this
        sample = self.samples[idx]
        right = sample["right"].item()
        left = sample["left"].item()
        middle = sample["middle"].item()
        distance = sample["distance"].item()

        vectorized = self._vectorize(right, middle, left, distance)
        return vectorized

    def _process_raw(self):
        if not self.force_reload and self.raw_path.is_file():
            return

        df = loading.load_df(self.json_root)
        df.write_parquet(self.raw_path)
        return

    def _process(self) -> pl.DataFrame:
        if not self.force_reload and self.processed_path.is_file():
            return pl.read_parquet(self.processed_path)

        raw_data = pl.read_parquet(self.raw_path)
        expl_chains = raw_data.get_column("explanation_chain")

        # print("Starting parallel processing")
        # print("Parallel processing done")

        picked_tripples = [self._pick_indices(len(chain)) for chain in expl_chains]
        length = sum([len(chain_pairs) for chain_pairs in picked_tripples])
        print(f"Total pairs: {length}")

        total_samples = {"left": [], "right": [], "middle": [], "distance": []}
        for chain, tripple in zip(expl_chains, picked_tripples):
            for left, middle, right in tripple:
                right = int(right)
                left = int(left)
                middle = int(middle)
                total_samples["left"].append(str(chain[left]))
                total_samples["right"].append(str(chain[right]))
                total_samples["middle"].append(str(chain[middle]))
                total_samples["distance"].append(middle / (right - left))

        df = pl.DataFrame(total_samples)
        df.write_parquet(self.processed_path)
        print(f"Total samples: {len(total_samples)}")

        print("Data processed!")
        return df

    def _build_vocab(self) -> SimpleVocab:
        vocab = SimpleVocab(PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, rise.operators() + ["[constant]", "[variable]"])
        vocab.save(self.vocab_path)
        return vocab

    # def _gen(self, expl_chains, picked_tripples) -> Iterator[list[str]]:
    #     for chain, tripple in zip(expl_chains, picked_tripples):
    #         for left, middle, right in tripple:
    #             yield chain[left].to_data().values()
    #             yield chain[middle].to_data().values()
    #             yield chain[right].to_data().values()

    def _pick_indices(self, max_index: int) -> set[tuple[int, int, int]]:
        xs = torch.randint(0, max_index, (self.pairs_per_expl,))
        ys = torch.randint(0, max_index, (self.pairs_per_expl,))
        r = set()
        for x, y in zip(xs, ys):
            distance = torch.abs(x - y)
            if distance < 2:
                continue
            r.add((torch.minimum(x, y), distance // 2, torch.maximum(x, y)))

        return r

    def _vectorize(self, left: str, middle: str, right: str, distance: float) -> dict:
        l_ids, l_anc, l_sib = self._pyrec_to_tensor(rise.PyRecExpr(left))
        tgt_ids, tgt_anc, tgt_sib = self._pyrec_to_tensor(rise.PyRecExpr(middle))
        r_ids, r_anc, r_sib = self._pyrec_to_tensor(rise.PyRecExpr(right))

        return {
            "l_ids": l_ids,
            "l_anc": l_anc,
            "l_sib": l_sib,
            "tgt_ids": tgt_ids,
            "tgt_anc": tgt_anc,
            "tgt_sib": tgt_sib,
            "r_ids": r_ids,
            "r_anc": r_anc,
            "r_sib": r_sib,
            "distance": torch.tensor([distance]),
        }

    def _pyrec_to_tensor(self, expr: rise.PyRecExpr) -> tuple[Tensor, Tensor, Tensor]:
        graph_data = expr.to_data()
        # n_edges = graph_data.size()

        # node_ids = torch.tensor([node.id for node in graph_data.nodes], dtype=torch.int32)
        # tokenized_values = [
        #     tokenizer.encode(BOS_TOKEN + node.name + SEP_TOKEN + node.value + EOS_TOKEN)
        #     if node.value
        #     else tokenizer.encode(BOS_TOKEN + node.name + EOS_TOKEN)
        #     for node in graph_data.nodes
        # ]
        ids = torch.tensor([self.vocab.token2id(node.name) for node in graph_data.nodes], dtype=torch.int32)
        #     for node in graph_data.nodes]

        # adjacency_matrix = torch.sparse_coo_tensor(
        #     torch.tensor(graph_data.transposed_adjacency()),
        #     torch.full((n_edges,), True, dtype=torch.bool),
        #     dtype=torch.bool,
        #     size=torch.Size((150, 150)),
        # )
        anc_matrix = torch.tensor(graph_data.anc_matrix(self.max_distance), dtype=torch.int32)
        sib_matrix = torch.tensor(graph_data.sib_matrix(self.max_distance), dtype=torch.int32)

        return ids, anc_matrix, sib_matrix

    @property
    def raw_path(self) -> Path:
        return self.cache_dir / Path("df_raw.parquet")

    @property
    def vocab_path(self) -> Path:
        return self.cache_dir / Path("vocab.json")

    @property
    def processed_path(self) -> Path:
        return self.cache_dir / Path(f"processed.parquet")


# # @lru_cache(maxsize=100)
# def _load_shard(chunk_path: Path):
#     return torch.load(chunk_path, weights_only=False)


class DictCollator:
    def __init__(self, pad_id: int, max_len: int):
        self.pad_id = pad_id
        self.max_len = max_len

    def __call__(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        # batch is a list of dictionaries
        batched_data = {}

        # Iterate over each key in the dictionary
        for key in batch[0].keys():
            # If the value is a tensor and represents a sequence (e.g., 1D or 2D tensor)
            if isinstance(batch[0][key], torch.Tensor):
                # Stack or pad the tensors for each key
                if batch[0][key].dim() == 0:
                    # If it's a scalar tensor, stack them
                    batched_data[key] = torch.stack([sample[key] for sample in batch])
                elif batch[0][key].dim() == 1:  # Check if it's a sequence (1D or higher)
                    # Pad sequences to the same length
                    samples = [sample[key] for sample in batch]
                    # Generate padding directions depending on max_len
                    paddings = [(0, self.max_len - s.size(-1)) for s in samples]
                    padded_elements = [F.pad(s, p, "constant", self.pad_id) for s, p in zip(samples, paddings)]

                    batched_data[key] = torch.stack(padded_elements)
                elif batch[0][key].dim() == 2:
                    # Find largest dimensions of the square 2-dimensional matrices
                    # (they are always square)
                    # Get elements in batch
                    samples = [sample[key] for sample in batch]
                    # Generate padding directions depending on largest element in batch
                    paddings = [(0, self.max_len - s.size(-1), 0, self.max_len - s.size(-2)) for s in samples]
                    padded_elements = [F.pad(s, p, "constant", self.pad_id) for s, p in zip(samples, paddings)]

                    batched_data[key] = torch.stack(padded_elements)
                else:
                    print(batch[0][key])
                    raise ValueError("Cannot deal with dimensions bigger than 2")
            else:
                # If the value is not a tensor, just collect them in a list
                batched_data[key] = [sample[key] for sample in batch]

        # for k, v in batched_data.items():
        #     print(k)
        #     print(v.size())
        return batched_data


# def subsequent_mask(size: int):
#     attn_shape = (1, size, size)
#     sub_sequent_mask = torch.triu(np.ones(attn_shape), k=1).astype("uint8")
#     return torch.from_numpy(sub_sequent_mask) != 0


# def make_std_mask(nl: Tensor, pad: int):
#     "Create a mask to hide padding and future words."
#     nl_mask = (nl == pad).unsqueeze(-2)
#     nl_mask = nl_mask | subsequent_mask(nl.size(-1)).type_as(nl_mask.data)
#     return nl_mask
