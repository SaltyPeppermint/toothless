import json
from pathlib import Path

import polars as pl

# import matplotlib.pyplot as plt
from eggshell import rise  # type: ignore
# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.normalizers import BertNormalizer
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
# from tokenizers.pre_tokenizers import Split
# from tokenizers.normalizers import Strip
# from tokenizers.normalizers import Sequence as NormalizerSequence

import torch
from torch import Tensor
from torch.utils import data
import torch.nn.functional as F


# from eggshell import TreeData
from toothless.utils import loading

CHUNK_SIZE = 128

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
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def mask_token_id(self):
        return self.vocab[self.mask_token]

    @property
    def unk_token_id(self):
        return self.vocab[self.unk_token_id]

    @property
    def bos_token_id(self):
        return self.vocab[self.bos_token]

    @property
    def eos_token_id(self):
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

        v = [d["vocab"][i] for i in range(3, len(d["vocab"]))]
        return SimpleVocab(d["pad_token"], d["unk_token"], d["mask_token"], d["bos_token"], d["eos_token"], v)


class CustomDataset(data.Dataset):
    def __init__(
        self,
        json_root: Path,
        pairs_per_expl: int,
        max_rel_distance: int = 15,
        random_state: int = 42,
        force_reload: bool = False,
    ):
        self.json_root = Path(json_root)
        self.pairs_per_expl = pairs_per_expl
        self.max_rel_distance = max_rel_distance
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
        normal_tokens = rise.operators() + ["[constant]", "[variable]"]
        vocab = SimpleVocab(PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN, normal_tokens)
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
        ids = torch.tensor(
            [self.vocab.bos_token_id]
            + [self.vocab.token2id(node.name) for node in graph_data.nodes]
            + [self.vocab.eos_token_id],
            dtype=torch.int64,
        )
        #     for node in graph_data.nodes]

        # adjacency_matrix = torch.sparse_coo_tensor(
        #     torch.tensor(graph_data.transposed_adjacency()),
        #     torch.full((n_edges,), True, dtype=torch.bool),
        #     dtype=torch.bool,
        #     size=torch.Size((150, 150)),
        # )
        anc_matrix = torch.tensor(graph_data.anc_matrix(self.max_rel_distance, double_pad=True), dtype=torch.int64)
        sib_matrix = torch.tensor(graph_data.sib_matrix(self.max_rel_distance, double_pad=True), dtype=torch.int64)

        return ids, anc_matrix, sib_matrix

    @property
    def raw_path(self) -> Path:
        return self.cache_dir / Path("df_raw.parquet")

    @property
    def vocab_path(self) -> Path:
        return self.cache_dir / Path("vocab.json")

    @property
    def processed_path(self) -> Path:
        return self.cache_dir / Path("processed.parquet")


def make_std_mask(tgt: Tensor, pad_id: int):
    "Create a mask to hide padding and future words."
    # print(f"tgt shape: {tgt.size()}")
    tgt_mask = (tgt == pad_id).unsqueeze(-2)
    # plt.imsave("padding_mask.png", tgt_mask[0])
    # print(f"tgt_mask: {tgt_mask.size()}")
    triangle_mask = triangle_matrix(tgt.size(-1))
    # print(f"triangle_matrix: {triangle_mask.size()}")
    # plt.imsave("triangle_mask.png", triangle_mask)
    tgt_mask = tgt_mask | triangle_mask
    # print(f"combined_mask: {tgt_mask.size()}")
    # plt.imsave("combined_mask.png", tgt_mask[0])
    # print("===")
    return tgt_mask.unsqueeze(-3)


def triangle_matrix(sz: int, device: torch.device | None = None) -> Tensor:
    m = torch.full((sz, sz), True, device=device, dtype=torch.bool)
    return torch.triu(m, diagonal=1)


class DictCollator:
    def __init__(self, pad_id: int, max_len: int):
        self.pad_id = pad_id
        self.max_len = max_len

    def __call__(self, batch: list[dict[str, Tensor]]) -> tuple[dict[str, Tensor], int]:
        # batch is a list of dictionaries
        batched_data = {}

        # Iterate over each key in the dictionary
        for key in batch[0].keys():
            # If the value is a tensor and represents a sequence (e.g., 1D or 2D tensor)
            if isinstance(batch[0][key], torch.Tensor):
                # Stack or pad the tensors for each key
                if batch[0][key].dim() == 0:  # If it's a scalar tensor, stack them
                    batched_data[key] = torch.stack([sample[key] for sample in batch])
                elif batch[0][key].dim() == 1:  # Check if it's a sequence (1D)
                    batched_data[key] = self.pad_1d([sample[key] for sample in batch], key)
                elif batch[0][key].dim() == 2:  # Check if it's a 2D matrix
                    batched_data[key] = self.pad_2d([sample[key] for sample in batch], key)
                else:
                    raise ValueError(f"Cannot deal with dimensions bigger than 2: {batch[0][key]}")
            else:
                # If the value is not a tensor, just collect them in a list
                batched_data[key] = [sample[key] for sample in batch]

        batched_data["l_mask"] = (batched_data["l_ids"] == self.pad_id).unsqueeze(-2).unsqueeze(-2)
        batched_data["r_mask"] = (batched_data["r_ids"] == self.pad_id).unsqueeze(-2).unsqueeze(-2)

        if batched_data["tgt_ids"] is not None:
            # The _y versions are always shifted right.
            # For matrices this means right and down.
            full_tgt_ids = batched_data["tgt_ids"]
            batched_data["tgt_ids"] = full_tgt_ids[:, :-1]
            batched_data["tgt_ids_y"] = full_tgt_ids[:, 1:]
            batched_data["tgt_mask"] = make_std_mask(batched_data["tgt_ids"], self.pad_id)

            full_tgt_anc = batched_data["tgt_anc"]
            batched_data["tgt_anc"] = full_tgt_anc[:, :-1, :-1]
            batched_data["tgt_anc_y"] = full_tgt_anc[:, :1, :1]

            full_tgt_sib = batched_data["tgt_sib"]
            batched_data["tgt_sib"] = full_tgt_sib[:, :-1, :-1]
            batched_data["tgt_sib_y"] = full_tgt_sib[:, :1, :1]

        return batched_data, (batched_data["tgt_ids_y"] != self.pad_id).data.sum()

    def pad_1d(self, samples: list[Tensor], key: str) -> Tensor:
        # Pad sequences to the same length
        # Generate padding directions depending on max_len
        pad_len = self.max_len
        if "tgt" in key:
            pad_len += 1  # Extra padding since tgt will be shifted
        paddings = [(0, pad_len - s.size(-1)) for s in samples]
        padded_elements = [F.pad(s, p, "constant", self.pad_id) for s, p in zip(samples, paddings)]
        return torch.stack(padded_elements)

    def pad_2d(self, samples: list[Tensor], key: str) -> Tensor:
        # Find largest dimensions of the square 2-dimensional matrices
        # (they are always square)
        # Generate padding directions depending on largest element in batch
        pad_len = self.max_len
        if "tgt" in key:
            pad_len += 1  # Extra padding since tgt will be shifted
        paddings = [(0, pad_len - s.size(-1), 0, pad_len - s.size(-2)) for s in samples]
        padded_elements = [F.pad(s, p, "constant", self.pad_id) for s, p in zip(samples, paddings)]
        return torch.stack(padded_elements)


# def subsequent_mask(size: int):
#     attn_shape = (1, size, size)
#     sub_sequent_mask = torch.triu(np.ones(attn_shape), k=1).astype("uint8")
#     return torch.from_numpy(sub_sequent_mask) != 0


# def make_std_mask(nl: Tensor, pad: int):
#     "Create a mask to hide padding and future words."
#     nl_mask = (nl == pad).unsqueeze(-2)
#     nl_mask = nl_mask | subsequent_mask(nl.size(-1)).type_as(nl_mask.data)
#     return nl_mask
