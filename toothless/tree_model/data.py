import json
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import polars as pl
import torch
from eggshell import rise  # type: ignore
from tokenizers import Tokenizer
from tokenizers.models import BPE
from torch import Tensor
from torch.utils import data

# from eggshell import TreeData
from toothless.utils import loading

CHUNK_SIZE = 10000

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SEP_TOKEN = "<sep>"
MASK_TOKEN = "<mask>"


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
        self.tokenizer = self._build_tokenizer()
        self.length, self.n_chunks = self._process()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        # TODO Implement an iterator for this
        chunk_path = self.processed_paths(idx // CHUNK_SIZE)
        chunk = _load_shard(chunk_path)
        sample = chunk[idx % CHUNK_SIZE]
        sample = {k: v.to_dense() for k, v in sample.items()}

        return sample

    def _process_raw(self):
        if not self.force_reload and self.raw_path.is_file():
            return

        df = loading.load_df(self.json_root)
        df.write_parquet(self.raw_path)
        return

    def _process(self) -> tuple[int, int]:
        if self.is_processed():
            with open(self.meta_path) as f:
                j = json.load(f)
                length = j["len"]
            return j["len"], j["n_chunks"]

        raw_expl_chains = pl.read_parquet(self.raw_path).get_column(
            "explanation_chain"
        )

        print("Starting parallel processing")
        expl_chains = [rise.PyRecExpr.batch_new(i) for i in raw_expl_chains]
        print("Parallel processing done")

        picked_tripples = [
            self._pick_indices(len(chain)) for chain in expl_chains
        ]
        length = sum([len(chain_pairs) for chain_pairs in picked_tripples])
        print(f"Total pairs: {length}")

        save_buffer = []
        spill_buffer = []
        n_chunks = 0
        for chain, tripple in zip(expl_chains, picked_tripples):
            pairs = [
                self._vectorize(
                    chain[left],
                    chain[middle],
                    chain[right],
                    middle / (right - left),
                )
                for left, middle, right in tripple
            ]

            remaining_space = CHUNK_SIZE - len(save_buffer)
            save_buffer.extend(pairs[:remaining_space])
            spill_buffer.extend(pairs[remaining_space:])

            if remaining_space == 0:
                print(
                    f"Saving {len(save_buffer)} pairs in {
                        self.processed_paths(n_chunks)
                    }"
                )
                torch.save(save_buffer, self.processed_paths(n_chunks))

                save_buffer = spill_buffer
                spill_buffer = []
                n_chunks += 1

        # Save last few in the buffer if exist
        if len(save_buffer) != 0:
            print(
                f"Saving the last {len(save_buffer)} pairs in {
                    self.processed_paths(n_chunks)
                }"
            )
            torch.save(save_buffer, self.processed_paths(n_chunks))

        with open(
            Path(self.processed_dir) / Path("meta.json"),
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump({"len": length, "n_chunks": n_chunks}, f)

        print("Data processed!")
        return length, n_chunks

    def _build_tokenizer(self) -> Tokenizer:
        # def _tokenizer(self, expl_chains, picked_tripples) -> Tokenizer:
        # if self.tokenizer_path.exists() and not self.force_reload:
        #     return Tokenizer.from_file(str(self.tokenizer_path))

        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        # tokenizer.normalizer = NormalizerSequence(
        #     [
        #         Strip(),
        #         BertNormalizer(clean_text=True, strip_accents=True, lowercase=True),
        #     ]  # type: ignore
        # )  # type: ignore
        # tokenizer.pre_tokenizer = PretokenizerSequence([Split(" ", behavior="removed")])  # type: ignore

        # trainer = BpeTrainer(
        #     vocab_size=1000,  # type: ignore
        #     special_tokens=[UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, PAD_TOKEN, MASK_TOKEN],  # type: ignore
        # )
        # iterator = self._gen(expl_chains, picked_tripples)
        # tokenizer.train_from_iterator(
        #     iterator=iterator, trainer=trainer, length=len(expl_chains) * 3 * self.pairs_per_expl * 50
        # )
        tokenizer.add_tokens(rise.operators())
        tokenizer.add_tokens(["[constant]", "[variable]"])
        tokenizer.save(str(self.tokenizer_path))
        return tokenizer

    def _gen(self, expl_chains, picked_tripples) -> Iterator[list[str]]:
        for chain, tripple in zip(expl_chains, picked_tripples):
            for left, middle, right in tripple:
                yield chain[left].to_data().values()
                yield chain[middle].to_data().values()
                yield chain[right].to_data().values()

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

    def _vectorize(
        self,
        left: rise.PyRecExpr,
        middle: rise.PyRecExpr,
        right: rise.PyRecExpr,
        distance: float,
    ) -> dict:
        l_tok, l_anc, l_sib = self._pyrec_to_tensor(left)
        x_tok, x_anc, x_sib = self._pyrec_to_tensor(middle)
        r_tok, r_anc, r_sib = self._pyrec_to_tensor(right)

        return {
            "l_tok": l_tok,
            "l_anc": l_anc,
            "l_sib": l_sib,
            "x_tok": x_tok,
            "x_anc": x_anc,
            "x_sib": x_sib,
            "r_tok": r_tok,
            "r_anc": r_anc,
            "r_sib": r_sib,
            "distance": torch.tensor([distance]),
        }

    def _pyrec_to_tensor(
        self, expr: rise.PyRecExpr
    ) -> tuple[Tensor, Tensor, Tensor]:
        graph_data = expr.to_data()
        # n_edges = graph_data.size()

        # node_ids = torch.tensor([node.id for node in graph_data.nodes], dtype=torch.int32)
        # tokenized_values = [
        #     tokenizer.encode(BOS_TOKEN + node.name + SEP_TOKEN + node.value + EOS_TOKEN)
        #     if node.value
        #     else tokenizer.encode(BOS_TOKEN + node.name + EOS_TOKEN)
        #     for node in graph_data.nodes
        # ]
        tokenized_values = torch.tensor(
            [
                self.tokenizer.token_to_id(node.name)
                for node in graph_data.nodes
            ],
            dtype=torch.int32,
        )
        #     for node in graph_data.nodes]

        # adjacency_matrix = torch.sparse_coo_tensor(
        #     torch.tensor(graph_data.transposed_adjacency()),
        #     torch.full((n_edges,), True, dtype=torch.bool),
        #     dtype=torch.bool,
        #     size=torch.Size((150, 150)),
        # )
        anc_matrix = torch.tensor(
            graph_data.anc_matrix(self.max_distance), dtype=torch.int32
        )
        sib_matrix = torch.tensor(
            graph_data.sib_matrix(self.max_distance), dtype=torch.int32
        )

        return tokenized_values, anc_matrix, sib_matrix

    def is_processed(self) -> bool:
        if not self.meta_path.is_file():
            return False
        with open(self.meta_path) as f:
            meta_info = json.load(f)

        return (
            not self.force_reload
            and all(
                [
                    self.processed_paths(i).is_file()
                    for i in range(0, meta_info["n_chunks"])
                ]
            )
            and self.tokenizer_path.is_file()
        )

    @property
    def raw_path(self) -> Path:
        return self.cache_dir / Path("df_raw.parquet")

    @property
    def processed_dir(self) -> Path:
        processed_dir = self.cache_dir / Path("processed")
        processed_dir.mkdir(exist_ok=True)
        return processed_dir

    @property
    def meta_path(self) -> Path:
        return self.processed_dir / Path("meta.json")

    @property
    def tokenizer_path(self) -> Path:
        return self.cache_dir / Path("tokenizer.json")

    def processed_paths(self, i) -> Path:
        return self.processed_dir / Path(f"data_{i}.pt")


@lru_cache(maxsize=100)
def _load_shard(chunk_path: Path):
    return torch.load(chunk_path, weights_only=False)


# def subsequent_mask(size: int):
#     attn_shape = (1, size, size)
#     sub_sequent_mask = torch.triu(np.ones(attn_shape), k=1).astype("uint8")
#     return torch.from_numpy(sub_sequent_mask) != 0


# def make_std_mask(nl: Tensor, pad: int):
#     "Create a mask to hide padding and future words."
#     nl_mask = (nl == pad).unsqueeze(-2)
#     nl_mask = nl_mask | subsequent_mask(nl.size(-1)).type_as(nl_mask.data)
#     return nl_mask
