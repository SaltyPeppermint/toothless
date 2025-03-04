# import re
# import wordninja

from pathlib import Path
import torch
from torch import Tensor
import torch.utils.data as data
from torch_geometric.data import Data

from tqdm import tqdm
import numpy as np

import string
from utils import PAD, UNK

punc = string.punctuation


class BaseASTDataSet(data.Dataset):
    def __init__(self, config, data_set_name: str):
        super(BaseASTDataSet, self).__init__()
        self.data_set_name = data_set_name
        print(f"loading {data_set_name} data...")
        data_dir = Path(config.data_dir) / data_set_name

        self.ignore_more_than_k = config.is_ignore
        self.max_rel_pos = config.max_rel_pos
        self.max_src_len = config.max_src_len
        self.max_tgt_len = config.max_tgt_len

        ast_path = data_dir / "split_pot.seq" if config.is_split else Path("un_split_pot.seq")
        matrices_path = data_dir / "split_matrices.npz" if config.is_split else Path("un_split_matrices.npz")

        self.ast_data = _load_list(ast_path)
        self.nl_data = _load_seq(data_dir / "nl.original")
        self.matrices_data = _load_matrices(matrices_path)

        self.data_set_len = len(self.ast_data)
        self.src_vocab = config.src_vocab
        self.tgt_vocab = config.tgt_vocab
        # self.collector = Collater([], [])

    # def collect_fn(self, batch):
    #     return self.collector.collate(batch)

    def __len__(self):
        return self.data_set_len

    # def __getitem__(self, index):  # -> T_co:
    #     pass

    def convert_ast_to_tensor(self, ast_seq):
        ast_seq = ast_seq[: self.max_src_len]
        return word2tensor(ast_seq, self.max_src_len, self.src_vocab)

    def convert_nl_to_tensor(self, nl):
        nl = nl[: self.max_tgt_len - 2]
        nl = ["<s>"] + nl + ["</s>"]
        return word2tensor(nl, self.max_tgt_len, self.tgt_vocab)


class FastASTDataSet(BaseASTDataSet):
    def __init__(self, config, data_set_name):
        print("Data Set Name : < Fast AST Data Set >")
        super(FastASTDataSet, self).__init__(config, data_set_name)
        self.max_anc_rel_pos = config.max_anc_rel_pos
        self.max_sib_rel_pos = config.max_sib_rel_pos
        self.edges_data = self.convert_ast_to_edges()

    def convert_ast_to_edges(self):
        print("building edges.")

        anc_edge_data = self.matrices_data["ancestor"]
        sib_edge_data = self.matrices_data["sibling"]

        edges_data = []

        def edge2list(edges, edge_type):
            if edge_type == "ancestor":
                max_rel_pos = self.max_anc_rel_pos
            elif edge_type == "sibling":
                max_rel_pos = self.max_sib_rel_pos
            else:
                raise ValueError(f"Unknown edge type: {edge_type}")
            ast_len = min(len(edges), self.max_src_len)
            start_node = -1 * torch.ones((self.max_rel_pos + 1, self.max_src_len), dtype=torch.long)
            for key in edges.keys():
                if key[0] < self.max_src_len and key[1] < self.max_src_len:
                    value = edges.get(key)
                    if value > max_rel_pos and self.ignore_more_than_k:
                        continue
                    value = min(value, max_rel_pos)
                    start_node[value][key[1]] = key[0]

            start_node[0][:ast_len] = torch.arange(ast_len)
            return start_node

        for i in tqdm(range(self.data_set_len)):
            anc_edges = anc_edge_data[i]
            sib_edges = sib_edge_data[i]
            ast_seq = self.ast_data[i]
            nl = self.nl_data[i]

            anc_edge_list = edge2list(anc_edges, "ancestor")
            sib_edge_list = edge2list(sib_edges, "sibling")

            ast_vec = self.convert_ast_to_tensor(ast_seq)
            nl_vec = self.convert_nl_to_tensor(nl)

            data = Data(
                src_seq=ast_vec,
                anc_edges=anc_edge_list,
                sib_edges=sib_edge_list,
                tgt_seq=nl_vec[:-1],
                target=nl_vec[1:],
            )

            edges_data.append(data)

        return edges_data

    def __getitem__(self, index):
        return self.edges_data[index], self.edges_data[index].target


def word2tensor(seq, max_seq_len: int, vocab):
    seq_vec = [vocab.w2i[x] if x in vocab.w2i else UNK for x in seq]
    seq_vec = seq_vec + [PAD for i in range(max_seq_len - len(seq_vec))]
    seq_vec = torch.tensor(seq_vec, dtype=torch.long)
    return seq_vec


def _load_list(file_path: Path):
    _data = []
    print(f"loading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            _data.append(eval(line))
    return _data


def _load_seq(file_path):
    data_ = []
    print(f"loading {file_path} ...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            data_.append(line.split())
    return data_


def _load_matrices(file_path):
    print("loading matrices...")
    matrices = np.load(file_path, allow_pickle=True)
    return matrices


def subsequent_mask(size: int):
    attn_shape = (1, size, size)
    sub_sequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(sub_sequent_mask) != 0


def make_std_mask(nl: Tensor, pad: int):
    "Create a mask to hide padding and future words."
    nl_mask = (nl == pad).unsqueeze(-2)
    nl_mask = nl_mask | subsequent_mask(nl.size(-1)).type_as(nl_mask.data)
    return nl_mask


# def clean_nl(s):
#     s = s.strip()
#     if s[-1] == ".":
#         s = s[:-1]
#     s = s.split(". ")[0]
#     s = re.sub(r"[<].+?[>]", "", s)
#     s = re.sub(r"[[]%]", "", s)
#     s = s[0:1].lower() + s[1:]
#     processed_words = []
#     for w in s.split():
#         if w not in punc:
#             processed_words.extend(wordninja.split(w))
#         else:
#             processed_words.append(w)
#     return processed_words
