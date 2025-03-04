# import logging
# import os
# from pathlib import Path
# import pickle
# import unicodedata
# from collections import Counter
# from tokenizers.trainers import BpeTrainer
# from tokenizers import Tokenizer
# from tokenizers.models import BPE


# from tqdm import tqdm

# PAD = 0
# UNK = 1
# BOS = 2
# EOS = 3

# PAD_WORD = "<pad>"
# UNK_WORD = "<unk>"
# BOS_WORD = "<s>"
# EOS_WORD = "</s>"

# log = logging.getLogger()


# def load_vocab(data_dir, is_split, data_type):
#     log.info(f"load vocab from {data_dir}, is_split = {is_split}")

#     return src_vocab


# def create_vocab(cache_dir: Path):
#     # create vocab
#     log.info("init vocab")
#     tokenizer = Tokenizer(BPE(unk_token="<unk>"))
#     trainer = BpeTrainer(special_tokens=["<unk>", "<pad>", "<s>", "</s>", "<mask>"])
#     tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer)
#     output_dir = cache_dir / "vocab.tok"
#     os.makedirs(output_dir, exist_ok=True)
#     tokenizer.save(output_dir)
#     return tokenizer
