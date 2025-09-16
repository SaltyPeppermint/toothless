import tyro

from toothless.data import TripleDataSet
from toothless.args import DataArgs
from toothless.tokenizer import build_tokenizer


if __name__ == "__main__":
    data_args = tyro.cli(DataArgs)
    dataset = TripleDataSet(data_args)
    build_tokenizer(dataset, data_args.tokenizer_samples)
    print(len(dataset))
