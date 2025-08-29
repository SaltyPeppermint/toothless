import tyro

from toothless.data import TripleDataSet
from toothless.args import DataArguments


if __name__ == "__main__":
    data_args = tyro.cli(DataArguments)
    dataset = TripleDataSet(data_args)
    print(len(dataset))
