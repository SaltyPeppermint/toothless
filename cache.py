import tyro

from toothless.data import TripleDataSet
from toothless.args import DataArgs


if __name__ == "__main__":
    data_args = tyro.cli(DataArgs)
    dataset = TripleDataSet(data_args)
    print(len(dataset))
