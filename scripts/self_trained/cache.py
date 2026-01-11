import tyro

from toothless.self_trained.args import DataArgs
from toothless.self_trained.data import TripleDataSet

if __name__ == "__main__":
    data_args = tyro.cli(DataArgs)
    dataset = TripleDataSet(data_args)
    print(len(dataset))
