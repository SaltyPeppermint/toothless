from tokenizers import models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
import tokenizers

from .data import TripleDataSet


def build_tokenizer(dataset: TripleDataSet, n_samples: int) -> Tokenizer:
    vocab_path = dataset.cache / "tokenizer.json"
    if vocab_path.is_file():
        return Tokenizer.from_file(str(vocab_path))

    tokenizer = Tokenizer(models.BPE())

    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFKC(), normalizers.Replace(tokenizers.Regex(r"mf(i|u)\d*"), "[var]")]  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Split(")", behavior="isolated"),
            pre_tokenizers.Split("(", behavior="isolated"),
        ]
    )  # pyright: ignore[reportAttributeAccessIssue]
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )  # pyright: ignore[reportAttributeAccessIssue]

    trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["[PAD]", "[CLS]", "[SEP]"])  # pyright: ignore[reportCallIssue]

    tokenizer.train_from_iterator(dataset.get_tokenizer_training_corpus(n_samples), trainer=trainer, length=n_samples)
    tokenizer.save(str(vocab_path))
    return tokenizer
