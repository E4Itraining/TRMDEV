import torch
import pytest
from trm.data.synthetic import SyntheticRecursionDataset


@pytest.fixture
def dataset():
    return SyntheticRecursionDataset(
        num_samples=100, seq_len=16, vocab_size=64, num_classes=5, max_nest=3, seed=0
    )


def test_length(dataset):
    assert len(dataset) == 100


def test_item_keys(dataset):
    item = dataset[0]
    assert "input_ids" in item
    assert "label" in item
    assert "nest_depth" in item


def test_item_shapes(dataset):
    item = dataset[0]
    assert item["input_ids"].shape == (16,)
    assert item["label"].shape == ()
    assert item["nest_depth"].shape == ()


def test_label_range(dataset):
    for i in range(len(dataset)):
        label = dataset[i]["label"].item()
        assert 0 <= label < 5


def test_vocab_range(dataset):
    for i in range(len(dataset)):
        ids = dataset[i]["input_ids"]
        assert (ids >= 0).all() and (ids < 64).all()


def test_reproducibility():
    ds1 = SyntheticRecursionDataset(num_samples=10, seed=42)
    ds2 = SyntheticRecursionDataset(num_samples=10, seed=42)
    for i in range(10):
        assert torch.equal(ds1[i]["input_ids"], ds2[i]["input_ids"])
        assert ds1[i]["label"] == ds2[i]["label"]


def test_nested_reduce():
    tokens = [1, 2, 3, 4]
    # depth=1: sum([1,2,3,4]) % 10 = 10 % 10 = 0
    assert SyntheticRecursionDataset._nested_reduce(tokens, 1, 10) == 0
    # depth=2: left=[1,2]→3, right=[3,4]→7, (3+7)%10 = 0
    assert SyntheticRecursionDataset._nested_reduce(tokens, 2, 10) == 0
