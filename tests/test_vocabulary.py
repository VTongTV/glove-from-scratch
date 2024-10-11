import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.vocabulary import Vocabulary

def test_count():
    vocab = Vocabulary()
    vocab.count(["the", "cat", "sat", "on", "the", "mat"])
    assert vocab.word2count["the"] == 2
    assert vocab.word2count["cat"] == 1
    assert vocab.total_tokens == 6

def test_build_indices():
    vocab = Vocabulary()
    vocab.count(["b", "a", "a", "c", "a", "b"])
    vocab.build_indices()
    assert vocab.word2idx["a"] == 0
    assert vocab.word2idx["b"] == 1
    assert vocab.word2idx["c"] == 2
    assert vocab.idx2word[0] == "a"

def test_filter_top_n():
    vocab = Vocabulary()
    vocab.count(["a", "b", "a", "c", "a", "b"])
    vocab.filter_top_n(2)
    assert "a" in vocab
    assert "b" in vocab
    assert "c" not in vocab
    assert len(vocab) == 2

def test_contains():
    vocab = Vocabulary()
    vocab.count(["hello", "world"])
    vocab.build_indices()
    assert "hello" in vocab
    assert "missing" not in vocab
