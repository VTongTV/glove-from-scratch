import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.vocabulary import Vocabulary
from data.cooccurrence import CooccurrenceMatrix

def _make_vocab_and_matrix(tokens, window_size=2, context_type="symmetric"):
    vocab = Vocabulary()
    vocab.count(tokens)
    vocab.build_indices()
    mat = CooccurrenceMatrix(vocab, window_size=window_size, context_type=context_type)
    mat.build([tokens])
    return vocab, mat

def test_basic_cooccurrence():
    vocab, mat = _make_vocab_and_matrix(["a", "b", "c", "d"], window_size=2)
    assert mat.entries["b"]["a"] > 0
    assert mat.entries["b"]["c"] > 0

def test_distance_weighting():
    vocab, mat = _make_vocab_and_matrix(["a", "b", "c"], window_size=2)
    ab_dist = 1
    ac_dist = 2
    assert abs(mat.entries["a"]["b"] - (1.0 / ab_dist)) < 1e-9
    assert abs(mat.entries["a"]["c"] - (1.0 / ac_dist)) < 1e-9

def test_symmetric_context():
    vocab, mat = _make_vocab_and_matrix(["a", "b", "c", "d"], window_size=2, context_type="symmetric")
    left_count = mat.entries["c"]["b"]
    right_count = mat.entries["c"]["d"]
    assert left_count > 0
    assert right_count > 0

def test_asymmetric_context():
    vocab, mat = _make_vocab_and_matrix(["a", "b", "c", "d"], window_size=2, context_type="asymmetric")
    left_count = mat.entries["c"]["b"]
    right_count = mat.entries["c"]["d"]
    assert left_count > 0
    assert right_count == 0

def test_save_load_roundtrip():
    vocab, mat = _make_vocab_and_matrix(["a", "b", "c"], window_size=2)
    path = os.path.join(os.path.dirname(__file__), "..", "test_cooc.npz")
    mat.save(path)
    mat2 = CooccurrenceMatrix(vocab, window_size=2)
    mat2.load(path)
    for i_word, contexts in mat.entries.items():
        for j_word, count in contexts.items():
            assert abs(mat2.entries[i_word][j_word] - count) < 1e-9
    os.remove(path if os.path.exists(path) else path + ".npz")

def test_nonzero_entries():
    vocab, mat = _make_vocab_and_matrix(["x", "y", "z"], window_size=2)
    entries = list(mat.nonzero_entries())
    assert len(entries) > 0
    for i, j, v in entries:
        assert v > 0
