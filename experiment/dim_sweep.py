import config
from data.tokenizer import tokenize_file
from data.vocabulary import Vocabulary
from data.cooccurrence import CooccurrenceMatrix
from glove.model import GloVeModel
from glove.train import prepare_training_data, train
from glove.io import save_vectors
from eval.analogy import load_analogies, evaluate_analogies
import json

def run(corpus_path, analogy_path, output_dir="outputs"):
    vocab = Vocabulary()
    vocab.count_from_file(corpus_path, tokenize_file)
    vocab.filter_top_n(config.VOCAB_SIZE)
    cooc = CooccurrenceMatrix(vocab)
    cooc.build(tokenize_file(corpus_path))
    rows, cols, vals = prepare_training_data(cooc)
    analogies = load_analogies(analogy_path)
    results = {}
    for dim in config.DIMENSIONALITIES:
        model = GloVeModel(len(vocab), dim)
        train(model, rows, cols, vals)
        vectors = model.get_vectors(vocab)
        acc = evaluate_analogies(vectors, analogies)
        results[dim] = acc
    with open(f"{output_dir}/dim_sweep.json", "w") as f:
        json.dump(results, f, default=str)
