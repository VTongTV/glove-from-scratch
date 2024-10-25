from data.tokenizer import tokenize_file
from data.vocabulary import Vocabulary
from data.cooccurrence import CooccurrenceMatrix
from glove.model import GloVeModel
from glove.train import prepare_training_data, train
from eval.analogy import load_analogies, evaluate_analogies
import json

def run(corpus_paths, analogy_path, output_dir="outputs"):
    analogies = load_analogies(analogy_path)
    results = {}
    for name, path in corpus_paths:
        vocab = Vocabulary()
        vocab.count_from_file(path, tokenize_file)
        vocab.filter_top_n(400000)
        cooc = CooccurrenceMatrix(vocab)
        cooc.build(tokenize_file(path))
        rows, cols, vals = prepare_training_data(cooc)
        model = GloVeModel(len(vocab))
        train(model, rows, cols, vals)
        vectors = model.get_vectors(vocab)
        results[name] = evaluate_analogies(vectors, analogies)
    with open(f"{output_dir}/corpus_comparison.json", "w") as f:
        json.dump(results, f, default=str)
