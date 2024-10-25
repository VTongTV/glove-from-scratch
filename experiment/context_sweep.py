import config
from data.tokenizer import tokenize_file
from data.vocabulary import Vocabulary
from data.cooccurrence import CooccurrenceMatrix
from glove.model import GloVeModel
from glove.train import prepare_training_data, train
from eval.analogy import load_analogies, evaluate_analogies
import json

def run(corpus_path, analogy_path, output_dir="outputs"):
    vocab = Vocabulary()
    vocab.count_from_file(corpus_path, tokenize_file)
    vocab.filter_top_n(config.VOCAB_SIZE)
    analogies = load_analogies(analogy_path)
    results = {}
    for window in [2, 4, 6, 8, 10]:
        for ctype in ["symmetric", "asymmetric"]:
            cooc = CooccurrenceMatrix(vocab, window_size=window, context_type=ctype)
            cooc.build(tokenize_file(corpus_path))
            rows, cols, vals = prepare_training_data(cooc)
            model = GloVeModel(len(vocab))
            train(model, rows, cols, vals)
            vectors = model.get_vectors(vocab)
            acc = evaluate_analogies(vectors, analogies)
            results[f"{ctype}_w{window}"] = acc
    with open(f"{output_dir}/context_sweep.json", "w") as f:
        json.dump(results, f, default=str)
