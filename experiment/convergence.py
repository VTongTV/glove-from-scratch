import json
import config
from data.tokenizer import tokenize_file
from data.vocabulary import Vocabulary
from data.cooccurrence import CooccurrenceMatrix
from glove.model import GloVeModel
from glove.train import prepare_training_data, train

def run(corpus_path, output_dir="outputs"):
    vocab = Vocabulary()
    vocab.count_from_file(corpus_path, tokenize_file)
    vocab.filter_top_n(config.VOCAB_SIZE)
    cooc = CooccurrenceMatrix(vocab)
    cooc.build(tokenize_file(corpus_path))
    rows, cols, vals = prepare_training_data(cooc)
    model = GloVeModel(len(vocab))
    loss_history = train(model, rows, cols, vals)
    with open(f"{output_dir}/convergence.json", "w") as f:
        json.dump({"loss": loss_history}, f)
