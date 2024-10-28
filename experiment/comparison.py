import json, time, config
from data.tokenizer import tokenize_file
from data.vocabulary import Vocabulary
from data.cooccurrence import CooccurrenceMatrix
from glove.model import GloVeModel
from glove.train import prepare_training_data, train
from baseline.word2vec import train_skipgram, train_cbow
from eval.analogy import load_analogies, evaluate_analogies

def run(corpus_path, analogy_path, output_dir="outputs"):
    vocab = Vocabulary()
    vocab.count_from_file(corpus_path, tokenize_file)
    vocab.filter_top_n(config.VOCAB_SIZE)
    cooc = CooccurrenceMatrix(vocab)
    cooc.build(tokenize_file(corpus_path))
    rows, cols, vals = prepare_training_data(cooc)
    analogies = load_analogies(analogy_path)
    results = {}

    t0 = time.time()
    model = GloVeModel(len(vocab))
    loss = train(model, rows, cols, vals)
    glove_time = time.time() - t0
    vectors = model.get_vectors(vocab)
    glove_acc = evaluate_analogies(vectors, analogies)
    results["glove"] = {"time": glove_time, "accuracy": glove_acc}

    sentences = list(tokenize_file(corpus_path))
    t0 = time.time()
    sg_vecs = train_skipgram(sentences)
    sg_time = time.time() - t0
    sg_acc = evaluate_analogies(sg_vecs, analogies)
    results["skipgram"] = {"time": sg_time, "accuracy": sg_acc}

    t0 = time.time()
    cb_vecs = train_cbow(sentences)
    cb_time = time.time() - t0
    cb_acc = evaluate_analogies(cb_vecs, analogies)
    results["cbow"] = {"time": cb_time, "accuracy": cb_acc}

    with open(f"{output_dir}/comparison.json", "w") as f:
        json.dump(results, f, default=str)
