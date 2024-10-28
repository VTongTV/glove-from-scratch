import argparse, json, os
import config
from data.tokenizer import tokenize_file
from data.vocabulary import Vocabulary
from data.cooccurrence import CooccurrenceMatrix
from glove.model import GloVeModel
from glove.train import prepare_training_data, train
from glove.io import save_vectors
from eval.analogy import load_analogies, evaluate_analogies
from eval.similarity import load_ws353, load_mc, load_rg, load_scws, load_rw
from eval.metrics import evaluate_similarity
from experiment.dim_sweep import run as dim_sweep
from experiment.context_sweep import run as context_sweep
from experiment.convergence import run as convergence
from experiment.comparison import run as comparison

def run(args):
    os.makedirs(args.output, exist_ok=True)
    if "dim" in args.experiments:
        dim_sweep(args.corpus, args.analogies, args.output)
    if "context" in args.experiments:
        context_sweep(args.corpus, args.analogies, args.output)
    if "convergence" in args.experiments:
        convergence(args.corpus, args.output)
    if "comparison" in args.experiments:
        comparison(args.corpus, args.analogies, args.output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--analogies", required=True)
    p.add_argument("--output", default="outputs")
    p.add_argument("--experiments", nargs="+", default=["dim", "context", "convergence", "comparison"])
    run(p.parse_args())
