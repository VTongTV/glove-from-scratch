# GloVe: Global Vectors for Word Representation

from-scratch numpy implementation of GloVe (Pennington, Socher & Manning, EMNLP 2014).

## usage

train glove on a tokenized corpus:
```
python -m glove.train --corpus data/corpus/wiki.txt --dim 300 --output outputs/vectors.txt
```

run baselines (svd, word2vec):
```
python -m baseline.svd_baselines --corpus data/corpus/wiki.txt --dim 300
python -m baseline.word2vec --corpus data/corpus/wiki.txt --dim 300
```

evaluate on word analogy and similarity tasks:
```
python -m eval.analogy --vectors outputs/vectors.txt --analogies data/analogies.txt
python -m eval.similarity --vectors outputs/vectors.txt
```

## structure

| directory | purpose |
|-----------|---------|
| `data/` | corpus download, tokenization, vocabulary, co-occurrence |
| `glove/` | model, weighting, adagrad, training, io |
| `baseline/` | svd baselines, word2vec wrappers |
| `eval/` | analogy evaluation, similarity evaluation, metrics |
| `experiment/` | dimension sweep, context sweep, corpus comparison |
| `viz/` | plots and visualizations |
| `tests/` | unit tests |

## reference

Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. EMNLP 2014.
