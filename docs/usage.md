# usage

## train

```
python -m glove.train --corpus data/corpus/wiki.txt --dim 300 --output outputs/vectors.txt
```

## evaluate

```
python -m eval.analogy --vectors outputs/vectors.txt --analogies data/analogies.txt
python -m eval.similarity --vectors outputs/vectors.txt
```

## run experiments

```
python -m experiment.runner --corpus data/corpus/wiki.txt --analogies data/analogies.txt
```

## baselines

```
python -m baseline.svd_baselines --corpus data/corpus/wiki.txt --dim 300
python -m baseline.word2vec --corpus data/corpus/wiki.txt --dim 300
```
