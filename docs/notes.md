# implementation notes

## differences from the original c implementation

- tokenizer uses regex instead of the stanford tokenizer. minor differences in token boundaries.
- adagrad epsilon is 1e-8. the original may use a different value.
- co-occurrence matrix stored as dict of dicts instead of custom binary format.
- no parallel training. the original uses openmp with 32 threads.
- convergence check added. the original runs a fixed number of iterations.

## paper benchmarks (glove 300d, 6b tokens)

| metric | paper |
|--------|-------|
| semantic | 77.4% |
| syntactic | 67.0% |
| total | 71.7% |
| ws353 | 65.8 |
| mc | 72.7 |
| rg | 77.8 |
| scws | 53.9 |
| rw | 38.1 |

## known limitations

- single-threaded. training on large corpora will be slow.
- no memory-mapped co-occurrence matrix. large vocabularies may not fit in ram.
- the analogy dataset parser uses simple heuristics to classify semantic vs syntactic categories.
