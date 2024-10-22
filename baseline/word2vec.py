import logging
import config

logger = logging.getLogger(__name__)

def train_skipgram(sentences, dim=config.EMBEDDING_DIM, window=config.WINDOW_SIZE, negative=10):
    from gensim.models import Word2Vec
    model = Word2Vec(sentences=sentences, vector_size=dim, window=window, min_count=1, sg=1, negative=negative, workers=4)
    return {word: model.wv[word] for word in model.wv.index_to_key}

def train_cbow(sentences, dim=config.EMBEDDING_DIM, window=config.WINDOW_SIZE, negative=10):
    from gensim.models import Word2Vec
    model = Word2Vec(sentences=sentences, vector_size=dim, window=window, min_count=1, sg=0, negative=negative, workers=4)
    return {word: model.wv[word] for word in model.wv.index_to_key}
