# encoding=utf8
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec

filename = 'text8\\text8'

words = word2vec.Text8Corpus(filename)
model = Word2Vec()
model.build_vocab(words)
model.train(words, total_examples=model.corpus_count, epochs=model.iter)
print(model['class'])
print(model.most_similar(['class']))
