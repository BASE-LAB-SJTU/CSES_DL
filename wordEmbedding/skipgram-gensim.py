import logging
import os

from pattern.text.en import tokenize
from time import time
from gensim.models.word2vec import Word2Vec
import multiprocessing
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
from preprocessUtil import preprocessUtil
preprocessUtil = preprocessUtil()

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for filename in files:
                file_path = root + '/' + filename
                for line in open(file_path):
                    sline = line.strip()
                    if sline == "":
                        continue
                    rline = preprocessUtil.cleanToString(sline)
                    tokenized_line = ' '.join(tokenize(rline))
                    is_alpha_word_line = tokenized_line.lower().split()
                    yield is_alpha_word_line


def train():
    begin = time()
    data_path = "data/train"
    sentences = MySentences(data_path)
    model = Word2Vec(sentences,size=100,window=10,min_count=20,negative=25,
                      sg = 1, iter = 10, workers=multiprocessing.cpu_count())
    # model = Word2Vec.load("data/model/word2vec_model_Epoch_50")
    # model.build_vocab(sentences, update=True)
    # for i in range(5):
    #     iter = str(i * 3)
    #     model.train(sentences, total_examples=model.corpus_count, epochs=3)
    #     model.save("data/model/word2vec_model_Epoch_" + iter)
    model.wv.save_word2vec_format("data/model/word2vec",
                                  "data/model/vocabulary",
                                  binary=False)

    # vector = model.wv['computer']
    # sim = model.similar_by_word('report',topn=100)
    # print(vector)
    # model.train([["hello", "world"]], total_examples=1, epochs=1)
    end = time()
    print("Total procesing time: %d seconds" % (end - begin))

if __name__ == '__main__':
    train()