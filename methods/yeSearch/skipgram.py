import logging
import os

from pattern.text.en import tokenize
from time import time
from gensim.models.word2vec import Word2Vec
import multiprocessing
import methods.preprocess as preprocess

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class Code(object):
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
                    rline = preprocess.code2string(sline)
                    tokenized_line = ' '.join(tokenize(rline))
                    is_alpha_word_line = tokenized_line.lower().split()
                    yield is_alpha_word_line


def train(conf):
    path = conf.get('workdir')

    begin = time()
    codes = Code(path + conf.data_params['train_database'])

    if conf.data_params['reload'] == 0:
        model = Word2Vec(codes,size=100,window=10,min_count=20,negative=25,
                      sg = 1, iter = 10, workers=multiprocessing.cpu_count())
        model.build_vocab(codes, update=True)
    else:
        model = Word2Vec.load(path + conf.data_params['train_model'] + str(conf.data_params['reload']))

    for i in range(conf.model_params['iter']):
        model.train(codes, total_examples=model.corpus_count, epochs=conf.model_params['epochs'])
        model.save(path + conf.data_params['train_model'] +
                   str(conf.data_params['reload'] + (i + 1) * conf.model_params['epochs']))
    model.wv.save_word2vec_format(path + conf.data_params['train_word_wordembedding'],
                                  path + conf.data_params['vocabulary'],
                                  binary = False)
    conf.model_params.set('reload',conf.model_params['epochs']*conf.model_params['iter'])

    end = time()
    print("Total procesing time: %d seconds" % (end - begin))

