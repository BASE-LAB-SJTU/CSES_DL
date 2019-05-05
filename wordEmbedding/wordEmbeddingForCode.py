import json
import math
import os

from gensim import corpora
from gensim.models import KeyedVectors
import time
import logging

from evaluate_metrics import f_measure, firstPos, ACC, MRR, MAP, NDCG
from wordEmbedding.preprocessUtil import preprocessUtil

logger = logging.getLogger(__name__)


class WordEmbeddingForCode:
    def __init__(self, codebaseDir):
        self.preprocess = preprocessUtil()
        self.WORD_EMBEDDING = './data/model/word2vec'
        self.QUERY_SET = './data/eval/query2answer.json'
        self.CODEBASE_DIR = codebaseDir

        self.codebase = self.load_codebase()
        self.link = self.load_link()
        self.querys = self.load_query()[0]

        self.dict = corpora.Dictionary([i[1] for i in self.codebase] + self.querys)
        self.w2v = KeyedVectors.load_word2vec_format(self.WORD_EMBEDDING, binary=False)
    # ### Prepared doc & query for matching
    # def split_word(self,str):
    #     str = str.replace('_', ' ')
    #     return re.findall(r"\w+", str)

    def load_codebase(self):
        # load json files from directory
        files = os.listdir(self.CODEBASE_DIR)
        json_files = []
        for f in files:
            if f[0] == '.' or os.path.isdir(self.CODEBASE_DIR + '/' + f) or '.json' not in f[-5:]:  # filter hiden file and directory
                continue
            json_files.append(self.CODEBASE_DIR + '/' + f)
        # open file and build codebase
        all_codes = []
        for jpath in json_files:
            all_codes.extend(self.load_json(jpath))
        self.codebase = [(word[0], self.preprocess.codeCleanToList(word[1])) for word in all_codes]
        print("codebase size:", len(self.codebase))
        return self.codebase

    def load_query(self):
        querys, true_results = [], []
        with open(self.QUERY_SET, 'r')as query_f:
            query_list = json.load(query_f)
            for q in query_list:
                querys.append(q['query'])
                true_results.append(q['answerList'])
            self.querys = querys
        return querys, true_results

    def load_link(self):
        qrys, reals = self.load_query()
        self.link = dict()
        for i in range(len(qrys)):
            self.link[qrys[i]] = reals[i]
        return self.link

    def load_json(self, path):
        with open(path, 'r') as load_f:
            codes = []
            load_dict = json.load(load_f)
            for code in load_dict:
                codes.append((code["id"], code["methbody"]))
            print("loading file ", path)
            return codes

    def getIDF(self, word):
        return math.log10((self.dict.num_docs + 1) / (1 + self.dict.dfs[self.dict.token2id[word]]))

    def wwSimilarity(self, word1, word2):
        try:
            sim = self.w2v.wv.similarity(str.lower(word1), str.lower(word2))
            return sim
        except KeyError:
            return 0

    def wTSimilarity(self, word, codeList):
        res = max([self.wwSimilarity(word, i) for i in codeList])
        return res

    def TSSimilarity(self, qryList, codeList):
        try:
            ts = (sum([(self.wTSimilarity(i, codeList) * self.getIDF(i)) for i in qryList])) / (
                sum([self.getIDF(i) for i in qryList]))
            # print("ts", ts)
            return ts
        except ZeroDivisionError:
            return 0

    def getTotalSimilarity(self, rawQuery, doc):
        res = self.TSSimilarity(rawQuery, doc) + self.TSSimilarity(doc, rawQuery)
        return res

    def getTopNRank(self, n, query,  codebase=None):
        if codebase is None:
            codebase = self.codebase
        resultList = []
        for code in codebase:
            score = self.getTotalSimilarity(code[1], query)
            resultList.append((code[0], code[1], score))
        resultList = sorted(resultList, key=lambda x: x[2], reverse=True)
        return resultList[:n]

    def getTopNCode(self, n, query):
        return [i[1] for i in self.getTopNRank(n, query)]

    def eval(self, topK):
        time_start = time.time()
        acc, mrr, map, ndcg, fm, fpos = 0, 0, 0, 0, 0, 0
        data_len = len(self.querys)
        print("Total : " + str(data_len))
        for i in range(data_len):
            print(i)
            query = self.querys[i]
            real = self.link[query]
            predict = self.getTopNRank(topK, query, self.codebase)
            predict = [i[0] for i in predict]
            temp_f = f_measure(real, predict)
            temp_pos = firstPos(real, predict)
            temp_acc = ACC(real, predict)
            temp_mrr = MRR(real, predict)
            temp_map = MAP(real, map)
            temp_ndcg = NDCG(real, predict)
            acc += temp_acc
            mrr += temp_mrr
            map += temp_map
            ndcg += temp_ndcg
            fm += temp_f
            fpos += temp_pos
        acc = acc / float(data_len)
        mrr = mrr / float(data_len)
        map = map / float(data_len)
        ndcg = ndcg / float(data_len)
        fm = fm / float(data_len)
        fpos = fpos / float(data_len)
        logger.info('ACC={}, MRR={}, MAP={}, nDCG={}, fpos={}, f-measure={}'.format(acc, mrr, map, ndcg, fpos, fm))
        time_end = time.time()
        cost_s = int(time_end - time_start)
        cost_m = int(cost_s / 60)
        cost_h = int(cost_m / 60)
        print('totally cost :', cost_h, 'h', cost_m % 60, 'min', cost_s % 60, 's')
        return acc, mrr, map, ndcg, fm, fpos
