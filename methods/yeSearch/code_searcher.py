import json
import math
import os
import csv
import heapq
from gensim import corpora
from gensim.models import KeyedVectors
import time
import logging
from evaluate.metrics import firstPos, ACC, MRR, MAP
import methods.preprocess as preprocess

logger = logging.getLogger(__name__)

class CodeSearcher:
    def __init__(self, conf=None):
        self.conf = dict() if conf is None else conf
        self.path = self.conf.get('workdir', '../data/github/')
        self.data_params = conf.get('data_params', dict())
        self.w2v = None
        self.dict = None

    ### preprocess ###
    def clean_codebase(self):
        # clean json files from directory
        origin = self.path + self.data_params['origin_codebase_dir']
        clean_path = self.path + self.data_params['eval_codebase_dir']
        files = os.listdir(origin)
        json_files = []
        for f in files:
            json_files.append((origin + '/' + f, clean_path + '/' + f))
        for jpath in json_files:
            self.clean_json(jpath[0], jpath[1])

    def clean_json(self, origin_path, clean_path):
        with open(clean_path, "w") as clean_f:
            with open(origin_path, 'r') as load_f:
                clean_codes = []
                load_dict = json.load(load_f)
                for code in load_dict:
                    clean_codes.append(
                        {"id": code["id"],
                         "methname": preprocess.code2list(code['methname']),
                         "methbody": preprocess.code2list(code["methbody"])
                         }
                    )
                print("cleaning file ", origin_path)
                json.dump(clean_codes, clean_f)

    def generate_idf(self):
        codebase = self.load_codebase()
        voc_path = self.path + self.data_params['train_vocabulary']
        idf_path = self.path + self.data_params['eval_idf']
        dict = corpora.Dictionary([i[1] for i in codebase])
        with open(idf_path, 'w') as write_f:
            with open(voc_path, 'r') as load_f:
                vocs = load_f.readlines()
                for voc in vocs:
                    clean_codes = voc.split(" ")
                    cur = clean_codes[0]
                    if cur not in dict.token2id:
                        continue
                    write_f.writelines(clean_codes[0]+" "+str(self.calc_idf(clean_codes[0]))+"\n")

    def calc_idf(self,word):
        return math.log10((self.dict.num_docs + 1) / (1 + self.dict.dfs[self.dict.token2id[word]]))

    ### load Database ###
    def load_codebase(self):
        codebase_dir = self.path + self.data_params['eval_codebase_dir']
        # load json files from directory
        files = os.listdir(codebase_dir)
        json_files = []
        for f in files:
            if f[0] == '.' or os.path.isdir(codebase_dir + '/' + f) or '.json' not in f[-5:]:  # filter hiden file and directory
                continue
            json_files.append(codebase_dir + '/' + f)
        # open file and build codebase
        all_codes = []
        for jpath in json_files:
            all_codes.extend(self.load_json(jpath))
        codebase = [(word[0], preprocess.code2list(word[1])) for word in all_codes]
        print("codebase size:", len(codebase))
        return codebase

    def load_query(self):
        query_path = self.path + self.data_params['eval_query']
        with open(query_path, 'r')as query_f:
            query_list = json.load(query_f)
            for query in query_list:
                query['query'] = preprocess.code2list(raw=query['query'] )
            querys = query_list
        return querys

    def load_json(self, path):
        with open(path, 'r') as load_f:
            codes = []
            load_dict = json.load(load_f)
            for code in load_dict:
                codes.append((code["id"], code["methbody"]))
            print("loading file ", path)
            return codes

    def load_idf(self):
        path = self.path + self.data_params['eval_idf']
        idf = {}
        with open(path, 'r') as load_f:
            list = load_f.readlines()
            for voc in list:
                clean_codes = voc.split(" ")
                idf[clean_codes[0]] = float(clean_codes[1].replace("\n",''))
        return idf

    ### yesearch methods ###
    def get_idf_reloaded(self, word):
        if word not in self.idf:
            return 0
        else:
            return self.idf[word]

    def wwSimilarity(self, word1, word2):
        try:
            sim = self.w2v.wv.similarity(word1, word2)
            return sim
        except KeyError:
            return 0

    def wTSimilarity(self, word, codeList):
        if word in codeList:
            return 1
        sims = [self.wwSimilarity(word, i) for i in codeList]
        if not sims:
            return 0
        return max(sims)

    def TSSimilarity(self, qryList, codeList):
        cur_idf = [self.get_idf_reloaded(i) for i in qryList]
        s = sum(cur_idf)
        if s == 0:
            return 0
        ts = sum([(self.wTSimilarity(qryList[i], codeList) * cur_idf[i]) for i in range(len(cur_idf))]) / s
        return ts

    def getTotalSimilarity(self, rawQuery, doc):
        res = self.TSSimilarity(rawQuery, doc) + self.TSSimilarity(doc, rawQuery)
        return res

    def getTopNRank(self, n, query,  codebase=None):
        heap = []
        for code in codebase:
            if not code:continue
            score = self.getTotalSimilarity(query,code[1])
            item = (score,code[0])
            if len(heap)<n:
                heapq.heappush(heap,item)
            elif score>heap[0][0]:
                heapq.heapreplace(heap,item)
        return [i[1] for i in heap]

    def getTopNCode(self, n, query):
        return [i[1] for i in self.getTopNRank(n, query)]

    ### Evaluation ###
    def eval(self, topK):
        # eval result init
        output_log = open(self.path + self.data_params['eval_log'], 'w')
        csv_writer = csv.writer(output_log, dialect='excel')
        csv_writer.writerow(['id', 'acc', 'mrr', 'map', 'firstpos'])

        # load model and dataset
        codebase = self.load_codebase()
        querys = self.load_query()
        self.w2v = KeyedVectors.load_word2vec_format(self.path + self.data_params['train_word_wordembedding'],
                                                     binary=False)
        self.dict = corpora.Dictionary([i[1] for i in codebase] + querys)
        self.idf = self.load_idf()

        for i in range(len(querys)):
            query = querys[i]['query']
            real = querys[i]['answerList']

            time_start = time.time()
            predict = self.getTopNRank(topK, query, codebase)
            time_end = time.time()

            temp_pos = firstPos(real, predict)
            temp_acc = ACC(real, predict)
            temp_mrr = MRR(real, predict)
            temp_map = MAP(real, map)
            temp_time = int(time_end - time_start)
            csv_writer.writerow([i, temp_acc, temp_mrr, temp_map, temp_pos, temp_time])
            logger.info('ID={}, ACC={}, MRR={}, MAP={}, fpos={}, time={}'.format(i, temp_acc, temp_mrr, temp_map, temp_pos, temp_time))

