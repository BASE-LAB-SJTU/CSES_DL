import codecs
import csv
import gc
import json
import pickle
import random
import threading
import time
import traceback
import os
import numpy as np
import logging
import tables
from scipy.stats import rankdata
from evaluate import metrics

from methods.deepcs.utils import normalize, cos_np_for_normalized
from methods.preprocess import query2list
random.seed(42)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

class CodeSearcher:
    def __init__(self, conf=None):
        self.conf = dict() if conf is None else conf
        self.path = self.conf.get('workdir', '../data/github/')
        self.train_params = conf.get('training_params', dict())
        self.data_params = conf.get('data_params', dict())
        self.model_params = conf.get('model_params', dict())

        self.vocab_methname = self.load_pickle(self.path + self.data_params['vocab_methname'])
        self.vocab_apiseq = self.load_pickle(self.path + self.data_params['vocab_apiseq'])
        self.vocab_tokens = self.load_pickle(self.path + self.data_params['vocab_tokens'])
        self.vocab_desc = self.load_pickle(self.path + self.data_params['vocab_desc'])

        self._eval_sets = None
        self.codebase_id = None
        self._code_reprs = None
        self._code_base = None
        self._code_base_chunksize = 2000000

    def load_pickle(self, filename):
        return pickle.load(open(filename, 'rb'))

    ##### Data Set #####
    def load_training_data_chunk(self, offset, chunk_size):
        logger.debug('Loading a chunk of training data..')
        logger.debug('methname')
        chunk_methnames = self.load_hdf5(self.path + self.data_params['train_methname'], offset, chunk_size)
        logger.debug('apiseq')
        chunk_apiseqs = self.load_hdf5(self.path + self.data_params['train_apiseq'], offset, chunk_size)
        logger.debug('tokens')
        chunk_tokens = self.load_hdf5(self.path + self.data_params['train_tokens'], offset, chunk_size)
        logger.debug('desc')
        chunk_descs = self.load_hdf5(self.path + self.data_params['train_desc'], offset, chunk_size)
        return chunk_methnames, chunk_apiseqs, chunk_tokens, chunk_descs

    def load_valid_data_chunk(self, chunk_size):
        logger.debug('Loading a chunk of validation data..')
        logger.debug('methname')
        chunk_methnames = self.load_hdf5(self.path + self.data_params['valid_methname'], 0, chunk_size)
        logger.debug('apiseq')
        chunk_apiseqs = self.load_hdf5(self.path + self.data_params['valid_apiseq'], 0, chunk_size)
        logger.debug('tokens')
        chunk_tokens = self.load_hdf5(self.path + self.data_params['valid_tokens'], 0, chunk_size)
        logger.debug('desc')
        chunk_descs = self.load_hdf5(self.path + self.data_params['valid_desc'], 0, chunk_size)
        return chunk_methnames, chunk_apiseqs, chunk_tokens, chunk_descs

    def load_use_data(self):
        logger.info('Loading use data..')
        logger.info('methname')
        methnames = self.load_hdf5(self.path + self.data_params['use_methname'], 0, -1)
        logger.info('apiseq')
        apiseqs = self.load_hdf5(self.path + self.data_params['use_apiseq'], 0, -1)
        logger.info('tokens')
        tokens = self.load_hdf5(self.path + self.data_params['use_tokens'], 0, -1)
        return methnames, apiseqs, tokens

    def load_use_codebase(self):
        """load codebase
        codefile: h5 file that stores raw code
        """
        logger.info('Loading codebase (chunk size={})..'.format(self._code_base_chunksize))
        if self._code_base==None:
            codebase=[]
            #codes=codecs.open(self.path+self.data_params['use_codebase']).readlines()
            codes=codecs.open(self.path+self.data_params['use_codebase'],encoding='utf8',errors='replace').readlines()
                #use codecs to read in case of encoding problem
            for i in range(0,len(codes),self._code_base_chunksize):
                codebase.append(codes[i:i+self._code_base_chunksize])
            self._code_base=codebase

    ##### Evaluation Data Set #####
    def load_MAT_json_dir(self, poolsize):
        logger.info("get json file name list from directory...")
        fileList = []
        dir = self.path + self.data_params['eval_codebase_json_dir']
        files = os.listdir(dir)
        for f in files:
            if f[0] == '.' or os.path.isdir(dir + '/' + f):  # filter hiden file and directory
                continue
            elif ".json" not in f[-5:]:  # filter json file
                continue
            fileList.append(f)
        logger.info("loading json...")
        apiseq, methname, tks, ids = [], [], [], []
        for fname in fileList:
            with open(dir + fname, 'r') as load_f:
                load_dict = json.load(load_f)
                for code in load_dict:
                    apiseq.append(code["apiseq"])
                    methname.append(code["methname"])
                    tks.append(code["tokens"])
                    ids.append(code["id"])
                print(fname, " three size:", len(apiseq))
        if poolsize < len(apiseq) and poolsize!=-1:
            methname = methname[:poolsize]
            apiseq = methname[:poolsize]
            tks = tks[:poolsize]
            ids = ids[:poolsize]
        print("total size:", len(apiseq))
        return methname, apiseq, tks, ids

    def load_eval_codebase(self, poolsize):
        logger.debug('Loading a chunk of validation data..')
        chunk_methnames, chunk_apiseqs, chunk_tokens, ids = self.load_MAT_json_dir(poolsize)
        chunk_tokens=[self.convert(self.vocab_tokens, i) for i in chunk_tokens]
        chunk_apiseqs=[self.convert(self.vocab_apiseq, i) for i in chunk_apiseqs]
        chunk_methnames=[self.convert(self.vocab_methname, i) for i in chunk_methnames]
        return chunk_methnames, chunk_apiseqs, chunk_tokens, ids

    ### save evaluation codebase in vector representations by iteration to avoid memory overflow ###
    def save_eval_codebase_vector_iter(self, model):
        logger.info("get json file name list from directory...")
        fileList = []
        json_dir = self.data_params['eval_codebase_json']
        files = os.listdir(json_dir)

        for f in files:
            if f[0] == '.' or os.path.isdir(json_dir + '/' + f):  # filter hidden file and directory
                continue
            elif ".json" not in f[-5:]:  # filter json file
                continue
            fileList.append(f)

        logger.info("save evaluation codebase in vector representations by iteration...")
        ids = []
        fvec = tables.open_file(self.path + self.data_params['eval_codevecs'], 'w')
        filters = tables.Filters(complib='blosc', complevel=5)

        opend = True
        for fname in fileList:
            apiseqs, methnames, tokens = [], [], [],
            with open(json_dir + fname, 'r') as load_f:
                load_dict = json.load(load_f)
                for code in load_dict:
                    apiseqs.append(code["apiseq"])
                    methnames.append(code["methname"])
                    tokens.append(code["tokens"])
                    ids.append(code["id"])
                tokens = [self.convert(self.vocab_tokens, i) for i in tokens]
                apiseqs = [self.convert(self.vocab_apiseq, i) for i in apiseqs]
                methnames = [self.convert(self.vocab_methname, i) for i in methnames]
                methnames = self.pad(methnames, self.data_params['methname_len'])
                apiseqs = self.pad(apiseqs, self.data_params['apiseq_len'])
                tokens = self.pad(tokens, self.data_params['tokens_len'])
                vecs = model.repr_code([methnames, apiseqs, tokens], batch_size=1000)
                del methnames, apiseqs, tokens
                gc.collect()
                vecs = vecs.astype('float32')
                vecs = normalize(vecs)
                npvecs = np.array(vecs)
                if opend:
                    opend = False
                    ds = fvec.create_earray(fvec.root, 'vecs', tables.Atom.from_dtype(npvecs.dtype), (0, 400),
                                            filters=filters)
                ds.append(vecs)
        fvec.close()
        with open(self.path + self.data_params['eval_codebase_id'], 'w') as write_f:
            for id in ids:
                write_f.writelines(id + " ")

    ### save evaluation codebase in vector representations ###
    def save_eval_codebase_vector(self, model, poolsize):
        logger.info('save eval codebase vector..')

        methnames, apiseqs, tokens, ids = self.load_eval_codebase(poolsize)
        methnames = self.pad(methnames, self.data_params['methname_len'])
        apiseqs = self.pad(apiseqs, self.data_params['apiseq_len'])
        tokens = self.pad(tokens, self.data_params['tokens_len'])
        vecs = model.repr_code([methnames, apiseqs, tokens], batch_size=1000)
        del methnames, apiseqs, tokens
        gc.collect()
        vecs = vecs.astype('float32')
        vecs = normalize(vecs)
        self.save_code_reprs(vecs,'eval_codevecs')
        with open(self.path + self.data_params['eval_codebase_id'], 'w') as write_f:
            for id in ids:
                write_f.writelines(id + " ")

    def load_eval_codebase_vector(self):
        logger.debug('Loading code vectors (chunk size={})..'.format(self._code_base_chunksize))
        if self._code_reprs==None:
            """reads vectors (2D numpy array) from a hdf5 file"""
            codereprs=[]
            h5f = tables.open_file(self.path + self.data_params['eval_codevecs'])
            vecs= h5f.root.vecs
            for i in range(0,len(vecs),self._code_base_chunksize):
                codereprs.append(vecs[i:i+self._code_base_chunksize])
            h5f.close()
            self._code_reprs=codereprs
        return self._code_reprs

    def load_eval_query_from_json(self):
        """load test query and true result ids from json file"""
        querys, true_results = [], []
        with open(self.path + self.data_params['eval_query'], 'r')as query_f:
            query_list = json.load(query_f)
            for q in query_list:
                querys.append(q['query'])
                true_results.append(q['answerList'])
        return querys, true_results

    def load_eval_query(self):
        chunk_descs, true_results = self.load_eval_query_from_json()
        return  chunk_descs,true_results

    def load_eval_codebase_id(self):
        res = []
        with open(self.path + self.data_params['eval_codebase_id'], 'r') as load_f:
            vecs = load_f.read().split(" ")
            for i in range(0, len(vecs), self._code_base_chunksize):
                res.append(vecs[i:i + self._code_base_chunksize])
            self.codebase_id = res

    ##### Compute Representation #####
    def repr_code(self, model):
        methnames, apiseqs, tokens = self.load_use_data()
        methnames = self.pad(methnames, self.data_params['methname_len'])
        apiseqs = self.pad(apiseqs, self.data_params['apiseq_len'])
        tokens = self.pad(tokens, self.data_params['tokens_len'])

        vecs = model.repr_code([methnames, apiseqs, tokens], batch_size=1000)
        del methnames, apiseqs, tokens
        gc.collect()
        vecs = vecs.astype('float32')
        vecs = normalize(vecs)
        self.save_code_reprs(vecs)
        return vecs

    def save_code_reprs(self, vecs, phrase='use_codevecs'):
        npvecs = np.array(vecs)
        fvec = tables.open_file(self.path + self.data_params[phrase], 'w')
        atom = tables.Atom.from_dtype(npvecs.dtype)
        filters = tables.Filters(complib='blosc', complevel=5)
        ds = fvec.create_carray(fvec.root, 'vecs', atom, npvecs.shape, filters=filters)
        ds[:] = npvecs
        fvec.close()

    def load_code_reprs(self):
        logger.debug('Loading code vectors (chunk size={})..'.format(self._code_base_chunksize))
        if self._code_reprs is None:
            """reads vectors (2D numpy array) from a hdf5 file"""
            codereprs = []
            h5f = tables.open_file(self.path + self.data_params['use_codevecs'])
            vecs = h5f.root.vecs
            for i in range(0, len(vecs), self._code_base_chunksize):
                codereprs.append(vecs[i:i + self._code_base_chunksize])
            h5f.close()
            self._code_reprs = codereprs
        return self._code_reprs

    def load_hdf5(self, vecfile, start_offset, chunk_size):
        """reads training sentences(list of int array) from a hdf5 file"""
        table = tables.open_file(vecfile)
        data, index = (table.get_node('/phrases'), table.get_node('/indices'))
        data_len = index.shape[0]
        if chunk_size == -1:  # if chunk_size is set to -1, then, load all data
            chunk_size = data_len
        start_offset = start_offset % data_len
        offset = start_offset
        logger.debug("{} entries".format(data_len))
        logger.debug("starting from offset {} to {}".format(start_offset, start_offset + chunk_size))
        sents = []
        while offset < start_offset + chunk_size:
            if offset >= data_len:
                logger.warn('Warning: offset exceeds data length, starting from index 0..')
                chunk_size = start_offset + chunk_size - data_len
                start_offset = 0
                offset = 0
            len, pos = index[offset]['length'], index[offset]['pos']
            offset += 1
            sents.append(data[pos:pos + len].astype('int32'))
        table.close()
        return sents

    ##### Converting / reverting #####
    def convert(self, vocab, words):
        """convert words into indices"""
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [vocab.get(w, 0) for w in words]

    def revert(self, vocab, indices):
        """revert indices into words"""
        ivocab = dict((v, k) for k, v in vocab.items())
        return [ivocab.get(i, 'UNK') for i in indices]

    ##### Padding #####
    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Model Loading / saving #####
    def save_model_epoch(self, model, epoch):
        if not os.path.exists(self.path + 'models/' + self.model_params['model_name'] + '/'):
            os.makedirs(self.path + 'models/' + self.model_params['model_name'] + '/')
        model.save("{}models/{}/epo{:d}_code.h5".format(self.path, self.model_params['model_name'], epoch),
                   "{}models/{}/epo{:d}_desc.h5".format(self.path, self.model_params['model_name'], epoch),
                   overwrite=True)

    def load_model_epoch(self, model, epoch):
        assert os.path.exists(
            "{}models/{}/epo{:d}_code.h5".format(self.path, self.model_params['model_name'], epoch)) \
            , "Weights at epoch {:d} not found".format(epoch)
        assert os.path.exists(
            "{}models/{}/epo{:d}_desc.h5".format(self.path, self.model_params['model_name'], epoch)) \
            , "Weights at epoch {:d} not found".format(epoch)
        model.load("{}models/{}/epo{:d}_code.h5".format(self.path, self.model_params['model_name'], epoch),
                   "{}models/{}/epo{:d}_desc.h5".format(self.path, self.model_params['model_name'], epoch))

    ##### Training #####
    def train(self, model):
        if self.train_params['reload'] > 0:
            self.load_model_epoch(model, self.train_params['reload'])
        valid_every = self.train_params.get('valid_every', None)
        save_every = self.train_params.get('save_every', None)
        batch_size = self.train_params.get('batch_size', 128)
        nb_epoch = self.train_params.get('nb_epoch', 10)
        split = self.train_params.get('validation_split', 0)

        val_loss = {'loss': 1., 'epoch': 0}

        for i in range(self.train_params['reload'] + 1, nb_epoch):
            print('Epoch %d :: \n' % i, end='')
            logger.debug('loading data chunk..')
            chunk_methnames, chunk_apiseqs, chunk_tokens, chunk_descs = \
                self.load_training_data_chunk( \
                    (i - 1) * self.train_params.get('chunk_size', 100000), \
                    self.train_params.get('chunk_size', 100000))
            logger.debug('padding data..')
            chunk_padded_methnames = self.pad(chunk_methnames, self.data_params['methname_len'])
            chunk_padded_apiseqs = self.pad(chunk_apiseqs, self.data_params['apiseq_len'])
            chunk_padded_tokens = self.pad(chunk_tokens, self.data_params['tokens_len'])
            chunk_padded_good_descs = self.pad(chunk_descs, self.data_params['desc_len'])
            chunk_bad_descs = [desc for desc in chunk_descs]
            random.shuffle(chunk_bad_descs)
            chunk_padded_bad_descs = self.pad(chunk_bad_descs, self.data_params['desc_len'])

            hist = model.fit(
                [chunk_padded_methnames, chunk_padded_apiseqs, chunk_padded_tokens, chunk_padded_good_descs,
                 chunk_padded_bad_descs], epochs=1, batch_size=batch_size, validation_split=split)

            if hist.history['val_loss'][0] < val_loss['loss']:
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))

            if valid_every is not None and i % valid_every == 0:
                acc1, mrr = self.valid(model, 1000, 1)
                # acc,mrr,map,ndcg=self.eval(model, 1000, 1)

            if save_every is not None and i % save_every == 0:
                self.save_model_epoch(model, i)

    def valid(self, model, poolsize, K):
        """
        quick validation in a code pool.
        param:
            poolsize - size of the code pool, if -1, load the whole test set
        """
        # load test dataset
        if self._eval_sets is None:
            # self._eval_sets = dict([(s, self.load(s)) for s in ['dev', 'test1', 'test2']])
            methnames, apiseqs, tokens, descs = self.load_valid_data_chunk(poolsize)
            self._eval_sets = dict()
            self._eval_sets['methnames'] = methnames
            self._eval_sets['apiseqs'] = apiseqs
            self._eval_sets['tokens'] = tokens
            self._eval_sets['descs'] = descs

        c_1, c_2 = 0, 0
        data_len = len(self._eval_sets['descs'])
        for i in range(data_len):
            bad_descs = [desc for desc in self._eval_sets['descs']]
            random.shuffle(bad_descs)
            descs = bad_descs
            descs[0] = self._eval_sets['descs'][i]  # good desc
            descs = self.pad(descs, self.data_params['desc_len'])
            methnames = self.pad([self._eval_sets['methnames'][i]] * data_len, self.data_params['methname_len'])
            apiseqs = self.pad([self._eval_sets['apiseqs'][i]] * data_len, self.data_params['apiseq_len'])
            tokens = self.pad([self._eval_sets['tokens'][i]] * data_len, self.data_params['tokens_len'])
            n_good = K

            sims = model.predict([methnames, apiseqs, tokens, descs], batch_size=data_len).flatten()
            r = rankdata(sims, method='max')
            max_r = np.argmax(r)
            max_n = np.argmax(r[:n_good])
            c_1 += 1 if max_r == max_n else 0
            c_2 += 1 / float(r[max_r] - r[max_n] + 1)

        top1 = c_1 / float(data_len)
        # percentage of predicted most similar desc that is really the corresponding desc
        mrr = c_2 / float(data_len)
        logger.info('Top-1 Precision={}, MRR={}'.format(top1, mrr))

        return top1, mrr

    ##### Evaluation in the develop set #####
    def eval(self, model, poolsize, K):
        """
        validate in a code pool.
        param:
            poolsize - size of the code pool, if -1, load the whole test set
        """
        # load valid dataset
        if self._eval_sets is None:
            methnames, apiseqs, tokens, descs = self.load_valid_data_chunk(poolsize)
            self._eval_sets = dict()
            self._eval_sets['methnames'] = methnames
            self._eval_sets['apiseqs'] = apiseqs
            self._eval_sets['tokens'] = tokens
            self._eval_sets['descs'] = descs
        acc, mrr, map, ndcg = 0, 0, 0, 0
        data_len = len(self._eval_sets['descs'])
        print("Total : " + str(data_len))
        for i in range(data_len):
            print(i)
            desc = self._eval_sets['descs'][i]  # good desc
            descs = self.pad([desc] * data_len, self.data_params['desc_len'])
            methnames = self.pad(self._eval_sets['methnames'], self.data_params['methname_len'])
            apiseqs = self.pad(self._eval_sets['apiseqs'], self.data_params['apiseq_len'])
            tokens = self.pad(self._eval_sets['tokens'], self.data_params['tokens_len'])
            n_results = K
            sims = model.predict([methnames, apiseqs, tokens, descs], batch_size=data_len).flatten()
            negsims = np.negative(sims)
            predict = np.argsort(negsims)  # predict = np.argpartition(negsims, kth=n_results-1)
            predict = predict[:n_results]
            predict = [int(k) for k in predict]
            real = [i]
            acc += metrics.ACC(real, predict)
            mrr += metrics.MRR(real, predict)
            map += metrics.MAP(real, predict)
            ndcg += metrics.NDCG(real, predict)
        acc = acc / float(data_len)
        mrr = mrr / float(data_len)
        map = map / float(data_len)
        ndcg = ndcg / float(data_len)
        logger.info('ACC={}, MRR={}, MAP={}, nDCG={}'.format(acc, mrr, map, ndcg))

        return acc, mrr, map, ndcg

    ##### Evaluation with multithreading and codebase vector representation #####
    def faster_eval(self, model, n_results):
        # eval result init
        output_log = open(self.path + self.data_params['eval_log'], 'w')
        csv_writer = csv.writer(output_log, dialect='excel')
        csv_writer.writerow(['id', 'acc', 'mrr', 'map', 'firstpos'])

        # load valid dataset
        self.load_code_reprs()
        descs, true_results = self.load_eval_query()
        self.load_eval_codebase_id()
        data_len = len(descs)

        for i in range(data_len):
            real = true_results[i]
            desc = query2list(descs[i])
            desc = [self.convert(self.vocab_desc, desc)]  # convert desc sentence to word indices

            time_start = time.time()
            padded_desc = self.pad(desc, self.data_params['desc_len'])
            desc_repr = model.repr_desc([padded_desc])
            desc_repr = desc_repr.astype('float32')
            threads = []
            predict = []
            for i, code_reprs_chunk in enumerate(self._code_reprs):
                t = threading.Thread(target=self.eval_thread,
                                     args=(predict,desc_repr, code_reprs_chunk, i, n_results))
                threads.append(t)
                t.start()
            for t in threads:  # wait until all sub-threads finish
                t.join()
            time_end = time.time()

            predict.sort(reverse=True,key=lambda x: x[0])
            predict = predict[:n_results]
            predict = [i[1] for i in predict]
            temp_pos = metrics.firstPos(real, predict)
            predict = predict[:n_results]
            temp_acc = metrics.ACC(real, predict)
            temp_mrr = metrics.MRR(real, predict)
            temp_map = metrics.MAP(real, predict)
            temp_time = int(time_end - time_start)
            csv_writer.writerow([i, temp_acc, temp_mrr, temp_map, temp_pos, temp_time])
            logger.info('ID={}, ACC={}, MRR={}, MAP={}, fpos={}, time={}'.format(i, temp_acc, temp_mrr, temp_map, temp_pos, temp_time))

    def eval_thread(self,sims,desc_repr,code_reprs,i,n_results):
        # 1. compute similarity
        chunk_sims = cos_np_for_normalized(normalize(desc_repr), code_reprs)

        # 2. choose top results
        negsims = np.negative(chunk_sims[0])
        maxinds = np.argpartition(negsims, kth=n_results - 1)
        predict = maxinds[:n_results]

        try:
            sims.extend([(chunk_sims[0][predict],self.codebase_id[i][predict]) for predict in predict])
        except Exception:
            print("Exception while searching top entries:")
            traceback.print_exc()

    def search(self, model, query, n_results=10):
        desc = [self.convert(self.vocab_desc, query)]  # convert desc sentence to word indices
        padded_desc = self.pad(desc, self.data_params['desc_len'])
        desc_repr = model.repr_desc([padded_desc])
        desc_repr = desc_repr.astype('float32')

        codes = []
        sims = []
        threads = []
        for i, code_reprs_chunk in enumerate(self._code_reprs):
            t = threading.Thread(target=self.search_thread,
                                 args=(codes, sims, desc_repr, code_reprs_chunk, i, n_results))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:  # wait until all sub-threads finish
            t.join()
        return codes, sims

    def search_thread(self, codes, sims, desc_repr, code_reprs, i, n_results):
        # 1. compute similarity
        chunk_sims = cos_np_for_normalized(normalize(desc_repr), code_reprs)

        # 2. choose top results
        negsims = np.negative(chunk_sims[0])
        maxinds = np.argpartition(negsims, kth=n_results - 1)
        maxinds = maxinds[:n_results]

        print("i : " + str(i) + '\n')
        print("codebase size: " + str(len(self._code_base)))
        try:
            print("things in maxinds: " + str(maxinds))
            print("codebase size: " + str(len(self._code_base[i])))
            chunk_codes = [self._code_base[i][k] for k in maxinds]
            chunk_sims = chunk_sims[0][maxinds]
            codes.extend(chunk_codes)
            sims.extend(chunk_sims)
        except Exception:
            print("Exception while searching top entries:")
            traceback.print_exc()

    def postproc(self, codes_sims):
        codes_, sims_ = zip(*codes_sims)
        codes = [code for code in codes_]
        sims = [sim for sim in sims_]
        final_codes = []
        final_sims = []
        n = len(codes_sims)
        for i in range(n):
            is_dup = False
            for j in range(i):
                if codes[i][:80] == codes[j][:80] and abs(sims[i] - sims[j]) < 0.01:
                    is_dup = True
            if not is_dup:
                final_codes.append(codes[i])
                final_sims.append(sims[i])
        return zip(final_codes, final_sims)
