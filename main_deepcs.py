from __future__ import print_function

from methods.deepcs.code_searcher import CodeSearcher
from methods.deepcs.models import JointEmbeddingModel
import traceback
import argparse
from methods.deepcs import configs

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True   # assign with requirement
sess = tf.Session(config=config)
KTF.set_session(sess)

def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument("--proto", choices=["get_config"],  default="get_config",
                        help="Prototype config to use for config")
    parser.add_argument("--mode", choices=["train","eval","repr_code","search", "fast_eval", "eval_javadoc", "search_once"], default='train',
                        help="The mode to run. The `train` mode trains a model;"
                        " the `eval` mode evaluat models in a test set "
                        " The `repr_code/repr_desc` mode computes vectors"
                        " for a code snippet or a natural language description with a trained model."
                        " the `fast_eval` mode evaluat models in a test set with a fast mode"
                        " the 'search_once' mode search need one input query and return candidate results")
    parser.add_argument("--path")
    parser.add_argument("--query")
    parser.add_argument("--n")
    parser.add_argument("--verbose",action="store_true", default=True, help="Be verbose")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    conf = getattr(configs, args.proto)()
    codesearcher = CodeSearcher(conf)

    ##### Define model ######
    model = eval(conf['model_params']['model_name'])(conf)#initialize the model
    model.build()
    optimizer = conf.get('training_params', dict()).get('optimizer', 'adam')
    model.compile(optimizer=optimizer)
    
    if args.mode=='train':  
        codesearcher.train(model)
        
    elif args.mode=='eval':
        # evaluate for a particular epoch
        #load model
        if conf['training_params']['reload']>0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
        codesearcher.eval(model,-1,10)

    elif args.mode=='faster_eval':
        # evaluate for a particular epoch
        #load model
        if conf['training_params']['reload']>0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
        codesearcher.faster_eval(model,10)

    elif args.mode=='repr_code':
        #load model
        if conf['training_params']['reload']>0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
        vecs=codesearcher.repr_code(model)
        
    elif args.mode=='search':
        #search code based on a desc
        if conf['training_params']['reload']>0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
        codesearcher.load_code_reprs()
        codesearcher.load_use_codebase()
        while True:
            try:
                query = input('Input Query: ')
                n_results = int(input('How many results? '))
            except Exception:
                print("Exception while parsing your input:")
                traceback.print_exc()
                break
            codes,sims=codesearcher.search(model, query, n_results)
            zipped=zip(codes,sims)
            zipped=sorted(zipped, reverse=True, key=lambda x:x[1])
            zipped=codesearcher.postproc(zipped)
            zipped = list(zipped)[:n_results]
            results = '\n\n'.join(map(str,zipped)) #combine the result into a returning string
            print(results)

    elif args.mode=='save_codebase':
        codesearcher.save_eval_codebase_vector_iter(model)

    elif args.mode=='search_once':
        #search code based on a desc
        if conf['training_params']['reload']>0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
        codesearcher.load_code_reprs()
        codesearcher.load_use_codebase()
        try:
            query = args.query
            n_results = int(args.n)
            codes, sims = codesearcher.search(model, query, n_results)
            zipped = zip(codes, sims)
            zipped = sorted(zipped, reverse=True, key=lambda x: x[1])
            zipped = codesearcher.postproc(zipped)
            zipped = list(zipped)[:n_results]
            results = '\n\n'.join(map(str, zipped))  # combine the result into a returning string
            print(results)
        except Exception:
            print("Exception while parsing your input:")
            traceback.print_exc()

