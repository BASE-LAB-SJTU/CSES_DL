import argparse

from methods.yeSearch.code_searcher import CodeSearcher
from methods.yeSearch import configs
from methods.yeSearch import skipgram

def parse_args():
    parser = argparse.ArgumentParser("Test and Evaluate Word Embedding and Document Similarity Model")
    parser.add_argument("--path")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    conf = getattr(configs, args.proto)()

    codesearcher = CodeSearcher(conf)

    if args.mode=='train':
        skipgram.train(conf)

    elif args.mode == 'preprocess':
        codesearcher.clean_codebase()
        codesearcher.generate_idf()

    elif args.mode == 'eval':
        codesearcher.eval(20)

