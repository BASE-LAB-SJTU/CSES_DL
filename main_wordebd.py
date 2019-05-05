import argparse

from wordEmbedding.wordEmbeddingForCode import WordEmbeddingForCode


def parse_args():
    parser = argparse.ArgumentParser("Test and Evaluate Word Embedding and Document Similarity Model")
    parser.add_argument("--path")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    we = WordEmbeddingForCode(args.path)
    we.eval(20)

