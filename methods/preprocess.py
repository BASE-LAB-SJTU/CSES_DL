import re
from nltk import stem
from nltk.corpus import stopwords

#nltk.download('stopwords')
p = re.compile(r'([a-z]|\d)([A-Z])')
stemmer = stem.PorterStemmer()
code_stop_words = set(stopwords.words('codeStopWord'))
english_stop_words = set(stopwords.words('codeQueryStopWord'))
ka = re.compile(r'[^a-zA-Z]')


def code2string(raw):
    keep_alpha = re.sub(ka, ' ', raw)
    split_hump = re.sub(p, r'\1 \2', keep_alpha)
    lower = split_hump.lower().split()
    remove_stop = [w for w in lower if w not in code_stop_words]
    stemmed = ' '.join([stemmer.stem(j) for j in remove_stop])
    return stemmed


def code2list(raw):
    keep_alpha = re.sub(ka, ' ', raw)
    split_hump = re.sub(p, r'\1 \2', keep_alpha)
    lower = split_hump.lower().split()
    remove_stop = [w for w in lower if w not in code_stop_words]
    stemmed = [stemmer.stem(j) for j in remove_stop]
    return stemmed


def query2list(raw):
    keep_alpha = re.sub(ka, ' ', raw)
    split_hump = re.sub(p, r'\1 \2', keep_alpha)
    lower = split_hump.lower().split()
    remove_stop = [w for w in lower if w not in english_stop_words]
    stemmed = [stemmer.stem(j) for j in remove_stop]
    return stemmed
