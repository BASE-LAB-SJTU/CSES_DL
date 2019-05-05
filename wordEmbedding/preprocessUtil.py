
import os
import re

import nltk
from nltk import stem
nltk.download('stopwords')
import json
from nltk.corpus import stopwords

p = re.compile(r'([a-z]|\d)([A-Z])')
stemmer = stem.PorterStemmer()
code_stop_words = set(stopwords.words('code'))
english_stop_words = set(stopwords.words('english'))
ka = re.compile(r'[^a-zA-Z]')

class preprocessUtil:
    def codeClean(self,raw):
        keepAlpha = re.sub(ka, ' ', raw)
        split_hump = re.sub(p, r'\1 \2', keepAlpha)
        lower = split_hump.lower().split()
        removeStop = [w for w in lower if w not in code_stop_words]
        stemmed = ' '.join([stemmer.stem(j) for j in removeStop])
        return stemmed

    def codeCleanToList(self,raw):
        keepAlpha = re.sub(ka, ' ', raw)
        split_hump = re.sub(p, r'\1 \2', keepAlpha)
        lower = split_hump.lower().split()
        removeStop = [w for w in lower if w not in code_stop_words]
        stemmed = [stemmer.stem(j) for j in removeStop]
        return stemmed

    def nlCleanToList(self,raw):
        keepAlpha = re.sub(ka, ' ', raw)
        split_hump = re.sub(p, r'\1 \2', keepAlpha)
        lower = split_hump.lower().split()
        removeStop = [w for w in lower if w not in english_stop_words]
        stemmed = [stemmer.stem(j) for j in removeStop]
        return stemmed