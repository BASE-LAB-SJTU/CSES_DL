import math


def firstPos(real, predict):
    pos = len(predict)
    for idx, val in enumerate(predict):
        if val in real:
            pos = idx
            break
    return pos


def recall(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1
    return sum / float(len(real))


def precision(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1
    return sum / float(len(predict))


def f_measure(real, predict):
    pre = precision(real, predict)
    rec = recall(real, predict)
    try:
        f = 2 * pre * rec / (pre + rec)
    except ZeroDivisionError:
        f = -1
    return f


def ACC(real, predict):
    """accuracy = intersect / min(topk, len(real))"""
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1
    return sum / float(min(len(predict), len(real)))


def MAP(real, predict):
    sum = 0.0
    cur = 1
    l = len(real)
    for id, val in enumerate(predict):
        if val in real:
            sum = sum + cur / (id+1)
            cur+=1
            if cur == l:
                break
    return sum / l

def MRR(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1.0 / float(index + 1)
    return sum / float(min(len(predict), len(real)))


def NDCG(real, predict):
    dcg = 0.0
    idcg = IDCG(len(real))
    for i, predictItem in enumerate(predict):
        if predictItem in real:
            itemRelevance = 1
            rank = i + 1
            dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
    return dcg / float(idcg)


def IDCG(n):
    idcg = 0
    itemRelevance = 1
    for i in range(n):
        idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
    return idcg
