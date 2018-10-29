from flask import Flask
from flask import request
from flask import render_template

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

import imp
import re
import math
import json
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from scipy import spatial
from gensim.models import Word2Vec
import pymorphy2
import scipy.sparse

app = Flask(__name__)


def preprocessing(query, stopwords):
    morph = pymorphy2.MorphAnalyzer()
    text = re.sub(r'[^\w\s]', ' ', query).split()
    new_text = []
    for word in text:
        lemma = morph.parse(word)[0].normal_form
        if lemma not in stopwords:
            new_text.append(lemma)
    return ' '.join(new_text)


def score_BM25(qf, dl, avgdl, n, N) -> float:
    """
    Compute similarity score between search query and documents from collection
    :return: score
    """
    k1 = 2.0
    b = 0.75
    idf = math.log(1 + (N-n+0.5)/(n+0.5))
    score = idf * (k1+1)*qf/(qf+k1*(1-b+b*(dl/avgdl)))
    return score

def compute_sim(word, vocabulary, ns, ts, dls):
    """
    Compute similarity score between search query and documents from collection
    """
    avgdl = np.mean(dls)
    N = len(dls)

    result = {}

    if word in vocabulary:
        i = vocabulary.index(word)
        n = ns[i]

        for j, t in enumerate(ts[i]):

            if dls[j]:
                qf = t/dls[j]
            else:
                qf = 0

            score = score_BM25(qf, dls[j], avgdl, N, n)
            result[j] = score
    return result


def okapi(q):
    new_q = preprocessing(q, russian_stopwords)

    res = defaultdict(int)
    for word in new_q.split():
        sim_dict = compute_sim(word, vocabulary, ns, ts, dls)
        for k in sim_dict:
            res[k] += sim_dict[k]

    return res


def get_w2v_vectors(q, russian_stopwords, model):
    """Получает вектор документа"""
    new_q = preprocessing(q, russian_stopwords)
    vector = np.zeros(300)

    tfidf = {}
    for key, value in enumerate(vectorizer.transform([new_q]).toarray()[0]):
        if value != 0:
            tfidf[key] = value

    for word in new_q.split():
        try:
            vector += model.wv[word] * tfidf[vectorizer.vocabulary_[word]]
        except KeyError:
            pass
    if len(new_q) > 0:
        return (vector / len(new_q)).tolist()
    return np.zeros(300).tolist()


def w2v_sim(q, w2v_base, model):
    vector = get_w2v_vectors(q, russian_stopwords, model)

    res = defaultdict(int)
    for i, base_vec in enumerate(w2v_base):
        res[i] = 1 - spatial.distance.cosine(vector, base_vec)

    return res

def search(query, w2v_base, n, data, model):
    res_okapi = okapi(query)
    res_w2v = w2v_sim(query, w2v_base, model)

    res = {}
    for k in res_okapi:
        res[k] = -0.5*res_okapi[k] + 0.8*res_w2v[k]

    top = sorted(res, key=res.get, reverse=True)[:n]

    answers = []
    for el in top:
        d = [res[el]] + data.iloc[[el]].values[0].tolist()
        answers.append(d)

    return answers

@app.route('/',  methods=['GET'])
def first():
    if request.args:
        query = request.args['query']
        if query == '':
            return render_template('first.html', answers=[])
        if 'n' in request.args:
            n = int(request.args['n'])
        else:
            n = 5
        answers = search(query, w2v_base, n, data, model)
        return render_template('first.html', answers=answers)
    return render_template('first.html', answers=[])


if __name__ == '__main__':

    data = pd.read_csv('data.csv', sep='\t').fillna('')
    print('data')

    with open('vocabulary.txt', 'r', encoding='utf-8') as f:
        vocabulary = f.read().split()
    print('voc')

    russian_stopwords = set(stopwords.words('russian'))
    with open('vectorizer', 'rb') as f:
        vectorizer = pickle.load(f)
    print('vect')

#   w2v_model_path = "/home/evgeniyamiller/araneum_none_fasttextskipgram_300_5_2018.model"
    model = Word2Vec.load('araneum_none_fasttextskipgram_300_5_2018.model')
    print('ok')

    with open('w2v_base.json', 'r') as f:
        w2v_base = json.load(f)
        print('base')

    sparse_matrix = scipy.sparse.load_npz('matrix.npz')
    ts = np.transpose(sparse_matrix.toarray())
    print('ts')

    with open('ns.json', 'r') as f:
        ns = json.load(f)
        print('ns')

    with open('dls.json', 'r') as f:
        dls = json.load(f)
        print('dls')

    app.run()#debug=True)
