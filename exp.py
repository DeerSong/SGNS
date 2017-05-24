# coding: utf-8

import matplotlib.pyplot as plt
from IPython import display

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

import itertools
import pickle
import math
import re
import sys

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec

from word2vec_as_MF import Word2vecMF
from functions import *


model_enwik = Word2vecMF()
model_enwik.load_matrices(from_file='enwik-200/matrices250.npz')

MAX_ITER = 7
words = ["five", "he", "main", "him"]
#for i in range(MAX_ITER):
    #model_enwik.load_CW('enwik-200/PS37iter_fromSVD_dim200_step5e-05_factors', i)
    #l = len(model_enwik.W[0])
    #MF = model_enwik.MF(model_enwik.C, model_enwik.W)
    #print l*l*MF*1e-9
    # print MF
model_enwik.load_CW('enwik-200/PS9iter_fromSVD_dim200_step5e-05_factors', MAX_ITER)
for word in words:
    ans = model_enwik.nearest_words(word)
    for i in ans:
        print i
    print ""
