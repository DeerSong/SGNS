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


if __name__ == '__main__':
    if len(sys.argv)!=3:
        print "Parameter Error!"
        print "Program Shutdown!"
    filename = sys.argv[1]
    model_enwik = Word2vecMF()
    model_enwik.load_matrices(from_file='enwik-200/'+filename+'.npz')

    print(model_enwik.vocab.shape)
    print(model_enwik.inv_vocab.shape)
    print(model_enwik.D.shape)
    MAX_ITER = int(sys.argv[2])
    words = ["him", "five", "main"]
    #for i in range(MAX_ITER):
    #    model_enwik.load_CW('enwik-200/PS9iter_fromSVD_dim200_step5e-05_factors', i)
    #    MF = model_enwik.MF(model_enwik.C, model_enwik.W)
    #    print MF
    #model_enwik.load_CW('enwik-200/250ac', 7)
    x = datasets_corr(model_enwik, "datasets", "enwik-200/"+filename,MAX_ITER,plot_corrs=False)
    for i in x:
        print i
        print x[i]
