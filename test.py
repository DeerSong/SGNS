# -*- coding:utf-8 -*-  
from word2vec_as_MF import Word2vecMF
from functions import *
import os

MODEL = True
W = Word2vecMF()
if MODEL:
    f = open('enwik9','rb')
    data = f.readlines()
    # for i in data:
    #     print i
    W.data_to_matrices(data,200,5,'model')
else:
    W.load_matrices('model')

print W.vocab
print ""
print W.D

os.system('play /home/deersong/音乐/CloudMusic/xx.mp3')

opt_experiment(W,'PS',200,MAX_ITER = 10)