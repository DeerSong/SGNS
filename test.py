# coding: utf-8

# ## Imports

# In[1]:


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

def wiki_to_wordlist(sentence, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # 3. Convert words to lower case and split them
    words = sentence.split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return words

# ## Read and preprocess enwik9
# Load enwik 9
if __name__ == '__main__':
  if len(sys.argv)!=3:
    print "Parameter Error!"
    print "Program Shutdown!"
    os._exit(1)

  filename = sys.argv[1] # training data file
  status = int(sys.argv[2]) # 0:load from file, 1:load from npz model

  model_enwik = Word2vecMF()
  if status==0:
    f = open("data/enwik/"+filename)
    data = f.read()
    data = data.split('.')
    f.close()
    print "finish Reading"
    # data = np.genfromtxt("data/enwik/enwik9.txt", dtype=str, delimiter='.')
    #data = np.loadtxt("data/enwik/enwik9.txt", dtype=str, delimiter='.')

    sentences = []  # Initialize an empty list of sentences
    num = len(data)
    k = 0

    for k in range(len(data)):
        # for i in xrange(1,10):
        #     if (k/float(num)<0.1*i):
        #         print 0.1*i
        #         break
        data[k] = wiki_to_wordlist(data[k])
        k += 1

    # print data
    # print sentences
    sentences = data
    del data

    indices = []
    for i, sentence in enumerate(sentences):
        if not sentence:
            pass
        else:
            indices.append(i)

    real_sentences = np.array(sentences)[indices]
    del indices
    del sentences
    print "finish Parsing"

    # Create word2vec as matrix factorization model
    model_enwik.data_to_matrices(real_sentences, 200, 1, 'enwik-200/'+filename+'.npz')
  else:
    # If the model has been already created, load it from file
    model_enwik.load_matrices(from_file='enwik-200/'+filename+'.npz')
    print "finish Loading"
  # ## Train ro_sgns model starting from SVD of SPPMI

  # SVD initialization
  SPPMI = np.maximum(np.log(model_enwik.D) - np.log(model_enwik.B), 0)
  u, s, vt = svds(SPPMI, k=200)
  C_svd = u.dot(np.sqrt(np.diag(s))).T
  W_svd = np.sqrt(np.diag(s)).dot(vt)


  # In[ ]:

  model_enwik.C = C_svd
  model_enwik.W = W_svd

  model_enwik.save_CW('enwik-200/initializations/SVD_dim200', 0)


  # In[ ]:
  print "Begin to Train"
  # Train the model
  opt_experiment(model_enwik,
                 mode='PS', 
                 d=200,
                 eta = 3e-5,
                 MAX_ITER=20,
                 from_iter=0,
                 start_from='SVD',
                 init=(True, C_svd, W_svd))

