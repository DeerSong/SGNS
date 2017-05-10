
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

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec

from word2vec_as_MF import Word2vecMF
from functions import *


# ## Read and preprocess enwik9

# In[2]:

# Load enwik 9
f = open("data/enwik/enwik9.txt")
data = f.readlines()
f.close()
print "finish Reading"
# data = np.genfromtxt("data/enwik/enwik9.txt", dtype=str, delimiter='.')
#data = np.loadtxt("data/enwik/enwik9.txt", dtype=str, delimiter='.')


# In[ ]:

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
    return(words)


# In[ ]:

sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for line in data:
    if line != '\n':
        sents = line.split('.')
        for sentence in sents:
            sentences += [wiki_to_wordlist(sentence.decode('utf-8'))]
# print sentences

indices = []
for i, sentence in enumerate(sentences):
    if not sentence:
        pass
    else:
        indices.append(i)

real_sentences = np.array(sentences)[indices]
print "finish Parsing"

# In[ ]:

# Create word2vec as matrix factorization model
model_enwik = Word2vecMF()
model_enwik.data_to_matrices(real_sentences, 200, 5, 'enwik-200/matrices.npz')


# In[ ]:

# If the model has been already created, load it from file
model_enwik = Word2vecMF()
model_enwik.load_matrices(from_file='enwik-200/matrices.npz')


# ## Train ro_sgns model starting from SVD of SPPMI

# In[ ]:

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

# Train the model
opt_experiment(model_enwik,
               mode='PS', 
               d=200,
               eta = 5e-5,
               MAX_ITER=40,
               from_iter=0,
               start_from='SVD',
               init=(True, C_svd, W_svd))

