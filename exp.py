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
