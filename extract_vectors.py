import os
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
import pickle

model_path = "../../media/bach3/seungheon/W2V/AllMusic_Lastfm_sg1_size300_iter10/model"
model = KeyedVectors.load(model_path, mmap='r')

word_vectors = {}
for i in tqdm(model.wv.vocab):
    word_vectors[i] = model.wv[i]

with open("./static/vectors/word_vectors.pkl", 'wb') as f:
    pickle.dump(word_vectors, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Finish")