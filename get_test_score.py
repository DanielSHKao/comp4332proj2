import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from model_api import customMLP
from gensim.models import Word2Vec
#from train_mlp import customMLP
def test_obs(embed_model,mlp,src,tar):
    try:
        src_v = embed_model.wv[src]
    except:
        src_v = torch.zeros(32)
    try:
        tar_v = embed_model.wv[tar]
    except:
        tar_v = torch.zeros(32)
    
    v = torch.from_numpy(np.concatenate((src_v,tar_v))).float()
    v = v.unsqueeze(0)
    logit = mlp(v)
    score = F.softmax(logit,dim=1)[0][1].item()
    return score
mlp_model = customMLP(64,512,2,nn.ReLU())
w2v_model = Word2Vec.load("word2vec.model")
mlp_model = torch.load("edge_best_fc.pt")
mlp_model.eval()
df = pd.read_csv('data/test.csv')
scores=[]
for index, row in df.iterrows():
    s=test_obs(w2v_model,mlp_model,row['src'],row['dst'])
    scores.append(s)
    #print(row['src'],row['dst'], s)
df['score']=pd.Series(scores)
df.to_csv("data/pred.csv")