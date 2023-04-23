import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

class NetworkDataset(Dataset):
    def __init__(self,embed_model,true_edges,false_edges):
        super().__init__()
        self.embed_model=embed_model
        self.data=[]
        for e in true_edges:
          self.data.append((e,1))
        for e in false_edges:
          self.data.append((e,0))
        random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        edge, label = self.data[index]
        try:
            first = self.embed_model.wv[edge[0]]
        except:
            first = np.zeros(32)
        try:
            second = self.embed_model.wv[edge[1]]
        except:
            second = np.zeros(32)
        
        return (torch.from_numpy(np.concatenate((first,second))).float(), label)
class NetworkDataModule(pl.LightningDataModule):
    def __init__(self, embed_model,train_edges, valid_edges, test_edges, train_false_edges,valid_false_edges, batch_size,num_workers):
        super().__init__()
        self.num_workers=num_workers
        self.batch_size=batch_size
        self.train_set = NetworkDataset(embed_model,train_edges,train_false_edges)
        self.valid_set = NetworkDataset(embed_model,valid_edges,valid_false_edges)
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,num_workers=self.num_workers,shuffle = True,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size,num_workers=self.num_workers,shuffle = True,pin_memory=True)
    
