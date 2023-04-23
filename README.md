# comp4332proj2
This repo is created by Shiu-hong Kao and Shi-heng Lo for COMP4332 Project 2 in Spring 2023.

## Quickstart
Pull the repo to your local computer.

`node_embed.py` includes the random walk algorithm and trains the word2vec model. To start with, type
```
python node_embed.py
```
in the terminal to obtain the embedding model, which was pretrained and saved as `word2vec.model` in this repo.

To predict edges, we used a MLP model for binary classification, where 0 implies that edge doesn't exist and 1 represents the existence. To train the model, run the following command.
```
python train_mlp.py
```
We also pretrained the model and saved it as ```edge_best_fc.pt``` in this repo.

After training the repo, we highlight that we can obtain the score of edge by adding softmax function to the MLP logit and extracting the value of class 1.
