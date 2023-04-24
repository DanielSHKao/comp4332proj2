# COMP4332 Project 2 - Social Network Mining
This repo is created by Shiu-hong Kao and Shi-heng Lo for COMP4332 Project 2 in Spring 2023.

## Quickstart
Create a conda virtual environment.
```
conda create -n comp4332proj2 python=3.9.12 -y
conda activate comp4332proj2
```
Pull the repo to your local computer and install the requirement packages.
```
git clone https://github.com/DanielSHKao/comp4332proj2
pip install -r requirements.txt
```

`node_embed.py` includes the random walk algorithm and trains the word2vec model. To start with, type
```
python node_embed.py
```
in the terminal to obtain the embedding model, which was pretrained and saved as `word2vec.model` in this repo.

To predict edges, we used a MLP model for binary classification, where 0 implies that edge doesn't exist and 1 represents the existence. To train the model, run the following command.
```
python train_mlp.py
```
We also pretrained the model and saved it as ```edge_best_fc.pt``` in this repo. When training the MLP, we randomly generate a set of false edges that do not exist to balance the dataset with label 0.

Our MLP model achieves accuracy 93.39% on binary classification. We extracted the score of edge by adding softmax function to the MLP logit, regarding the value of class 1 as the prediction score. Our approach achieves AUC 0.9711 on the validation dataset.

For prediction, the following command will generate `data/pred.csv` based on `data/test.csv`.
```
python get_test_score.py
```
We have pre-processed this and stored the result in `./data/`.
