import torch
import pickle

def load_obj("embeddings/word2vec_pascal"):
    with open(name + ".pkl", "rb") as f:
        embed = torch.from_numpy(pickle.load(f, encoding="latin-1"))
    embed.requires_grad = False
    return embed   # [21, 300]