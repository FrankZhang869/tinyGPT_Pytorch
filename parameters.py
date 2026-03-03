import torch

batch_size = 32
block_size = 128
max_iters = 5000
eval_interval = 500
learning_rate = 2e-4
embed_dim = 192
num_heads = 6
num_layers = 4
dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"
