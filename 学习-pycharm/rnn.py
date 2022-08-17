import math
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from d2l import torch as d2l

batch_size,num_steps=32,35

train_iter,vocab = d2l.load_data_time_machine(batch_size,num_steps)

print(vocab[:10])
