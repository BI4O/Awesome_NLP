import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 限制数据类型
DTYPE = torch.Tensor
N_HIDDEN = 128


# 准备数据
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

# 数据预处理:
# 1.构造词表
word_list = ''.join(sentences).split()
word_list = list(set(word_list)) # 去重
word2ix = {w: i for i,w in enumerate(word_list)}
ix2word = {i: w for i,w in enumerate(word_list)}
n_class = len(word_list)

# 2.构造批次batch
# def make_batch(sentences):
# input_batch = [np.eye(n_class)[[word2ix[n] for n in sentences[0].split()]]]





# print([[word2ix[n] for n in sentences[0].split()]])
for i in sentences[0].split():
    print(word2ix[i])