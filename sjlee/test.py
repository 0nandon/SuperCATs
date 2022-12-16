
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

a = torch.randn(3, 4)
print(a.dtype)
a = F.pad(a, (0, 2, 0, 1), 'constant', 0).type(a.dtype)
print(a)