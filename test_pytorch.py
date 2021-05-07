SEED = 8734

import collections
import json
import os
import pickle
import random
import shutil
import sys
import types
from datetime import datetime

import numpy as np



np.random.seed(SEED)
random.seed(SEED)

import matplotlib.pyplot as plt
# %matplotlib inline

import pandas as pd
import time
from scipy import sparse

import torch.nn as nn
import torch.nn.functional as F
import torch

from collections import defaultdict

from sklearn.preprocessing import MinMaxScaler
print(torch.__version__)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.cuda.set_device(0)

USE_CUDA = True

CUDA = USE_CUDA and torch.cuda.is_available()

device = torch.device("cuda" if CUDA else "cpu")



if CUDA:
    print('run on cuda %s' % os.environ['CUDA_VISIBLE_DEVICES'])
else:
    print('cuda not available')
