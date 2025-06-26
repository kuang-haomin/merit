import os
import random
import SimpleITK as sitk
import numpy as np
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from lib.networks import MaxViT, MaxViT4Out, MaxViT_CASCADE, MERIT_Parallel, MERIT_Cascaded
from trainer import trainer_synapse,trainer_muregpro
from torchsummaryX import summary
from ptflops import get_model_complexity_info
import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from utils.dataset_synapse import Synapse_dataset, RandomGenerator, muregpro_dataset
from utils.utils import powerset
from utils.utils import one_hot_encoder
from utils.utils import DiceLoss
from utils.utils import val_single_volume

batch_slices = [1,5,9,6,11,23]
pred = np.zeros((128,128,50))
preds_mean = np.ones((len(batch_slices),128,128))
print(pred.max())
pred[:,:,batch_slices] = np.clip(preds_mean, 0, 1).transpose(2, 1, 0)
print(pred.max())
