## 학습을 시키는 부분을 구현

## 필요한것 import
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from model import *

## 기본적인 변수 세팅
lr = 2e - 4
batch_size = 128
num_epoch = 200

data_dir = './datasets/'
ckpt_dir = './checkpoint'
result_dir = './result'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##추가적인 세팅
result_dir_train = os.path.join(result_dir, 'train')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir_train):
    os.makedirs(os.path.join(result_dir_train, 'png'))
if not os.path.exists(result_dir_test):
    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_train, 'numpy'))

## 데이터를 불러오기 위한 부분
# transforms_train = 

## 네트워크 불러오기

## Loss Function(손실함수)를 정의하기

## Optimizer 정의하기

## Train 학습