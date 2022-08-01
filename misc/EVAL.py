import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from UNET import UNET
import pandas as pd
import torch.nn.functional as F
from NETutils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    SaveBestModel,
    save_model
)

EVAL_IMG_DIR = r'/home/ethan/PycharmProjects/UNET_pytorch/IR/notBoats'

model = UNET(in_channels=3,out_channels=1)
