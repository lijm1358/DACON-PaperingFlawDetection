import albumentations as A
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import wandb
from loss import create_criterion
from sklearn.model_selection import train_test_split
import random
import glob
import json
import os
from albumentations.pytorch.transforms import ToTensorV2
from sklearn import preprocessing
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from importlib import import_module

def main():
  wandb.init(
        project="papering",
        config={
            "Model": config["model"],"Dataset": config["dataset"],"Optimizer": config["optimizer"]["type"],"Learning Rate": config["optimizer"]["args"]["lr"],
            "Scheduler": config["scheduler"]["type"],"Epoch": config["params"]["epochs"],"Batch Size": config["params"]["batch_size"],
        },
    )
