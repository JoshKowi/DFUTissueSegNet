# get local path to project
import json
import shutil
import sys
import os

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

with open("config.json", "r") as f:
    config = json.load(f)
BASE_PATH = config["PROJECT_DIR"]
DATA_PATH = config["DATA_DIR"]

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import cv2
import numpy as np
import segmentation_models_pytorch as smp # Be careful: this needs to be the local version, not the pip package!
from segmentation_models_pytorch.utils import metrics, losses, base
import random
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from datetime import datetime
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

import segmentation_models_pytorch.utils.losses as losses
print(losses.__file__)


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            list_IDs,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
            to_categorical:bool=False,
            resize=(False, (256, 256)), # To resize, the first value has to be True
            n_classes:int=6,
            default_img=None,
            default_mask=None,
    ):
        self.ids = list_IDs
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.to_categorical = to_categorical
        self.resize = resize
        self.n_classes = n_classes
        self.default_img = default_img
        self.default_mask = default_mask

    def __getitem__(self, i):
        try:
              # read data
              image = cv2.imread(self.images_fps[i])
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              mask = cv2.imread(self.masks_fps[i], 0)     # ----------------- pay attention ------------------ #
        except Exception as e:
            print("********** Error occured loading default image and mask. *********")
            image = self.default_img
            mask = self.default_mask

        if self.resize[0]:
            image = cv2.resize(image, self.resize[1], interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, self.resize[1], interpolation=cv2.INTER_NEAREST)

        mask = np.expand_dims(mask, axis=-1)  # adding channel axis # ----------------- pay attention ------------------ #

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.to_categorical:
            mask = torch.from_numpy(mask)
            mask = F.one_hot(mask.long(), num_classes=self.n_classes)
            mask = mask.type(torch.float32)
            mask = mask.numpy()
            mask = np.squeeze(mask)

            mask = np.moveaxis(mask, -1, 0)

        return image, mask

    def __len__(self):
        return len(self.ids)

class Dataset_without_masks(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            list_IDs,
            images_dir,
            preprocessing=None,
            resize=(False, (256, 256)), # To resize, the first value has to be True
            n_classes:int=4,
    ):
        self.ids = list_IDs
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        self.preprocessing = preprocessing
        self.resize = resize
        self.n_classes = n_classes

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.resize[0]:
            image = cv2.resize(image, self.resize[1], interpolation=cv2.INTER_NEAREST)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image

    def __len__(self):
        return len(self.ids)

def get_training_augmentation():
    train_transform = [

        albu.OneOf(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
            ],
            p=0.8,
        ),

        albu.OneOf(
            [
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=0.1, border_mode=0), # scale only
                albu.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=0.1, border_mode=0), # rotate only
                albu.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=0.6, border_mode=0), # shift only
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=0.2, border_mode=0), # affine transform
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Perspective(p=0.2),
                albu.GaussNoise(p=0.2),
                albu.Sharpen(p=0.2),
                albu.Blur(blur_limit=3, p=0.2),
                albu.MotionBlur(blur_limit=3, p=0.2),
            ],
            p=0.5,
        ),

        albu.OneOf(
            [
                albu.CLAHE(p=0.25),
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
                albu.RandomGamma(p=0.25),
                albu.HueSaturationValue(p=0.25),
            ],
            p=0.3,
        ),

    ]

    return albu.Compose(train_transform, p=0.9) # 90% augmentation probability


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(512, 512)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# Parameters
BASE_MODEL = 'MiT+pscse'
ENCODER = 'mit_b3'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 16
n_classes = 4
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0001 # learning rate
EPOCHS = 500
WEIGHT_DECAY = 1e-5
SAVE_WEIGHTS_ONLY = True
RESIZE = (False, (256,256)) # if resize needed
TO_CATEGORICAL = True
SAVE_BEST_MODEL = True
SAVE_LAST_MODEL = False

PERIOD = 10 # periodically save checkpoints
RAW_PREDICTION = False # if true, then stores raw predictions (i.e. before applying threshold)
RETRAIN = False

# For early stopping
EARLY_STOP = True # True to activate early stopping
PATIENCE = 50 # for early stopping

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def save(model_path, epoch, model_state_dict, optimizer_state_dict):

    state = {
        'epoch': epoch + 1,
        'state_dict': deepcopy(model_state_dict),
        'optimizer': deepcopy(optimizer_state_dict),
        }

    torch.save(state, model_path)

# Loss function
dice_loss = losses.DiceLoss()
focal_loss = losses.FocalLoss()
total_loss = base.SumOfLosses(dice_loss, focal_loss)
dce_loss = losses.DynamicCEAndSCELoss() # dynamic CE

# Metrics
metrics = [
    metrics.IoU(threshold=0.5),
    metrics.Fscore(threshold=0.5),
]


def read_names(text_file):
  """This function reads names from a text file"""
  with open(text_file, "r") as f:
    names = f.readlines()
    names = [name.split('\n')[0] for name in names] # remove \n (newline)

  return names

# Create a function to read names from a text file, and add extensions
def read_names_ext(txt_file, ext=".png"):
  with open(txt_file, "r") as f: names = f.readlines()
  
  names = [name.strip("\n") for name in names] # remove newline
  # Names are without extensions. So, add extensions
  names = [name + ext for name in names]

  return names

def delt(dir, names):
  """This function deletes files specified in (names) from a directory (dir)"""
  for name in names:
    if os.path.exists(os.path.join(dir, name)): os.remove(os.path.join(dir, name))

def copy(dir_src, dir_dst, names):
  """This function copy files specified in (names) from (dir_src) to (dir_dst)"""
  for name in names:
    shutil.copy(os.path.join(dir_src, name), os.path.join(dir_dst, name))

def copy_all(dir_src, dir_dst):
    """This function copy all files from (dir_src) to (dir_dst)"""
    for name in os.listdir(dir_src):
        shutil.copy(os.path.join(dir_src, name), os.path.join(dir_dst, name))

#. Could restore the 50 predictions that were used for good training run after making new predictions on all images.
def replace(dir_ann_phase_prev, dir_ann_phase_current, prev_phase_names):
  """
  dir_ann_phase_prev: Director of annotations from previous training phase
  dir_ann_phase_current: Director of annotations of current training phase
  prev_phase_names: Names of previous training phase
  """
  delt(dir_ann_phase_current, prev_phase_names) # delete files from the train directory
  copy(dir_ann_phase_prev, dir_ann_phase_current, prev_phase_names) # replace annotations of current phase by the prev phase