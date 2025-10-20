import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import glob as _glob
import csv
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

from math import log, cos, pi, floor
import math



csv.register_dialect(
    'mydialect',
    delimiter = ',',
    quotechar = '"',
    doublequote = True,
    skipinitialspace = True,
    lineterminator = '\r\n',
    quoting = csv.QUOTE_MINIMAL)

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )

    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


#region 데이터 저장용

def writecsv(csvname,contents):
    f = open(csvname, 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(contents)
    f.close()



def mkdir(paths):
  if not isinstance(paths, (list, tuple)):
    paths = [paths]
  for path in paths:
    if not os.path.exists(path):
      os.makedirs(path)

def split(path):
  """Return dir, name, ext."""
  dir, name_ext = os.path.split(path)
  name, ext = os.path.splitext(name_ext)
  return dir, name, ext


def glob(dir, pats, recursive=False):  # faster than match, python3 only
    pats = pats if isinstance(pats, (list, tuple)) else [pats]
    matches = []
    for pat in pats:
        matches += _glob.glob(os.path.join(dir, pat), recursive=recursive)
    return matches
#endregion

def split(path):
  """Return dir, name, ext."""
  dir, name_ext = os.path.split(path)
  name, ext = os.path.splitext(name_ext)
  return dir, name, ext

def cosine_decay(step,alpha,decay_steps):
  step = min(step, decay_steps)
  cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
  decayed = (1 - alpha) * cosine_decay + alpha
  return decayed

def csv2list(filename):
  lists=[]
  file=open(filename,"r")
  while True:
    line=file.readline().replace('\n','')
    if line:
      line=line.split(",")
      lists.append(line)
    else:
      break
  return lists
