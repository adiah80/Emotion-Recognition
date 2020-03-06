
##################### IMPORTS #####################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchaudio import load
import torchaudio
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse

sample_rate = 16000
max_paded_dim = 1376