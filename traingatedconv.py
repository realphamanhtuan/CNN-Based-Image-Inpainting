root_path = '/mnt/d/KLTN/CNN-Based-Image-Inpainting/'
train_glob = root_path + 'dataset/places2/train/*/*/*.jpg'
test_glob = root_path + 'dataset/places2/test/*.jpg'
mask_glob = root_path + 'dataset/irregular_mask1/*.png' #2 for partialconv

log_dir = root_path + 'training_logs'
save_dir = root_path + 'models'
checkpoint_path = root_path + "models/gatedconv.pth"
learning_rate = 1e-4 #5e-4 for gated conv
epoch = 50
train_batch_size = 4
test_batch_size = 4
log_interval = -1 #no log

import os
import torch
from dataloader.dataset import *
from gatedconvworker.gatedconvworker import GatedConvWorker

print("Creating output directories")
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

print("Initiating training sequence")
torch.cuda.empty_cache()

print("Initializing dataset with globs:", train_glob, test_glob, mask_glob)
data_train = Dataset(train_glob, mask_glob, False)
data_test = Dataset(test_glob, mask_glob, False)

worker = GatedConvWorker(checkpoint_path, learning_rate)
worker.Train(epoch, train_batch_size, test_batch_size, data_train, data_test, log_interval)

