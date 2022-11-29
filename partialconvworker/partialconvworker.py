import random
import torch
from torchvision import transforms
from torchvision import utils
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image

import os
import glob
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

from lib.worker import VanisherWorker
from torch.utils import data
from partialconvworker.model.partialconvnetwork import PartialConvUNet
from partialconvworker.model.loss import Loss

# reverses the earlier normalization applied to the image to prepare output
def unnormalize(x):
	x.transpose_(1, 3)
	x = x * torch.Tensor(STDDEV) + torch.Tensor(MEAN)
	x.transpose_(1, 3)
	return x

class SubsetSampler(data.sampler.Sampler):
	def __init__(self, start_sample, num_samples):
		self.num_samples = num_samples
		self.start_sample = start_sample

	def __iter__(self):
		return iter(range(self.start_sample, self.num_samples))

	def __len__(self):
		return self.num_samples


def requires_grad(param):
	return param.requires_grad

class PartialConvWorker(VanisherWorker):
	def __init__(self, checkpoint_path, learning_rate):
		devstring = "cuda" if torch.cuda.is_available else "cpu"
		print("DevString", devstring)
		self.device = torch.device(devstring)
		self.imageNormalizingTransform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

		self.model = PartialConvUNet().to(self.device)

		print("Setup Adam optimizer...")
		self.optimizer = torch.optim.Adam(filter(requires_grad, self.model.parameters()), lr=learning_rate)

		self.checkpoint_path = checkpoint_path
		if os.path.exists(self.checkpoint_path):
			print("Loading checkpoint at", checkpoint_path)
			checkpoint_dict = torch.load(checkpoint_path, map_location=devstring)
			self.model.load_state_dict(checkpoint_dict["model"])
			self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
	
	def Compute(self, gt_path, mask_path, out_path):
		self.model.eval()
		try:
			mask = Image.open(mask_path)
			mask = self.mask_transform(mask.convert("RGB"))

			gt_img = Image.open(gt_path)
			gt_img = self.img_transform(gt_img.convert("RGB"))
			img = gt_img * mask

			img = img.to(self.device)
			mask = mask.to(self.device)
			gt_img = gt_img.to(self.device)

			img.unsqueeze_(0)
			gt_img.unsqueeze_(0)
			mask.unsqueeze_(0)

			with torch.no_grad():
				output = self.model(img, mask)

			output = (mask * img) + ((1 - mask) * output)

			'''loss_func = CalculateLoss()
			print (img, mask, output, gt_img)
			loss_out = loss_func(img, mask, output, gt_img)
			for key, value in loss_out.items():
				print("KEY:{} | VALUE:{}".format(key, value))'''

			save_image(self.unnormalize(output), out_path)
			return True
		except:
			return False
	
	def Train(self, epochCount, train_batch_size, test_batch_size, data_train, data_test, log_interval):
		loss_func = Loss().to(self.device);

		self.model.freeze_enc_bn = True
		train_size = len(data_train)
		test_size = len(data_test)
		print("Loaded training dataset with {} train samples, {} test samples, and {} masks".format(train_size, test_size, data_train.maskCount))

		train_iters = train_size // train_batch_size
		test_iters = test_size // train_batch_size
		
		for epoch in range(0, epochCount):
			iterator_train = iter(data.DataLoader(data_train, batch_size=train_batch_size, num_workers=1, sampler=SubsetSampler(0, train_size)))

			# TRAINING LOOP
			print("\nEPOCH:{} of {} - starting training loop from iteration:0 to iteration:{}\n".format(epoch, epochCount, train_iters))
			
			for i in tqdm(range(0, train_iters)):
				torch.cuda.empty_cache()
				# Sets model to train mode
				self.model.train()

				# Gets the next batch of images
				mask, gt = [x.to(self.device) for x in next(iterator_train)]

				gt = self.imageNormalizingTransform(gt)
				image = gt * mask
				
				# Forward-propagates images through net
				# Mask is also propagated, though it is usually gone by the decoding stage
				output = self.model(image, mask)

				loss_dict = loss_func(image, mask, output, gt)
				loss = 0.0

				# sums up each loss value
				if (i + 1) % log_interval == 0 and log_interval != -1: print("")
				for key, value in loss_dict.items():
					loss += value
					if (i + 1) % log_interval == 0 and log_interval != -1:
				 		print(key, value.item(), epoch * train_iters + i + 1)

				# Resets gradient accumulator in optimizer
				self.optimizer.zero_grad()
				# back-propogates gradients through model weights
				loss.backward()
				# updates the weights
				self.optimizer.step()
				
				del mask
				del image
				del gt
				del output
				del loss_dict
				del loss
			del iterator_train

			state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
			torch.save(state, self.checkpoint_path)
			print("Saved to check point")
			del state

			print ("Testing after epoch", epoch)
			iterator_test = iter(data.DataLoader(data_test, batch_size=test_batch_size, num_workers=1, sampler=SubsetSampler(0, train_size)))
			test_losses = {}
			for i in tqdm(range(0, test_iters)):
				torch.cuda.empty_cache()
				# Sets model to train mode
				self.model.eval()

				# Gets the next batch of images
				mask, gt = [x.to(self.device) for x in next(iterator_test)]

				gt = self.imageNormalizingTransform(gt)
				image = gt * mask
				
				# Forward-propagates images through net
				# Mask is also propa
				# gated, though it is usually gone by the decoding stage
				output = self.model(image, mask)

				loss_dict = loss_func(image, mask, output, gt)

				# sums up each loss value
				for key, value in loss_dict.items():
					if key not in test_losses.keys():
						test_losses[key] = 0
					test_losses[key] += value.item()
				
				del mask
				del image
				del gt
				del output
				del loss_dict
			
			for key, value in test_losses.items():
				print(key, value / test_iters)
			
			del iterator_test


