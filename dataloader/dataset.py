import torch
import glob
from torchvision import transforms
from PIL import Image
import random
class Dataset (torch.utils.data.Dataset):
	def __init__(self, image_glob, mask_glob):
		super().__init__()

		self.imagePaths = self.GetImagePaths(image_glob)
		self.maskPaths = self.GetMaskPaths(mask_glob)
		self.maskCount = len(self.maskPaths)
		self.imageCount = len(self.imagePaths)

		self.toTensorTransform = transforms.ToTensor()

	def GetImagePaths(self, image_glob):
		return glob.glob(image_glob)

	def GetMaskPaths(self, mask_glob):
		return glob.glob(mask_glob)

	def __len__(self):
		return self.imageCount

	def __getitem__(self, index):
		gt_img = Image.open(self.imagePaths[index])
		#gt_img = self.imageTransform(gt_img.convert('RGB'))
		gt_img = self.toTensorTransform(gt_img.convert("RGB"))

		mask = Image.open(self.maskPaths[random.randint(0, self.maskCount - 1)])
		#mask = self.mask_transform(mask.convert('RGB'))
		mask = self.toTensorTransform(mask.convert("RGB"))

		return mask, gt_img #gt_img * mask
