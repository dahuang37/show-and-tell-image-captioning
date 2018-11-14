import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
import nltk
from build_vocab import Vocabulary
import pickle
import json


class CocoDataset(Dataset):
	"""
		COCO Custom Dataset
	"""
	def __init__(self, root, annFile, vocab=None, transform=None):
		"""
		Set the path for images, captions, and vocabulary wrapper

		Args:
			root: Image root [./data/coco/train2014]
			annFile: Json annotations for images
			vocab:
			transform:
		"""
		self.root = root
		self.annFile = annFile
		self.vocab = vocab
		self.transform = transform
		self.coco = COCO(annFile)
		print(self.coco)
		self.ids = list(self.coco.anns.keys())
		print(len(list(self.coco.anns.keys())))
		print("images",len(list(self.coco.imgs.keys())))

	def __getitem__(self, index):
		"""
		returns one data pair (image and caption)
		"""
		coco = self.coco
		vocab = self.vocab
		ann_id = self.ids[index]
		caption = coco.anns[ann_id]['caption']
		img_id = coco.anns[ann_id]['image_id']
		path = coco.loadImgs(img_id)[0]['file_name']
		image = Image.open(os.path.join(self.root, path)).convert('RGB')
		if self.transform is not None:
			image = self.transform(image)
		# # Convert caption (string) to word ids.
		tokens = nltk.tokenize.word_tokenize(str(caption).lower())
		caption = []
		caption.append(vocab('<start>'))
		for token in tokens:
			caption.append(vocab(token))
		caption.append(vocab('<end>'))
		target = torch.Tensor(caption)
		return image, target
		

	def __len__(self):
		return len(self.ids)

def collate_fn(data):
	images, captions = zip(*data)
	lengths = [len(cap) for cap in captions]
	
	targets = torch.zeros(len(captions), max(lengths)).long()
	for i, cap in enumerate(captions):
		end = lengths[i]
		targets[i, :end] = cap[:end]

	return images, targets, lengths


root = "./data/coco"
# train_root = os.path.join(root,"train2014")
# valid_root = os.path.join(root,"val2014")
# train_annFile = os.path.join(root, "annotations/captions_train2014.json")
valid_annFile = os.path.join(root, "annotations/captions_val2014.json")
with open(valid_annFile,"r") as f:
	test_file = json.load(f)
	print(test_file.keys())
	print(test_file["images"][0])
	print(test_file["annotations"][0])
	num_of_imgs = len(test_file["images"])
	test_dict = {'images':{}, 'annotations':{}}
	valid_dict = {'images':{}, 'annotations':{}}
	test_ids = np.random.choices(num_of_imgs, 4000)
	for i in rang

	# print(test_file["annotations"])
# 	coco = COCO(valid_annFile)
# 	print(list(coco.imgs.keys()))

def save_test_imgs(valid_annFile):
	coco = COCO(valid_annFile)
	num_of_imgs = len(coco.imgs.keys())
	print("Number of images: {}".format(num_of_imgs))
	# saving 4000 as test images



save_test_imgs(valid_annFile)

# print(os.listdir())

# with open("./data/coco/vocab.pkl", "rb") as f:
# 	vocab = pickle.load(f)


def get_data_loader(msg):#root, json, vocab, transform, batch_size, shuffle, num_workers
	pass
# train_set = CocoDataset(root=test_root,
# 					  annFile=test_annFile,
# 					  vocab = vocab,
# 					  transform=transforms.ToTensor())

# print(len(train_set))
# images, target = train_set[0]
# print(target)
# data_loader = torch.utils.data.DataLoader(dataset=train_set, 
# 										   batch_size=4,
# 										   shuffle=True,
# 										   num_workers=0,
# 										   collate_fn=collate_fn,
# 										   )
