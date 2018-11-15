import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
import nltk
from .build_vocab import Vocabulary
import pickle
import json
import argparse


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
		print(os.getcwd())
		self.root = root
		self.annFile = annFile
		self.vocab = vocab
		self.transform = transform
		self.coco = COCO(annFile)
		self.ids = list(self.coco.anns.keys())
		# print(len(list(self.coco.anns.keys())))
		# print("images",len(list(self.coco.imgs.keys())))

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
	"""
	Pad the captions to have equal (maxiimal) length

	Returns:
		images: shape (batch_size, 3, 224, 224)
		captions: shape (batch_size, padded_length)
		lengths: valid lengths for each padded captions shape (batch_size, )
	"""
	data.sort(key=lambda x: len(x[1]), reverse=True)

	images, captions = zip(*data)
	images = torch.stack(images, 0)
	lengths = [len(cap) for cap in captions]
	# important to initilize as zero <pad>
	targets = torch.zeros(len(captions), max(lengths)).long()
	for i, cap in enumerate(captions):
		end = lengths[i]
		targets[i, :end] = cap[:end]

	return images, targets, lengths



def generate_test_entries(valid_annFile, root="./data/coco/annotations", 
										new_valid_filename="captions_val2014_reserved.json",
										new_test_filename="captions_test2014_reserved.json"):
	"""
	reserves 4k images from validation as test
	"""
	with open(valid_annFile, "r") as f:
		test_file = json.load(f)

	coco = COCO(valid_annFile)
	
	num_of_imgs = len(test_file["images"])

	print("Original: %d val images, %d val annotations"%(num_of_imgs, len(test_file["annotations"])))

	test_dict = {'images':[], 'annotations':[], 'info': test_file["info"], 'licenses': test_file["licenses"]}
	valid_dict = {'images':[], 'annotations':[], 'info': test_file["info"], 'licenses': test_file["licenses"]}
	test_ids_idx = np.random.choice(num_of_imgs, 4000, replace=False)
	test_ids = np.array(list(coco.imgs.keys()))[test_ids_idx]
	
	# store the images entries
	for i, image_info in enumerate(test_file["images"]):
		id = image_info["id"]
		if id in test_ids:
			# test_ids.append(id)
			test_dict["images"].append(image_info)
		else:
			valid_dict["images"].append(image_info)

	# store the annotations entries
	for i, annotations in enumerate(test_file["annotations"]):
		id = annotations["image_id"]
		if id in test_ids:
			test_dict["annotations"].append(annotations)
		else:
			valid_dict["annotations"].append(annotations)
	
	print("Saving %d val images, %d val annotations" % (len(valid_dict["images"]), len(valid_dict["annotations"])))
	with open(os.path.join(root, new_valid_filename), "w") as f:
		json.dump(valid_dict, f)

	print("Saving %d test images (should be 4000) %d test annotations" % (len(test_dict["images"]), len(test_dict["annotations"])))
	with open(os.path.join(root, new_test_filename), "w") as f:
		json.dump(test_dict, f)


def get_vocab():
	with open("./data/coco/vocab.pkl", 'rb') as f:
		vocab = pickle.load(f)
	return vocab

def get_data_loader(mode, transform, vocab, batch_size=4, shuffle=True, num_workers=0):
	""" 
	Returns Data loader for custom coco dataset

	Params:
		# root:    	./data/coco/[train2014 | val2014]
		# annFile: 	./data/coco/annotations[captions_train2014.json | captions_val2014_reserved.json 
		# 								| captions_test2014_reserved.json]
		mode:		[train | val | test]
		vocab:   	loaded file from ./data/coco/vocab.pkl
		transform: 	pytorch transformer
		batch_size: num of images in a batch [default:4]
		shuffle:	shuffle or not [default: true]
		num_workers:thread used for dataloader [default:0]
	"""
	assert(mode in ["train", "val", "test"])

	root = "./data/coco/train2014" if mode == "train" else "./data/coco/val2014"

	annFile = "./data/coco/annotations/captions_train2014.json" if mode == "train" else "./data/coco/annotations/captions_" + mode + "2014_reserved.json"

	dataset = CocoDataset(root=root,
					  annFile=annFile,
					  vocab=vocab,
					  transform=transform)

	data_loader = torch.utils.data.DataLoader(dataset=dataset, 
										   batch_size=batch_size,
										   shuffle=shuffle,
										   num_workers=num_workers,
										   collate_fn=collate_fn,
										   )
	return data_loader


def main(args):
	# generate test images
	print("Createing annotations json for splited val and test")
	generate_test_entries(args.json)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--json', type=str, default="./data/coco/annotations/captions_val2014.json", help="path for val annoations")
	args = parser.parse_args()
	main(args)

