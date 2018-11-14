import os
import torch
from . import mscoco


def get_data_loader(dataset="mscoco"):
	"""
	return the given dataset loader
	Params:
		dataset: [default: mscoco | flickr30k | flickr8k | pascal | sbu]
	"""
	data_loader = getattr(eval(dataset), "get_data_loader")
	return data_loader
