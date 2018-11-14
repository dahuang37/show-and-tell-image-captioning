import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
import torch.optim as optim
import sys
import time
import argparse
from utils import *
from dataset import * 
import json
import sklearn.neighbors as neighbors
#from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=int, default='0', help="pretrain model or not")
parser.add_argument('--lr', type=float, default=0.001, help="leanring rate")
parser.add_argument('--bs', type=int, default=4, help="batch size")
opt = parser.parse_args()

size = 224

train_transform_config = [
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(), 
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	]

train_transform_config = [transforms.Resize((size, size))] + train_transform_config
train_transform = transforms.Compose(train_transform_config)


test_transform_config = [
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	]

test_transform_config = [transforms.Resize((size, size))] + test_transform_config
test_transform = transforms.Compose(test_transform_config)




test_set = TinyImageNetDataset("../data/tiny-imagenet-200/", \
								train=False, \
								transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set,\
											batch_size=opt.bs,\
											shuffle=False,\
											num_workers=0) 

loss_tracker = LossTracker()

def adjust_learning_rate(optimizer, epoch, k=0.01):
	'''
	performs exponential decay with rate of 0.01
	'''
	lr = opt.lr * np.exp(-k*epoch)
	print("Learning rate: %f" % (lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def train(net, criterion, optimizer, epoch, init_e=0, save_epoch=3):
	'''
	1. gets the dataset so every time we got a different triplet
	2. foward pass the neural network and backprop w.r.t. tripletmarginloss 
	3. 
	'''
	e = init_e
	
	while(e < epoch):
		net.train()

		print("reshuffle train set")
		train_set = TinyImageNetDataset("../data/tiny-imagenet-200/", \
								train=True, \
								transform=train_transform, debug=False)

		train_loader = torch.utils.data.DataLoader(train_set,\
												batch_size=opt.bs,\
												shuffle=True,\
												num_workers=4)
		
		embedding_saver = []
		embedding_meta = {"label": [], "image_path": []}

		e += 1
		running_loss = 0.
		
		adjust_learning_rate(optimizer, e-1)

		epoch_start_time = time.time()
		for i, data in enumerate(train_loader):

			query_image, positive_image, negative_image, label = data['image'], data['positive'], data['negative'], data['label']

			query_image, positive_image, negative_image = query_image.to(device), positive_image.to(device), negative_image.to(device)

			optimizer.zero_grad()

			query_output = net(query_image)
			positive_output = net(positive_image)
			negative_output = net(negative_image)
			
			loss = criterion(query_output, positive_output, negative_output)
			loss.backward()

			optimizer.step()

			# loss takes average across the size of batches
			running_loss += loss.item()

			if e % save_epoch==0:
				'''
				saves every s epoch
				'''
				embedding_saver.append(query_output.cpu())
				embedding_meta["label"] += data['label']
				embedding_meta["image_path"] += data["image_path"]
			
			progress_bar(i, len(train_loader), "Loss: %f"%((running_loss/(i+1))))
		
		epoch_loss = running_loss/(i+1)
		loss_tracker.append(epoch_loss)
		
		print("Epoch: %d | Loss: %.3f" % (e, epoch_loss))

		if e % save_epoch == 0:
			# saves model state 
			torch.save(net.state_dict(), "checkpoint_"+str(e)+".pt")
			# saves training embedding
			embedding = torch.cat(embedding_saver)
			torch.save(embedding, "embedding_"+str(e)+".pth")
			# saves meta data: labels, image paths of embedding
			with open("embedding_meta_"+str(e)+".json", "w") as f:
				json.dump(embedding_meta, f)
			loss_tracker.save_loss("training_loss_"+str(e)+".json")
			# reset the variables to save memories
			embedding = None
			embedding_saver = None
			embedding_meta = None
			
		
	print("Finishing training, saving loss")
	
	
		
def test(net, mode="test"):
	"""
	1. compute the embedding of the test image
	2. compute the eculidean distance bewteen test embedding and all the embedding in the training set
	3. ...
	support demo, which saves 5 unique embedding for visualization
	"""
	net.eval()
	
	embedding_saver = []
	embedding_meta_saver = {"label": [], "image_path": []}

	with torch.no_grad():
		for i, data in enumerate(test_loader):
			if demo and len(embedding_meta_saver["label"]) >= 5:
				break
			image, label = data['image'], data['label']
			image = image.to(device)
			output = net(image)

			if demo and label[0] in embedding_meta_saver["label"]:
				continue

			# saves the test embedding on cpu
			embedding_saver.append(output.cpu())
			# saves the label and image paths
			embedding_meta_saver["label"] += label
			embedding_meta_saver["image_path"] += data['image_path']
			
			progress_bar(i, len(test_loader))

		torch.save(torch.cat(embedding_saver), "embedding_"+mode+".pth")
		with open("embedding_meta_"+mode+".json", "w") as f:
			json.dump(embedding_meta_saver, f)

		print('Total time: ', time.time()-time_init)



def main():
	print(opt)
	lr = opt.lr
	net = models.squeezenet1_1(pretrained=True)
	'''
	counter = 1
	for name, param in net.named_parameters():
	 	print("%d: %s" %(counter, name) )
	 	if counter >= 31:
	 		break
	 	param.requires_grad = False
	 	counter += 1
	
	num_ftrs = net.fc.in_features
	net.fc = nn.Linear(num_ftrs, 4096)
	net.load_state_dict(torch.load("./src/checkpoint_15.pt"))
	'''
	net.classifier._modules["1"] = nn.Conv2d(512, 256, kernel_size=(1,1))
	net.num_classes = 256
	net = net.to(device)
	optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9)
	criterion = nn.TripletMarginLoss(margin=0)
	
	# summary(net,input_size=(3, 224, 224))
	# print(os.listdir())
	print("training now")
	train(net, criterion, optimizer, 50)
	
	
	print("starting to test")
	test(net, mode="test")
	
	


if __name__ == "__main__":
	main()
