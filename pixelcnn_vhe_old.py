#PixelCNN:
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from pixelcnn.utils import * 
from pixelcnn.model import * 
from PIL import Image

#VHE:
from builtins import super
import random

import torch
from torch import nn, optim
from torch.distributions.normal import Normal

from vhe import VHE, DataLoader, Factors, Result

#######pixelcnn options #########
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='mnist', help='Can be either cifar|mnist')
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=10,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=None,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-sm', '--nr_softmax_bins', type=int, default=None,
                    help='Number of softmax bins (use instead of nr_logistic_mix)')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=64,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
args = parser.parse_args()


if args.nr_logistic_mix is None and args.nr_softmax_bins is None:
	args.nr_logistic_mix = 10

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(args.lr, args.nr_resnet, args.nr_filters)
assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)

sample_batch_size = 25
obs = (1, 28, 28) if 'mnist' in args.dataset or 'omni' in args.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
flip = lambda x : - x
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
resizing = lambda x: x.resize((28,28))
omni_transforms = transforms.Compose([resizing, transforms.ToTensor(), rescaling, flip])

if 'mnist' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                        train=True, transform=ds_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    if args.nr_logistic_mix:
        loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
        sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)
    else:
        loss_op   = lambda real, fake : softmax_loss_1d(real, fake)
        sample_op = lambda x : sample_from_softmax_1d(x)


elif 'cifar' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    if args.nr_logistic_mix:
        loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
        sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
    else:
        raise NotImplementedError("No 3D Softmax")

elif 'omni' in args.dataset :

    train_loader = torch.utils.data.DataLoader(datasets.Omniglot(args.data_dir, download=True, 
                        background=True, transform=omni_transforms), batch_size=1, 
                            shuffle=True, **kwargs)

    #d = datasets.Omniglot(args.data_dir, download=True, 
    #                   background=True, transform=omni_transforms)
    
    test_loader = torch.utils.data.DataLoader(datasets.Omniglot(args.data_dir, download=True, 
                        background=False, transform=omni_transforms), batch_size=1, 
                            shuffle=True, **kwargs)
    
    if args.nr_logistic_mix:
        loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
        sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)
    else:
        loss_op   = lambda real, fake : softmax_loss_1d(real, fake)
        sample_op = lambda x : sample_from_softmax_1d(x)

else :
    raise Exception('{} dataset not in {mnist, cifar10, omniglot}'.format(args.dataset))
#######end pixelcnn options #########





x_dim = 5
c_dim = 10 #28, 28
z_dim = 10 #28, 28
h_dim = 10


#a pixelcnn px
class Px(nn.Module):
	def __init__(self):
		super().__init__()
        #self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        #self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
		#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
		kernel=5

		self.pad = nn.ZeroPad2d((kernel - 1, 0, kernel - 1, 0))
		self.cond_conv_1 = nn.Conv2d(c_dim+z_dim, args.nr_filters * 2, kernel, stride=1, padding=0)
		self.cond_conv_2 = nn.Conv2d(args.nr_filters * 2, args.nr_filters * 2, kernel, stride=2, padding=0)
		self.cond_conv_3 = nn.Conv2d(args.nr_filters * 2, args.nr_filters * 2, kernel, stride=2, padding=0)
		
		self.model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix,
            nr_softmax_bins=args.nr_softmax_bins)

	def sample(model, cond_blocks=None): 
		assert latents is not None
		model.train(False)
		data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
		data = data.cuda()
		for i in range(obs[1]):
			for j in range(obs[2]):
				data_v = Variable(data, volatile=True)
				out   = model(data_v, sample=True, cond_blocks=cond_blocks)
				out_sample = sample_op(out)
				data[:, :, i, j] = out_sample.data[:, :, i, j]
		return data, out #the last output form the model should be the dist

	def forward(self, c, z, x=None):

		#make z and c into a big volume, d x 28 x 28
		#TODO: make cond_blocks
		#assume c_dim, z_dim, etc

		cond = torch.cat((c,z), 1)

		cond_blocks = {}
		cond_blocks[(28, 28)] = self.cond_conv_1(self.pad(cond))
		cond_blocks[(14, 14)] = self.cond_conv_2(self.pad(cond_blocks[(28, 28)]))
		cond_blocks[(7, 7)] = self.cond_conv_3(self.pad(cond_blocks[(14, 14)]))
				

		if x is None: 
			x, dist = self.sample(self.model, cond_blocks=cond_blocks)
			return Result(x, -loss_op(x, dist)/x.size(0) )# batch_size loss_op, luke
		else:
			#return x and distribution (or is it a loss?)
			return Result(x, -loss_op(x, self.model(x, cond_blocks=cond_blocks, sample=False))/x.size(0)) #batch_size

total_c_dim = c_dim*28*28

class Qc(nn.Module):
	def __init__(self):
		super(Qc, self).__init__()
		self.kernel = 5
		self.pad = nn.ZeroPad2d((self.kernel - 1, 0, self.kernel - 1, 0))
		self.embc = nn.Sequential(self.pad, nn.Conv2d(1, 10, self.kernel, stride=1, padding=0))
		self.conv_mu = nn.Sequential(self.pad, nn.Conv2d(10, 10, self.kernel, stride=1, padding=0))
		self.conv_sigma = nn.Sequential(self.pad, nn.Conv2d(10, 10, self.kernel, stride=1, padding=0), nn.Softplus())

	def forward(self, inputs, c=None):	
		emb = [self.embc(x) for x in inputs]
		emb = sum(emb)/len(emb)
		mu = self.conv_mu(emb)
		sigma = self.conv_sigma(emb)

		dist = Normal(mu, sigma)
		if c is None: c = dist.rsample()
		#print(dist.log_prob(c).sum(dim=1))
		return Result(c, dist.log_prob(c).sum(dim=1).sum(dim=1).sum(dim=1))

class Qz(nn.Module):
	def __init__(self):
		super(Qz,self).__init__()
		self.kernel = 5
		self.pad = nn.ZeroPad2d((self.kernel - 1, 0, self.kernel - 1, 0))
		self.embc = nn.Sequential(self.pad, nn.Conv2d(1, 10, self.kernel, stride=1, padding=0))
		self.conv_mu = nn.Sequential(self.pad, nn.Conv2d(10, 10, self.kernel, stride=1, padding=0))
		self.conv_sigma = nn.Sequential(self.pad, nn.Conv2d(10, 10, self.kernel, stride=1, padding=0), nn.Softplus())

		self.mu_z = nn.Linear(x_dim, z_dim)
		self.sigma_z = nn.Sequential(nn.Linear(x_dim, z_dim), nn.Softplus())

	def forward(self, inputs, c, z=None):
		emb = [self.embc(x) for x in inputs]
		emb = sum(emb)/len(emb)
		mu = self.conv_mu(emb)
		sigma = self.conv_sigma(emb)
		dist = Normal(mu, sigma)
		if z is None: z = dist.rsample()
		return Result(z, dist.log_prob(z).sum(dim=1).sum(dim=1).sum(dim=1))

encoder = Factors(c=Qc(), z=Qz())
decoder = Px()
vhe = VHE(encoder, decoder)
vhe = vhe.cuda()
print("created vhe")
print("number of parameters is",sum(p.numel() for p in vhe.parameters() if p.requires_grad))


########## Generate dataset############
#TODO:

"""
n = 0
classes = []
for i in range(1000):
	mu = torch.randn(1, x_dim)
	sigma = 0.1
	class_size = random.randint(10,20)
	classes.append(mu + sigma*torch.randn(class_size, x_dim))
data = torch.cat(classes)
class_labels = [i for i in range(len(classes)) for j in range(len(classes[i]))] 
"""

data, class_labels = zip(*[[img, label] for img, label in train_loader]) 

data = torch.cat(data)

#class_labels = class_labels
print("data shape", data.shape)
print("class_labels",class_labels[0])

# Training
batch_size = args.batch_size
n_inputs = 4
data_loader = DataLoader(data=data, c=class_labels, z=range(len(data)),
		batch_size=batch_size, n_inputs=n_inputs)
############bat



# Training
print("started training")

optimiser = optim.Adam(vhe.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimiser, step_size=1, gamma=args.lr_decay)
for epoch in range(1,11):
	batchnum = 0
	for batch in data_loader:
		inputs = {k:[item.cuda() for item in v]for k,v in batch.inputs.items()}

		sizes = {k:v.cuda() for k,v in batch.sizes.items()}
		target = batch.target.cuda()

		optimiser.zero_grad()
		score, kl = vhe.score(inputs=inputs, sizes=sizes, x=target, return_kl=True)
		(-score).backward() 
		optimiser.step()
		batchnum = batchnum + 1
		print("Batch %d Score %3.3f KLc %3.3f KLz %3.3f" % (batchnum, score.item(), kl.c.item(), kl.z.item()))
	print("Epoch %d Score %3.3f KLc %3.3f KLz %3.3f" % (epoch, score.item(), kl.c.item(), kl.z.item()))

	#may not want this, but can keep:
	scheduler.step()


for mu in [-1, 0, 1]:
	test_D = [mu + 0.1*torch.randn(1,x_dim) for _ in range(n_inputs)]
	print("\nPosterior predictive for", test_D)
	print(vhe.sample(inputs={"c":test_D}).x)