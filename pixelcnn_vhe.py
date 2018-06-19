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

from vhe import VHE, DataLoader, Factors

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
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
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
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

elif 'omni' in args.dataset :

    train_loader = torch.utils.data.DataLoader(datasets.Omniglot(args.data_dir, download=True, 
                        background=True, transform=omni_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(datasets.Omniglot(args.data_dir, download=True, 
                        background=False, transform=omni_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)
    
    loss_op   = TODO #lambda real, fake : discretized_mix_logistic_loss_1d(real, fake) #use binary cross-entropy loss
    sample_op = TODO #lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix) #sample bernoulli or whatev

else :
    raise Exception('{} dataset not in {mnist, cifar10, omniglot}'.format(args.dataset))
#######end pixelcnn options #########





x_dim = 5
c_dim = 10
z_dim = 10
h_dim = 10





#a pixelcnn px
class Px(nn.Module):
	def __init__(self):
		super().__init__()
		
		model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix, latent_dims=latent_dims)

		self.model = model.cuda()

	def sample(model, latents=None): 
		assert latents is not None
	    model.train(False)
	    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
	    data = data.cuda()
	    for i in range(obs[1]):
	        for j in range(obs[2]):
	            data_v = Variable(data, volatile=True)
	            out   = model(data_v, sample=True, latents=latents)
	            out_sample = sample_op(out) #TODO: write this fn (I think Luke's job?)
	            data[:, :, i, j] = out_sample.data[:, :, i, j]
	    return data, out #the last output form the model should be the dist

	def forward(self, c, z, x=None):

		if x is None: 
			x, dist = self.sample(self.model, latents=(c,z))
			return x, -loss_op(x, dist)/batch_size #TODO: loss_op, luke
		else:
			#return x and distribution (or is it a loss?)
			return x, -loss_op(x, self.model(x, latents=(c,z),  sample=False))/batch_size


class Qc(nn.Module):
	def __init__(self):
		super().__init__()
		self.enc = nn.Linear(x_dim, h_dim)
		self.mu_c = nn.Linear(h_dim, c_dim)
		self.sigma_c = nn.Sequential(nn.Linear(h_dim, c_dim), nn.Softplus())

	def forward(self, inputs, c=None):	
		embeddings = [self.enc(x) for x in inputs]
		mean_embedding = sum(embeddings)/len(embeddings)
		mu_c = self.mu_c(mean_embedding)
		sigma_c = self.sigma_c(mean_embedding)
		dist = Normal(mu_c, sigma_c)
		if c is None: c = dist.rsample()
		return c, dist.log_prob(c).sum(dim=1)

class Qz(nn.Module):
	def __init__(self):
		super().__init__()
		self.mu_z = nn.Linear(x_dim, z_dim)
		self.sigma_z = nn.Sequential(nn.Linear(x_dim, z_dim), nn.Softplus())

	def forward(self, inputs, c, z=None):	
		mu_z = self.mu_z(inputs[0])
		sigma_z = self.sigma_z(inputs[0])
		dist = Normal(mu_z, sigma_z)
		if z is None: z = dist.rsample()
		return z, dist.log_prob(z).sum(dim=1)

encoder = Factors(c=Qc(), z=Qz())
decoder = Px()
vhe = VHE(encoder, decoder)




########## Generate dataset############
#TODO:
n = 0
classes = []
for i in range(1000):
	mu = torch.randn(1, x_dim)
	sigma = 0.1
	class_size = random.randint(10,20)
	classes.append(mu + sigma*torch.randn(class_size, x_dim))
data = torch.cat(classes)
class_labels = [i for i in range(len(classes)) for j in range(len(classes[i]))] 


# Training
batch_size = 100
n_inputs = 1
data_loader = DataLoader(data=data, c=class_labels, z=range(len(data)),
		batch_size=batch_size, n_inputs=n_inputs)
############



# Training
optimiser = optim.Adam(vhe.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimiser, step_size=1, gamma=args.lr_decay)
for epoch in range(1,11):
	for batch in data_loader:
		optimiser.zero_grad()
		score, kl = vhe.score(inputs=batch.inputs, sizes=batch.sizes, x=batch.target, return_kl=True)
		(-score).backward() #TODO: Pixelcnn uses loss, and doesn't use negative ...
		optimiser.step()
	print("Epoch %d Score %3.3f KLc %3.3f KLz %3.3f" % (epoch, score.item(), kl.c.item(), kl.z.item()))

	#may not want this, but can keep:
	scheduler.step()


for mu in [-1, 0, 1]:
	test_D = [mu + 0.1*torch.randn(1,x_dim) for _ in range(n_inputs)]
	print("\nPosterior predictive for", test_D)
	print(vhe.sample(inputs={"c":test_D}).x)