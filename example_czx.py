from builtins import super
import random
import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions.normal import Normal

from vhe import VHE, NormalPrior, factors


# Model
x_dim = 5
c_dim = 10
z_dim = 10
h_dim = 10

class Px(nn.Module):
	def __init__(self):
		super().__init__()
		self.mu = nn.Linear(c_dim + z_dim, x_dim)
		self.sigma = nn.Sequential(nn.Linear(c_dim + z_dim, x_dim), nn.Softplus())

	def forward(self, c, z, x=None):
		cz = torch.cat([c,z], dim=1)
		dist = Normal(self.mu(cz), self.sigma(cz))
		if x is None: x = dist.rsample()
		return x, dist.log_prob(x).sum(dim=1)

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
		mu_z = self.mu_z(x)
		sigma_z = self.sigma_z(x)
		dist = Normal(mu_z, sigma_z)
		if z is None: z = dist.rsample()
		return z, dist.log_prob(z).sum(dim=1)

prior = factors(c=NormalPrior(), z=NormalPrior(), x=Px())
encoder = factors(c=Qc(), z=Qz())
vhe = VHE(prior, encoder)

# Generate dataset
n = 0
classes = []
class_idxs = []
for i in range(100):
	mu = torch.randn(1, x_dim)
	sigma = 0.1
	class_size = random.randint(10,20)
	classes.append(mu + sigma*torch.randn(class_size, x_dim))
	class_idxs.append(list(range(n, n+class_size)))
	n += class_size
data = torch.cat(classes)

def makeBatch(batch_size, n_inputs):
	classes = np.random.choice(range(len(class_idxs)), size=batch_size, p=[len(x)/n for x in class_idxs])
	elems = [np.random.choice(class_idxs[j], size=n_inputs+1) for j in classes]
	x_idx = torch.LongTensor([e[0] for e in elems])
	D_idx = [torch.LongTensor([e[k] for e in elems]) for k in range(1, n_inputs+1)]
	x = Variable(torch.index_select(data, 0, x_idx))
	D = [Variable(torch.index_select(data, 0, d_idx)) for d_idx in D_idx]
	return D, x

# Training
batch_size = 100
n_inputs = 1

optimiser = optim.Adam(vhe.parameters(), lr=1e-3)
for i in range(1000):
	optimiser.zero_grad()
	D, x = makeBatch(batch_size, n_inputs)
	score, kl = vhe.score(inputs = {"c":D, "z":[x]},
					  sizes = {"c":100, "z":1},
					  x=x,
					  return_kl=True)
	(-score).backward()
	optimiser.step()
	if i%100==0: print("Iteration %d Score %3.3f KLc %3.3f KLz %3.3f" % (i, score.item(), kl.c.item(), kl.z.item()))

for mu in [-1, 0, 1]:
	test_D = [mu + 0.1*torch.randn(1,x_dim) for _ in range(n_inputs)]
	print("\nPosterior predictive for", test_D)
	print(vhe.sample(inputs={"c":test_D}).x)
