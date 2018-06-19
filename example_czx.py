from builtins import super
import random

import torch
from torch import nn, optim
from torch.distributions.normal import Normal

from vhe import VHE, DataLoader, Factors


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
		mu_z = self.mu_z(inputs[0])
		sigma_z = self.sigma_z(inputs[0])
		dist = Normal(mu_z, sigma_z)
		if z is None: z = dist.rsample()
		return z, dist.log_prob(z).sum(dim=1)

encoder = Factors(c=Qc(), z=Qz())
decoder = Px()
vhe = VHE(encoder, decoder)

# Generate dataset
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


# Training
optimiser = optim.Adam(vhe.parameters(), lr=1e-3)
for epoch in range(1,11):
	for batch in data_loader:
		optimiser.zero_grad()
		score, kl = vhe.score(inputs=batch.inputs, sizes=batch.sizes, x=batch.target, return_kl=True)
		(-score).backward()
		optimiser.step()
	print("Epoch %d Score %3.3f KLc %3.3f KLz %3.3f" % (epoch, score.item(), kl.c.item(), kl.z.item()))

for mu in [-1, 0, 1]:
	test_D = [mu + 0.1*torch.randn(1,x_dim) for _ in range(n_inputs)]
	print("\nPosterior predictive for", test_D)
	print(vhe.sample(inputs={"c":test_D}).x)
